use crate::numbers::*;

use std::ops;
use std::ops::Index;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc::channel;
use std::sync::mpsc::Sender;
use std::sync::mpsc::Receiver;
use std::thread;

pub trait Arrays {
    fn new(self, backward_op: Option<Box<dyn Fn(&Vec<Array>, Array) -> Array + Send + Sync>>) -> Array;
}

impl Arrays for Vec<Array> {
    fn new(self, backward_op: Option<Box<dyn Fn(&Vec<Array>, Array) -> Array + Send + Sync>>) -> Array {
        let is_dimensions_valid = match self.split_first() {
            Some((first, elements)) => elements.iter().all(|item| *item.dimensions == *first.dimensions),
            None => true,
        };

        if !is_dimensions_valid {
            panic!("error: invalid dimensions supplied");
        }

        let mut dimensions = vec![self.len()];
        dimensions.append(&mut self.first().unwrap().dimensions.clone());

        let values = self.into_iter().map(|array| array.values).flatten().collect::<Vec<Float>>();

        Array { dimensions, values, children: Vec::new(), consumer_count: 0, backward_op, tx: None }
    }
}

impl Arrays for Vec<Float> {
    fn new(self, backward_op: Option<Box<dyn Fn(&Vec<Array>, Array) -> Array + Send + Sync>>) -> Array {
        Arrays::new((vec![self.len()], self), backward_op)
    }
}

impl<'v> Arrays for (Vec<usize>, Vec<Float>) {
    fn new(self, backward_op: Option<Box<dyn Fn(&Vec<Array>, Array) -> Array + Send + Sync>>) -> Array {
        let (dimensions, values) = self;
        Array { dimensions, values, children: Vec::new(), consumer_count: 0, backward_op, tx: None }
    }
}

pub struct Array {
    dimensions: Vec<usize>,
    values: Vec<Float>,
    children: Vec<Array>,
    consumer_count: usize,
    backward_op: Option<Box<dyn Fn(&Vec<Array>, Array) -> Array + Send + Sync>>,
    tx: Option<Mutex<Sender<Arc<Array>>>>,
}
 
#[macro_export]
macro_rules! array {
    ( $( $x:expr ),* ) => {
        {
            let mut values = Vec::new();

            $(
                values.push($x);
            )*
 
            Arrays::new(values, None)
        }
    };
}

impl Array {
    fn dimensions(&self) -> &Vec<usize> {
        &self.dimensions
    }

    fn values(&self) -> &Vec<Float> {
        &self.values
    }

    fn propagate_consumers(&mut self) {
        for child in &mut self.children {
            child.consumer_count += 1;
            child.propagate_consumers();
        }
    }

    fn await_results(mut self, rx: Receiver<Arc<Array>>) {
        if self.consumer_count <= 0 {
            panic!("error: cannot await results from end node");
        }

        let mut delta = vec![1.0; self.values.len()];
        let sum = |acc: &mut Vec<Float>, x: &Vec<Float>| {
            acc.iter_mut().zip(x).for_each(|(s, x)| *s += *x);
        };

        while self.consumer_count > 0 {
            let received = rx.recv().unwrap();
            self.consumer_count -= 1;
            sum(&mut delta, &received.values);
        }

        let delta = Arrays::new((self.dimensions.clone(), delta), None);
        self.backward(Some(delta));
    }

    fn backward(mut self, delta: Option<Array>) {
        self.propagate_consumers();

        // TODO note that consumer_count should be reset each time, unless we want to keep the graph
        let delta = match delta {
            Some(x) => x,
            None => Arrays::new((self.dimensions.clone(), vec![1.0; self.values.len()]), None),
        };

        match &self.backward_op {
            Some(x) => {
                self.consumer_count = 0;
                let delta = Arc::new((*x)(&self.children, delta));
                // start a new thread which will wait on all consumers
                for mut child in self.children {
                    match child.tx {
                        Some(x) => {
                            x.lock().unwrap().send(Arc::clone(&delta)).unwrap();
                        },
                        None => {
                            let (tx, rx) = channel();
                            child.tx = Some(Mutex::new(tx));
                            thread::spawn(move|| {
                                child.await_results(rx);
                            });
                        },
                    }
                }
            },
            None => panic!("error: operation is not differentiable"),
        }
    }
}

impl Index<Vec<usize>> for Array {
    type Output = Float;
 
    fn index(&self, indices: Vec<usize>) -> &Self::Output {
        let is_indices_valid = indices.len() == self.dimensions.len()
            && !indices.iter().zip(&self.dimensions).filter(|&(i, d)| *i >= *d).peekable().peek().is_some();

        if !is_indices_valid {
            panic!("error: invalid indices supplied")
        } 
 
        let mut iter = indices.iter();
        let first = iter.next().unwrap();
 
        // dimensions will always have at least one element
        let index: usize = iter.zip(self.dimensions.iter().skip(1)).fold(*first, |acc, (i, d)| acc * d + i);
        &self.values[index]
    }
}

fn add_values(a: &Vec<Float>, b: &Vec<Float>) -> Vec<Float> {
    a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<Float>>()
}

fn mul_values(a: &Vec<Float>, b: &Vec<Float>) -> Vec<Float> {
    a.iter().zip(b).map(|(x, y)| x * y).collect::<Vec<Float>>()
}

impl<'a, 'b> ops::Add<&'b Array> for &'a Array {
    type Output = Array;

    #[inline]
    fn add(self, other: &Array) -> Array {
        let backward_op = Box::new(|_: &Vec<Array>, x: Array| array![Arrays::new(x.values.clone(), None), Arrays::new(x.values.clone(), None)]);
        Arrays::new((self.dimensions.clone(), add_values(&self.values, &other.values)), Some(backward_op))
    }
}

impl<'a, 'b> ops::Mul<&'b Array> for &'a Array {
    type Output = Array;

    #[inline]
    fn mul(self, other: &Array) -> Array {
        let backward_op = Box::new(|c: &Vec<Array>, x: Array| array![Arrays::new((x.dimensions.clone(), mul_values(&c[1].values, &x.values)), None)]);
        Arrays::new((self.dimensions.clone(), mul_values(&self.values, &other.values)), Some(backward_op))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new() {
        let matrix = array![array![
            array![0.0], array![1.0]
        ],
        array![
            array![2.0], array![3.0]
        ],
        array![
            array![4.0], array![5.0]
        ]];
        
        assert_eq!(matrix.dimensions, vec![3, 2, 1]);
        assert_eq!(matrix.values, (0..=5).map(|x| x as Float).collect::<Vec<Float>>());
    }
    
    #[test]
    #[should_panic]
    fn test_invalid_dimensions() { 
        array![array![ 
            array![0.0], array![1.0]
        ],
        array![
            array![2.0, 3.0], array![4.0]
        ]];                              
    }
    
    #[test]
    fn test_access() {
        let matrix = array![array![
            array![0.0, 1.0, 2.0], array![3.0, 4.0, 5.0]
        ],
        array![
            array![6.0, 7.0, 8.0], array![9.0, 10.0, 11.0]
        ],
        array![
            array![12.0, 13.0, 14.0], array![15.0, 16.0, 17.0]
        ]];

        assert_eq!(matrix[vec![1, 1, 2]], 11.0);
    }

    #[test]
    fn test_arithmetic() {
        let a = array![array![
            array![0.0, 1.0], array![2.0, 3.0]
        ],
        array![
            array![4.0, 5.0], array![6.0, 7.0]
        ]];

        let b = array![array![
            array![2.0, 4.0], array![6.0, 8.0]
        ],
        array![
            array![10.0, 12.0], array![14.0, 16.0]
        ]];

        let sum_expect = array![array![
            array![2.0, 5.0], array![8.0, 11.0]
        ],
        array![
            array![14.0, 17.0], array![20.0, 23.0]
        ]];

        let product_expect = array![array![
            array![0.0, 4.0], array![12.0, 24.0]
        ],
        array![
            array![40.0, 60.0], array![84.0, 112.0]
        ]];

        let sum = &a + &b;
        let product = &a * &b;

        assert_eq!(sum.dimensions, sum_expect.dimensions);
        assert_eq!(sum.dimensions, product_expect.dimensions);

        assert_eq!(sum.values, sum_expect.values);
        assert_eq!(product.values, product_expect.values);
    }

    #[test]
    fn test_propagate_consumers() {
    }

    #[test]
    fn test_backward() {
    }
}

