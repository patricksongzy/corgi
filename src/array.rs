use crate::numbers::*;

use std::ops;
use std::ops::Index;

use std::mem;

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::mpsc::channel;
use std::sync::mpsc::Sender;
use std::sync::mpsc::Receiver;
use std::thread;

pub trait Arrays {
    fn new(self, backward_op: Option<Arc<dyn Fn(&Vec<Array>, Array) -> Vec<Array> + Send + Sync>>) -> Array;
}

impl Arrays for Vec<Array> {
    fn new(self, backward_op: Option<Arc<dyn Fn(&Vec<Array>, Array) -> Vec<Array> + Send + Sync>>) -> Array {
        let is_dimensions_valid = match self.split_first() {
            Some((first, elements)) => elements.iter().all(|item| *item.dimensions == *first.dimensions),
            None => true,
        };

        if !is_dimensions_valid {
            panic!("error: invalid dimensions supplied");
        }

        let mut dimensions = vec![self.len()];
        dimensions.append(&mut (*self.first().unwrap().dimensions).clone());

        // take ownership if possible, but clone otherwise
        let values = self.into_iter().map(|array| Arc::try_unwrap(array.values).unwrap_or_else(|x| (*x).clone())).flatten().collect::<Vec<Float>>();

        Arrays::new((Arc::new(dimensions), Arc::new(values)), backward_op)
    }
}

impl Arrays for Vec<Float> {
    fn new(self, backward_op: Option<Arc<dyn Fn(&Vec<Array>, Array) -> Vec<Array> + Send + Sync>>) -> Array {
        Arrays::new((Arc::new(vec![self.len()]), Arc::new(self)), backward_op)
    }
}

impl<'v> Arrays for (Arc<Vec<usize>>, Arc<Vec<Float>>) {
    fn new(self, backward_op: Option<Arc<dyn Fn(&Vec<Array>, Array) -> Vec<Array> + Send + Sync>>) -> Array {
        let (dimensions, values) = self;
        Array { dimensions, values: values, children: Arc::new(Mutex::new(Vec::new())), consumer_count: Arc::new(Mutex::new(0)), backward_op, tx: Arc::new(Mutex::new(None)) }
    }
}

// TODO add poisoned flag, if Array has been modified
pub struct Array {
    dimensions: Arc<Vec<usize>>,
    values: Arc<Vec<Float>>,
    children: Arc<Mutex<Vec<Array>>>,
    consumer_count: Arc<Mutex<usize>>,
    backward_op: Option<Arc<dyn Fn(&Vec<Array>, Array) -> Vec<Array> + Send + Sync>>,
    tx: Arc<Mutex<Option<Sender<Array>>>>,
}
 
#[macro_export]
macro_rules! arr {
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
    // pub(crate) fn dimensions(&self) -> &Vec<usize> {
    //     &self.dimensions
    // }

    // pub(crate) fn values(&self) -> &Vec<Float> {
    //     &self.values
    // }

    fn with_children(mut self, children: Vec<Array>) -> Array {
        self.children = Arc::new(Mutex::new(children));
        self
    }

    fn propagate_consumers(&mut self) {
        for child in &mut *self.children.lock().unwrap() {
            *child.consumer_count.lock().unwrap() += 1;
            child.propagate_consumers();
        }
    }

    fn await_results(&mut self, rx: Receiver<Array>, delta: Array) {
        let mut consumer_count = self.consumer_count.lock().unwrap();

        if *consumer_count == 0 {
            panic!("error: cannot await results from end node");
        }

        let mut delta = Arc::try_unwrap(delta.values).unwrap_or_else(|x| (*x).clone());
        *consumer_count -= 1;
        let sum = |acc: &mut Vec<Float>, x: &Vec<Float>| {
            acc.iter_mut().zip(x).for_each(|(s, x)| *s += *x);
        };

        while *consumer_count > 0 {
            let received = rx.recv().unwrap();
            *consumer_count -= 1;
            sum(&mut delta, &received.values);
        }

        mem::drop(consumer_count);

        let delta = Arrays::new((Arc::clone(&self.dimensions), Arc::new(delta)), None);
        self.backward(Some(delta));
    }

    fn backward(&mut self, delta: Option<Array>) {
        let delta = match delta {
            Some(x) => x,
            None => {
                self.propagate_consumers();
                Arrays::new((Arc::clone(&self.dimensions), Arc::new(vec![1.0; self.values.len()])), None)
            },
        };

        match &self.backward_op {
            Some(x) => {
                let children_guard = self.children.lock().unwrap();
                let delta = (*x)(&children_guard, delta);
                let mut handles = Vec::new();
                // start a new thread which will wait on all consumers
                for (i, delta) in delta.into_iter().enumerate() {
                    let mut tx_guard = children_guard[i].tx.lock().unwrap();
                    match &*tx_guard {
                        Some(x) => {
                            x.send(delta).unwrap();
                        },
                        None => {
                            let mut child = children_guard[i].clone();

                            let (tx, rx) = channel();
                            *tx_guard = Some(tx);
                            handles.push(thread::spawn(move|| {
                                child.await_results(rx, delta);
                            }));
                        },
                    }
                }

                // wait for all threads to finish
                for handle in handles {
                    handle.join().unwrap();
                }
            },
            None => {
                if self.children.lock().unwrap().len() != 0 {
                    panic!("error: operation is not differentiable")
                }
            },
        }
    }
}

impl Clone for Array {
    fn clone(&self) -> Array {
        let backward_op = match &self.backward_op {
            Some(x) => Some(Arc::clone(&x)),
            None => None,
        };

        Array { dimensions: Arc::clone(&self.dimensions), values: Arc::clone(&self.values), children: Arc::clone(&self.children), consumer_count: Arc::clone(&self.consumer_count), backward_op, tx: Arc::clone(&self.tx) }
    }
}

impl Index<Vec<usize>> for Array {
    type Output = Float;
 
    fn index(&self, indices: Vec<usize>) -> &Self::Output {
        let is_indices_valid = indices.len() == self.dimensions.len()
            && !indices.iter().zip(&*self.dimensions).filter(|&(i, d)| *i >= *d).peekable().peek().is_some();

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
        let backward_op = Arc::new(|_: &Vec<Array>, x: Array| vec![Arrays::new((Arc::clone(&x.dimensions), Arc::clone(&x.values)), None); 2]);
        Arrays::new((Arc::clone(&self.dimensions), Arc::new(add_values(&self.values, &other.values))), Some(backward_op)).with_children(vec![self.clone(), other.clone()])
    }
}

impl<'a, 'b> ops::Mul<&'b Array> for &'a Array {
    type Output = Array;

    #[inline]
    fn mul(self, other: &Array) -> Array {
        let backward_op = Arc::new(|c: &Vec<Array>, x: Array| vec![Arrays::new((Arc::clone(&c[0].dimensions), Arc::new(mul_values(&c[1].values, &x.values))), None),
                                                                   Arrays::new((Arc::clone(&c[1].dimensions), Arc::new(mul_values(&c[0].values, &x.values))), None)]);
        Arrays::new((Arc::clone(&self.dimensions), Arc::new(mul_values(&self.values, &other.values))), Some(backward_op)).with_children(vec![self.clone(), other.clone()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new() {
        let matrix = arr![arr![
            arr![0.0], arr![1.0]
        ],
        arr![
            arr![2.0], arr![3.0]
        ],
        arr![
            arr![4.0], arr![5.0]
        ]];
        
        assert_eq!(*matrix.dimensions, vec![3, 2, 1]);
        assert_eq!(*matrix.values, (0..=5).map(|x| x as Float).collect::<Vec<Float>>());
    }

    #[test]
    fn test_new_clone() {
        let a = arr![1.0, 2.0, 3.0];
        let b = arr![a.clone()];
        let c = arr![a.clone(), a.clone()];

        assert_eq!(*b.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(*c.values, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }
    
    #[test]
    #[should_panic]
    fn test_invalid_dimensions() { 
        arr![arr![ 
            arr![0.0], arr![1.0]
        ],
        arr![
            arr![2.0, 3.0], arr![4.0]
        ]];                              
    }
    
    #[test]
    fn test_access() {
        let matrix = arr![arr![
            arr![0.0, 1.0, 2.0], arr![3.0, 4.0, 5.0]
        ],
        arr![
            arr![6.0, 7.0, 8.0], arr![9.0, 10.0, 11.0]
        ],
        arr![
            arr![12.0, 13.0, 14.0], arr![15.0, 16.0, 17.0]
        ]];

        assert_eq!(matrix[vec![1, 1, 2]], 11.0);
    }

    #[test]
    fn test_arithmetic() {
        let a = arr![arr![
            arr![0.0, 1.0], arr![2.0, 3.0]
        ],
        arr![
            arr![4.0, 5.0], arr![6.0, 7.0]
        ]];

        let b = arr![arr![
            arr![2.0, 4.0], arr![6.0, 8.0]
        ],
        arr![
            arr![10.0, 12.0], arr![14.0, 16.0]
        ]];

        let sum_expect = arr![arr![
            arr![2.0, 5.0], arr![8.0, 11.0]
        ],
        arr![
            arr![14.0, 17.0], arr![20.0, 23.0]
        ]];

        let product_expect = arr![arr![
            arr![0.0, 4.0], arr![12.0, 24.0]
        ],
        arr![
            arr![40.0, 60.0], arr![84.0, 112.0]
        ]];

        let sum = &a + &b;
        let product = &a * &b;

        assert_eq!(*sum.dimensions, *sum_expect.dimensions);
        assert_eq!(*sum.dimensions, *product_expect.dimensions);

        assert_eq!(*sum.values, *sum_expect.values);
        assert_eq!(*product.values, *product_expect.values);
    }

    #[test]
    fn test_propagate_consumers() {
        let a = arr![5.0];
        let b = arr![2.0];

        let product = &a * &b;
        let mut sum = &product + &a;

        sum.propagate_consumers();
        assert_eq!(*product.consumer_count.lock().unwrap(), 1);
        assert_eq!(*b.consumer_count.lock().unwrap(), 1);
        assert_eq!(*a.consumer_count.lock().unwrap(), 2);
    }

    #[test]
    fn test_backward_op() {
        let a = arr![5.0];
        let b = arr![2.0];

        let mut product = &a * &b;
        let result = (*product.backward_op.unwrap())(&vec![a, b], arr![1.0]);
        assert_eq!(result.len(), 2);
        assert_eq!(result.iter().map(|x| (*x.values).clone()).collect::<Vec<Vec<Float>>>(), vec![vec![2.0], vec![5.0]]);
    }

    #[test]
    fn test_backward_single() {
        let a = arr![5.0];
        let b = arr![2.0];

        let mut product = &a * &b;
        product.backward(None);
        // assert_eq!(*product.consumer_count.lock().unwrap(), 0);
        // assert_eq!(*b.consumer_count.lock().unwrap(), 0);
        // assert_eq!(*a.consumer_count.lock().unwrap(), 0);
    }

    #[test]
    fn test_backward_multi() {
        let a = arr![5.0, 2.0];
        let b = arr![6.0, 3.0];
        let c = &a * &b;
        let d = &c + &a;
        let mut e = &a * &d;
        e.backward(None);
        thread::sleep(std::time::Duration::from_millis(10));
    }
}

