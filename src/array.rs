#![macro_escape]

use crate::numbers::*;

use std::ops::Index;

pub trait Arrays {
    fn new (self) -> Array;
}

impl Arrays for Vec<Array> {
    fn new(self) -> Array {
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

        Array { dimensions, values }
    }
}

impl Arrays for Vec<Float> {
    fn new(self) -> Array {
        Array { dimensions: vec![self.len()], values: self }
    }
}

pub struct Array {
    pub dimensions: Vec<usize>,
    pub values: Vec<Float>,
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
 
#[macro_export]
macro_rules! array {
    ( $( $x:expr ),* ) => {
        {
            let mut values = Vec::new();

            $(
                values.push($x);
            )*
 
            Arrays::new(values)
        }
    };
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
        let matrix = array![array![ 
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
}

