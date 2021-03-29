// use crate::numbers::*;
// use crate::array::*;

// use std::ops;

// use std::sync::Arc;
// use std::sync::Mutex;
// use std::sync::mpsc;

// pub struct Tensor {
//     array: Arc<Mutex<Array>>,
// }

// #[macro_export]
// macro_rules! ts {
//     ( $( $x:expr ),* ) => {
//         let mut values = Vec::new();

//         $(
//             values.push($x);
//         )*

//         Tensor::new(Arc::new(Mutex::new(Arrays::new(values, None))))
//     }
// }

// impl Tensor {
//     fn new(array: Arc<Array>) -> Tensor {
//         Tensor { array, children: Vec::new(), consumer_count: 0, backward_op: None, tx: None }
//     }

//     fn propagate_consumers(&mut self) {
//         let guard = *array.lock().unwrap();
//         for child in &mut guard.children {
//             child.consumer_count += 1;
//             child.propagate_consumers();
//         }
//     }

//     fn backward(mut self, delta: Option<Tensor>) {
//         let delta = match delta {
//             Some(x) => x,
//             None => {
//                 self.propagate_consumers();
//                 Tensor::new(Arrays::new((self.dimensions().clone(), vec![1.0; self.values().len()]), None))
//             }
//         };

//         match &self.backward_op {
//             Some(x) => {
//                 self.consumer_count = 0;
//                 let delta = (*x)(&self.children(), delta);
//                 for mut child in self.children() {
//                     match child.tx {
//                         Some(x) => {
//                         },
//                         None => {
//                         },
//                     }
//                 }
//             },
//             None => panic!("error: operation is not differentiable"),
//         }
//     }
// }

// impl Clone for Tensor {
//     #[inline]
//     fn clone(&self) -> Tensor {
//         Tensor::new(Arc::clone(&self.array))
//     }
// }


// fn add_values(a: &Vec<Float>, b: &Vec<Float>) -> Vec<Float> {
//     a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<Float>>()
// }

// impl<'a, 'b> ops::Add<&'b Tensor> for &'a Tensor {
//     type Output = Tensor;

//     #[inline]
//     fn add(self, other: &'b Tensor) -> Tensor {
//         let backward_op = Box::new(|_: &Vec<Tensor>, x: Tensor| Tensor { array: array![Arrays::new(Arc::clone(&x.array).values(), None), Arrays::new(Arc::clone(&x.array).values(), None)] });
//         Arrays::new((self.dimensions.clone(), add_values(&self.values(), &other.values())), Some(backward_op)).with_children(vec![Arc::clone(&self.array), Arc::clone(&other.array)])
//     }
// }
