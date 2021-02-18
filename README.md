# rust-nd
An n-dimensional array implementation for Rust.

# Example
```[Rust]
let a = Tensor::new(
    array![array![
        array![0.0, 1.0], array![2.0, 3.0]
    ],
    array![
        array![4.0, 5.0], array![6.0, 7.0]
    ]]
);

let b = Tensor::new(
    array![array![
        array![2.0, 4.0], array![6.0, 8.0]
    ],
    array![
        array![10.0, 12.0], array![14.0, 16.0]
    ]]
);

let sum = &a + &b;
let product = &a * &b;
```
