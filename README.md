<h1 align="center">Corgi</h1>
<p align="center">A neural network, and tensor dynamic automatic differentiation implementation for Rust.</p>
<p align="center">
    <a href="https://github.com/patricksongzy/corgi/">
        <img alt="Build: Github Workflow" src="https://img.shields.io/github/workflow/status/patricksongzy/corgi/Rust"></img>
    </a>
    <a href="https://crates.io/crates/corgi">
        <img alt="Download: crates.io" src="https://img.shields.io/crates/v/corgi"></img>
    </a>
    <a href="https://docs.rs/corgi">
        <img alt="Documentation: docs.rs" src="https://docs.rs/corgi/badge.svg"></img>
    </a>
    <a href="https://github.com/patricksongzy/corgi/blob/main/LICENSE">
        <img alt="Licence: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></img>
    </a>
</p>
<hr>

## BLAS
* The BLAS feature can be enabled, and requires CBLAS if used.

## Examples
* Fully-connected neural network ([full version](https://github.com/patricksongzy/corgi/blob/main/src/model.rs#L65)):
```rust
let initializer = Arc::new(|x: Float| {
    let range = 1.0 / x.sqrt();
    rand::thread_rng().gen_range(-range..=range)
});
let sigmoid = Arc::new(|x: Array| x.sigmoid());
let gd = GradientDescent::new(learning_rate);
let l1 = Dense::new(input_size, hidden_size, initializer.clone(), Some(sigmoid));
let l2 = Dense::new(hidden_size, output_size, initializer.clone(), None);
let mut model = Model::new(vec![Box::new(l1), Box::new(l2)], Box::new(gd));

for _ in 0..iterations {
    let mut input = vec![0.0; input_size * batch_size];
    let mut target = vec![0.0; output_size * batch_size];

    // initialize inputs, and targets

    let input = Arrays::new((vec![batch_size, input_size], input));
    let target = Arrays::new((vec![batch_size, output_size], target));

    let result = model.forward(input.clone());
    let loss = model.backward(target.clone());

    println!("loss: {}", loss);
}
```
* Dynamic computational graph:
```rust
let a = arr![5.0].tracked();
let b = arr![2.0].tracked();
let mut c = arr![0.0].tracked();

for _ in 0..10 {
    c = &c + &(&a * &b);
    if c[0] > 50.0 {
	c = &c * &a;
    }
}

assert_eq!(c, arr![195300.0]);

c.backward(None);
assert_eq!(c.gradient(), arr![1.0]);
assert_eq!(b.gradient(), arr![97650.0]);
assert_eq!(a.gradient(), arr![232420.0]);
```
* Custom operation (still needs some work):
```rust
// note proper implementations should handle tracked, and untracked cases
let op: array::ForwardOp = Arc::new(|x: &[&Array]| {
    Arrays::new((x[0].dimensions(), x[0].values().iter().zip(x[1].values()).map(|(x, y)| x * y).collect::<Vec<Float>>()))
});

let op_clone = Arc::clone(&op);
let backward_op: array::BackwardOp = Arc::new(move |c: &mut Vec<Array>, x: &Array| {
    vec![Some(Array::op(&vec![&c[1], x], Arc::clone(&op_clone), None)),
         Some(Array::op(&vec![&c[0], x], Arc::clone(&op_clone), None))]
});

let a = arr![1.0, 2.0, 3.0];
let b = arr![3.0, 2.0, 1.0];
let mut product = Array::op(&vec![&a, &b], op, Some(backward_op));
assert_eq!(product, arr![3.0, 4.0, 3.0]);
product.backward(None);
assert_eq!(b.gradient(), arr![1.0, 2.0, 3.0]);
assert_eq!(a.gradient(), arr![3.0, 2.0, 1.0]);
```

## Design
* Originally worked around the ergonomics of the `arr!` macro (which however, currently still needs more work).
* Dynamic-as-possible computational graph.
* Did not want to have to manage any 'graph' structures when using Corgi (the Arrays should represent the graph alone).
* Graph became more, and more dependent on threading for the backward pass, and the use of `Arc`, and `Mutex`.
* Graphs do note store consumers (at the moment). They store consumer counts instead.

### Tracked Arrays
* Tracked arrays are arrays which require gradients to be computed, and stored.
* For more information, see the documentation for `tracked()`, and `untracked()` in `array.rs`.

## Backward Pass
* An informal UML sequence diagram (it's not entirely up to specs, but should give an overview of the process):

![Informal UML sequence diagram](https://raw.githubusercontent.com/patricksongzy/corgi/main/images/sequence.svg?sanitize=true)

## Name
* Original name was going to be 'cog-(something)', since Rust's logo is a cog, and since cognition (get it?).
But as it turns out, many AI libraries are named 'cog-(something)'. Attempts at permutations of 'cog' sounded awkward, such as 'cogi', for 'cog-intelligence',
so the name Corgi was chosen.

## Acknowledgements
* Shields are from [shields.io](https://shields.io)

## Licence
* MIT
