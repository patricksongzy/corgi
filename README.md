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

## Examples
* For fully-connected examples, remember to call `model.update()`.
* Fully-connected [MNIST](https://github.com/patricksongzy/corgi-sample/blob/main/src/main.rs) (convolutional neural networks are in-progress).
* Fully-connected neural network ([full version](https://github.com/patricksongzy/corgi/blob/main/src/model.rs#L65)):
```rust
let initializer = initializer::make_he();
let relu = activation::make_relu();
let softmax = activation::make_softmax();
let ce = cost::make_cross_entropy();
let gd = GradientDescent::new(learning_rate);
let l1 = Dense::new(input_size, hidden_size, initializer.clone(), Some(relu));
let l2 = Dense::new(hidden_size, output_size, initializer.clone(), Some(softmax));
let mut model = Model::new(vec![Box::new(l1), Box::new(l2)], Box::new(gd), ce);

for _ in 0..iterations {
    let mut input = vec![0.0; input_size * batch_size];
    let mut target = vec![0.0; output_size * batch_size];

    // set inputs, and targets

    // arrays in corgi should not be mutated after creation, so we initialise the values first
    let input = Array::from((vec![batch_size, input_size], input));
    let target = Array::from((vec![batch_size, output_size], target));

    let _result = model.forward(input.clone());
    let loss = model.backward(target.clone());
    // update the parameters, and clear gradients (backward pass only sets gradients)
    model.update();

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
* [Custom operation](https://github.com/patricksongzy/corgi/blob/main/src/lib.rs#L34) (still needs some work).

## Important Design Notes
* Array values should never be modified from operations; instead, new arrays should be created.
* Arrays are untracked by default, so if gradients are required, `tracked()`, or `start_tracking()` must be used (see the documentation for details).
* Versions prior to 0.9.7 of Corgi did not prioritise optimisation, and will be slow.


## Design
* Eager execution.
* Dynamic-as-possible computational graph.
* Originally worked around the ergonomics of the `arr!` macro (which however, currently still needs more work).
* Did not want to have to manage any 'graph' structures when using Corgi (the Arrays should represent the graph alone).
* Graphs do note store consumers (at the moment). They store consumer counts instead.

## BLAS
* The `opeblas`, or `netlib` features can be enabled, and requires CBLAS if used.

### Tracked Arrays
* Tracked arrays are arrays which require gradients to be computed, and stored.
* For more information, see the documentation for `tracked()`, and `untracked()` in `array.rs`.

## Name
* Original name was going to be 'cog-(something)', since Rust's logo is a cog, and since cognition (get it?).
But, many AI libraries are named 'cog-(something)'. Attempts at permutations of 'cog' with other words sounded awkward, such as 'cogi', for 'cog-intelligence', so the name Corgi was chosen.

## Resources
* Shields are from [shields.io](https://shields.io).
* MIT 6.034 on OpenCourseWare for a primer on Backward Propagation.
* CS231n YouTube recordings for a primer on Convolutional Neural Networks.

A lot of the library was built around being as dynamic as possible, meaning if chosen well, some design choices might be similar to other dynamic computational graph libraries.

## Licence
* MIT
