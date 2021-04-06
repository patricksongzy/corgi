# Corgi
* An automatic differentiation implementation in Rust.

# Design
* Originally worked around the ergonomics of the `arr!` macro (which however, currently still needs more work).
* Dynamic-as-possible computational graph.
* Did not want to have to manage any 'graph' structures when using Corgi (the Arrays should represent the graph alone).
* Graph became more, and more dependent on threading for the backward pass, and the use of `Arc`, and `Mutex`.
* Graphs do note store consumers at the moment.

# Examples
* Dynamic computational graph:
```rust
let a = arr![5.0];
let b = arr![2.0];
let mut c = arr![0.0];

for _ in 0..10 {
    c = &c + &(&a * &b);
    if c[0] > 50.0 {
        c = &c * &a;
    }
}
 
c.backward(None);
assert_eq!(c, arr![195300.0]);
assert_eq!(c.gradient(), arr![1.0]);
assert_eq!(b.gradient(), arr![97650.0]);
assert_eq!(a.gradient(), arr![232420.0]);
```
* Fully-connected neural network ([full version](https://github.com/patricksongzy/corgi/blob/main/src/dense.rs))
```rust
struct Dense {
    weights: Array,
    biases: Array,
}

impl Dense {
    forward(&self, x: Array) -> Array {
        (&Array::matmul(&self.weights, &x, false, false) + &self.biases).sigmoid()
    }
}

y = layer.forward(x);
let mut error = (&target - &y).powf(2.0);
let loss = error.sum();

error.backward(None);

let mut grad_weights = layer.weights.gradient();
let mut grad_biases = layer.biases.gradient();

layer.weights = layer.weights.untracked() + (grad_weights.untracked() * -lr).untracked();
layer.biases = layer.biases.untracked() + (grad_biases.untracked() * -lr).untracked();

*layer.weights.gradient_mut() = None;
*layer.biases.gradient_mut() = None;
```

# Backward Pass
* An informal UML sequence diagram (it's not entirely up to specs, but should give an overview of the process)
[Informal UML sequence diagram](https://raw.github.com/patricksongzy/corgi/main/images/sequence.svg?sanitize=true)

# Name
* Original name was going to be 'cog-(something)', since Rust's logo is a cog, and since cognition (get it?).
But as it turns out, many AI libraries are named 'cog-(something)'. Attempts at permutations of 'cog' sounded awkward, such as 'cogi', for 'cog-intelligence',
so the name Corgi was chosen.

# Licence
* MIT
