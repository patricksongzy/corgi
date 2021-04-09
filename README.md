# Corgi
* An automatic differentiation implementation in Rust.
* https://crates.io/crates/corgi

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
* Fully-connected neural network ([full version](https://github.com/patricksongzy/corgi/blob/main/src/model.rs#L65))
```rust
use rand::Rng;
let mut rng = rand::thread_rng();

let learning_rate = 0.01;
let input_size = 1;
let hidden_size = 16;
let output_size = 1;
let initializer = Arc::new(|x: Float| {
    let range = 1.0 / x.sqrt();
    rand::thread_rng().gen_range(-range..=range)

});
let sigmoid = Arc::new(|x: Array| x.sigmoid());
let gd = GradientDescent::new(learning_rate);
let l1 = Dense::new(input_size, hidden_size, initializer.clone(), Some(sigmoid));
let l2 = Dense::new(hidden_size, output_size, initializer.clone(), None);
let mut model = Model::new(vec![Box::new(l1), Box::new(l2)], Box::new(gd));

for _ in 0..8 {
    let x = rng.gen_range(-1.0..1.0);
    let input = arr![arr![x]];
    let target = x.exp();

    let result = model.forward(input);
    let loss = model.backward(arr![target]);

    println!(
	"in: {}, out: {}, target: {}, loss: {}",
	x, result[0], target, loss
    );
}
```
* Custom operation (still needs some work)
```rust
let op: array::ForwardOp = Arc::new(|x: &[&Array]| {
    Arrays::new((x[0].dimensions(), x[0].values().iter().zip(x[1].values()).map(|(x, y)| x * y).collect::<Vec<Float>>()))
});

let op_clone = Arc::clone(&op);
let backward_op: array::BackwardOp = Arc::new(move |c: &mut Vec<Array>, x: &mut Array| {
    vec![
	Some(Array::op(&vec![c[1].untracked(), x.untracked()], &Vec::new(), Arc::clone(&op_clone), None)),
	Some(Array::op(&vec![c[0].untracked(), x.untracked()], &Vec::new(), Arc::clone(&op_clone), None)),
    ]
});

let a = arr![1.0, 2.0, 3.0];
let b = arr![3.0, 2.0, 1.0];

let mut product = Array::op(&vec![&a, &b], op, Some(backward_op));
assert_eq!(product, arr![3.0, 4.0, 3.0]);

product.backward(None);
assert_eq!(product.gradient(), arr![1.0, 1.0, 1.0]);
assert_eq!(b.gradient(), arr![1.0, 2.0, 3.0]);
assert_eq!(a.gradient(), arr![3.0, 2.0, 1.0]);
```

# Backward Pass
* An informal UML sequence diagram (it's not entirely up to specs, but should give an overview of the process)
![Informal UML sequence diagram](https://raw.githubusercontent.com/patricksongzy/corgi/main/images/sequence.svg?sanitize=true)

# Name
* Original name was going to be 'cog-(something)', since Rust's logo is a cog, and since cognition (get it?).
But as it turns out, many AI libraries are named 'cog-(something)'. Attempts at permutations of 'cog' sounded awkward, such as 'cogi', for 'cog-intelligence',
so the name Corgi was chosen.

# Licence
* MIT
