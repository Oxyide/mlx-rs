# Converting Python MLX to Rust MLX

This guide helps you translate code from the [Python MLX library](https://github.com/ml-explore/mlx) to [mlx-rs](https://github.com/oxideai/mlx-rs). It provides a direct mapping of Python MLX functions and classes to their Rust equivalents.

> **Note**: This document is inspired by the [Swift MLX Converting Python guide](https://swiftpackageindex.com/ml-explore/mlx-swift/documentation/mlx/converting-python).

## Table of Contents

- [Key Differences](#key-differences)
- [Array Creation](#array-creation)
- [Unary Operations](#unary-operations)
- [Binary Operations](#binary-operations)
- [Reductions](#reductions)
- [Indexing and Slicing](#indexing-and-slicing)
- [Linear Algebra](#linear-algebra)
- [Neural Network Layers](#neural-network-layers)
- [Optimizers](#optimizers)
- [Transforms](#transforms)
- [Utilities](#utilities)

## Key Differences

### Ownership and Borrowing
Rust's ownership model means you'll often use references (`&Array`) where Python uses direct values:

```python
# Python
result = mlx.add(a, b)
```

```rust
// Rust
let result = ops::add(&a, &b)?;
```

### Error Handling
Python uses exceptions, while Rust uses `Result<T, E>`:

```python
# Python - exceptions raised automatically
arr = mlx.zeros([2, 3])
```

```rust
// Rust - explicit error handling
let arr = Array::zeros::<f32>(&[2, 3])?;
// or with unwrap for quick prototyping
let arr = Array::zeros::<f32>(&[2, 3]).unwrap();
```

### Type System
Rust is statically typed and requires explicit type annotations:

```python
# Python - dynamic typing
arr = mlx.zeros([2, 3], dtype=mlx.float32)
```

```rust
// Rust - explicit types
let arr = Array::zeros::<f32>(&[2, 3])?;
// or
let arr = Array::zeros_dtype(&[2, 3], Dtype::Float32)?;
```

### Module Organization
Python uses flat namespace imports, while Rust uses hierarchical modules:

```python
# Python
import mlx.core as mx
import mlx.nn as nn
```

```rust
// Rust
use mlx_rs::{Array, ops};
use mlx_rs::nn;
```

## Array Creation

| Python (`mlx.core`) | Rust (`mlx_rs`) | Notes |
|---------------------|-----------------|-------|
| `zeros(shape, dtype=float32)` | `Array::zeros::<T>(&[...])` | Type parameter `T` determines dtype |
| `ones(shape, dtype=float32)` | `Array::ones::<T>(&[...])` | Similar to zeros |
| `full(shape, value, dtype=float32)` | `Array::full::<T>(&[...], value)` | Value must be an Array |
| `arange(start, stop, step, dtype)` | `ops::arange(start, stop, step)` | Returns Result |
| `linspace(start, stop, num, dtype)` | `ops::linspace(start, stop, num)` | Returns Result |
| `eye(n, m, k, dtype)` | `ops::eye(n, m, k)` | Identity matrix |
| `identity(n, dtype)` | `ops::identity(n)` | Square identity matrix |
| `array(data, dtype)` | `array!([...])` | Macro for convenient creation |
| `empty(shape, dtype)` | Not directly available | Use `zeros` or `full` instead |
| `random.uniform(low, high, shape)` | `random::uniform(low, high, &[...])` | Random arrays |
| `random.normal(shape, loc, scale)` | `random::normal(&[...], loc, scale)` | Normal distribution |

### Examples

**Python:**
```python
import mlx.core as mx

# Create arrays
a = mx.zeros([2, 3])
b = mx.ones([3, 4], dtype=mx.float32)
c = mx.arange(0, 10, 2)
d = mx.array([[1, 2], [3, 4]])
e = mx.random.uniform(0.0, 1.0, [2, 2])
```

**Rust:**
```rust
use mlx_rs::{Array, array, ops, random};

// Create arrays
let a = Array::zeros::<f32>(&[2, 3])?;
let b = Array::ones::<f32>(&[3, 4])?;
let c = ops::arange(0, Some(10), Some(2))?;
let d = array!([[1, 2], [3, 4]]);
let e = random::uniform(0.0, 1.0, &[2, 2])?;
```

## Unary Operations

| Python (`mlx.core`) | Rust (`mlx_rs::ops`) | Notes |
|---------------------|----------------------|-------|
| `abs(x)` | `abs(&x)` | Absolute value |
| `sin(x)` | `sin(&x)` | Sine |
| `cos(x)` | `cos(&x)` | Cosine |
| `tan(x)` | `tan(&x)` | Tangent |
| `exp(x)` | `exp(&x)` | Exponential |
| `log(x)` | `log(&x)` | Natural logarithm |
| `log2(x)` | `log2(&x)` | Base-2 logarithm |
| `log10(x)` | `log10(&x)` | Base-10 logarithm |
| `sqrt(x)` | `sqrt(&x)` | Square root |
| `square(x)` | `square(&x)` | Square |
| `sigmoid(x)` | `sigmoid(&x)` | Sigmoid function |
| `relu(x)` | `relu(&x)` | ReLU activation |
| `negative(x)` | `negative(&x)` | Negate values |
| `logical_not(x)` | `logical_not(&x)` | Logical NOT |

### Examples

**Python:**
```python
import mlx.core as mx

x = mx.array([1.0, 2.0, 3.0])
y = mx.sin(x)
z = mx.exp(x)
```

**Rust:**
```rust
use mlx_rs::{array, ops};

let x = array!([1.0, 2.0, 3.0]);
let y = ops::sin(&x)?;
let z = ops::exp(&x)?;
```

## Binary Operations

| Python (`mlx.core`) | Rust (`mlx_rs::ops` or operator) | Notes |
|---------------------|-----------------------------------|-------|
| `add(a, b)` or `a + b` | `ops::add(&a, &b)` or `&a + &b` | Addition |
| `subtract(a, b)` or `a - b` | `ops::subtract(&a, &b)` or `&a - &b` | Subtraction |
| `multiply(a, b)` or `a * b` | `ops::multiply(&a, &b)` or `&a * &b` | Element-wise multiplication |
| `divide(a, b)` or `a / b` | `ops::divide(&a, &b)` or `&a / &b` | Element-wise division |
| `maximum(a, b)` | `ops::maximum(&a, &b)` | Element-wise maximum |
| `minimum(a, b)` | `ops::minimum(&a, &b)` | Element-wise minimum |
| `power(a, b)` | `ops::power(&a, &b)` | Element-wise power |
| `equal(a, b)` or `a == b` | `ops::equal(&a, &b)` | Element-wise equality |
| `not_equal(a, b)` or `a != b` | `ops::not_equal(&a, &b)` | Element-wise inequality |
| `greater(a, b)` or `a > b` | `ops::greater(&a, &b)` | Element-wise greater than |
| `less(a, b)` or `a < b` | `ops::less(&a, &b)` | Element-wise less than |
| `logical_and(a, b)` | `ops::logical_and(&a, &b)` | Logical AND |
| `logical_or(a, b)` | `ops::logical_or(&a, &b)` | Logical OR |

### Examples

**Python:**
```python
import mlx.core as mx

a = mx.array([1, 2, 3])
b = mx.array([4, 5, 6])

c = a + b           # or mx.add(a, b)
d = a * b           # or mx.multiply(a, b)
e = mx.maximum(a, b)
```

**Rust:**
```rust
use mlx_rs::{array, ops};

let a = array!([1, 2, 3]);
let b = array!([4, 5, 6]);

let c = &a + &b;              // or ops::add(&a, &b)?
let d = &a * &b;              // or ops::multiply(&a, &b)?
let e = ops::maximum(&a, &b)?;
```

## Reductions

| Python (`mlx.core`) | Rust (`mlx_rs::ops`) | Notes |
|---------------------|----------------------|-------|
| `sum(x, axis, keepdims)` | `ops::sum(&x, axis, keepdims)` | Sum along axis |
| `mean(x, axis, keepdims)` | `ops::mean(&x, axis, keepdims)` | Mean along axis |
| `var(x, axis, keepdims)` | `ops::var(&x, axis, keepdims)` | Variance |
| `std(x, axis, keepdims)` | Not yet available | Standard deviation |
| `prod(x, axis, keepdims)` | `ops::prod(&x, axis, keepdims)` | Product along axis |
| `max(x, axis, keepdims)` | `ops::max(&x, axis, keepdims)` | Maximum value |
| `min(x, axis, keepdims)` | `ops::min(&x, axis, keepdims)` | Minimum value |
| `argmax(x, axis, keepdims)` | `ops::argmax(&x, axis, keepdims)` | Index of maximum |
| `argmin(x, axis, keepdims)` | `ops::argmin(&x, axis, keepdims)` | Index of minimum |
| `all(x, axis, keepdims)` | `ops::all(&x, axis, keepdims)` | Logical AND reduction |
| `any(x, axis, keepdims)` | `ops::any(&x, axis, keepdims)` | Logical OR reduction |

### Examples

**Python:**
```python
import mlx.core as mx

x = mx.array([[1, 2, 3], [4, 5, 6]])
total = mx.sum(x)                  # Sum all elements
row_sums = mx.sum(x, axis=1)       # Sum along axis 1
col_means = mx.mean(x, axis=0)     # Mean along axis 0
```

**Rust:**
```rust
use mlx_rs::{array, ops};

let x = array!([[1, 2, 3], [4, 5, 6]]);
let total = ops::sum_all(&x)?;           // Sum all elements
let row_sums = ops::sum(&x, &[1], false)?;  // Sum along axis 1
let col_means = ops::mean(&x, &[0], false)?; // Mean along axis 0
```

## Indexing and Slicing

| Python (`mlx.core`) | Rust (`mlx_rs`) | Notes |
|---------------------|-----------------|-------|
| `x[i]` | `x.index(i)` | Single index |
| `x[i, j]` | `x.index((i, j))` | Multi-dimensional index |
| `x[start:stop]` | `x.index(start..stop)` | Slice |
| `x[start:stop:step]` | Not directly available | Use explicit indexing |
| `x[:, i]` | `x.index((.., i))` | Select column |
| `take(x, indices, axis)` | `ops::take(&x, &indices, axis)` | Gather elements |

### Examples

**Python:**
```python
import mlx.core as mx

x = mx.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
row = x[0]          # First row
element = x[1, 2]   # Element at [1, 2]
slice = x[0:2]      # First two rows
col = x[:, 1]       # Second column
```

**Rust:**
```rust
use mlx_rs::{array, ops::indexing::IndexOp};

let x = array!([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
let row = x.index(0)?;          // First row
let element = x.index((1, 2))?; // Element at [1, 2]
let slice = x.index(0..2)?;     // First two rows
let col = x.index((.., 1))?;    // Second column
```

## Linear Algebra

| Python (`mlx.core`) | Rust (`mlx_rs`) | Notes |
|---------------------|-----------------|-------|
| `matmul(a, b)` or `a @ b` | `ops::matmul(&a, &b)` | Matrix multiplication |
| `tensordot(a, b, axes)` | Not yet available | Tensor contraction |
| `transpose(x, axes)` | `ops::transpose(&x, axes)` | Transpose |
| `reshape(x, shape)` | `ops::reshape(&x, &[...])` | Reshape array |
| `flatten(x)` | `ops::flatten(&x)` | Flatten to 1D |
| `linalg.inv(x)` | `linalg::inv(&x)` | Matrix inverse |
| `linalg.norm(x, ord, axis)` | `linalg::norm(&x, ord, axis)` | Vector/matrix norm |
| `linalg.qr(x)` | `linalg::qr(&x)` | QR decomposition |
| `linalg.svd(x)` | `linalg::svd(&x)` | SVD decomposition |
| `linalg.cholesky(x)` | `linalg::cholesky(&x)` | Cholesky decomposition |

### Examples

**Python:**
```python
import mlx.core as mx

a = mx.array([[1, 2], [3, 4]])
b = mx.array([[5, 6], [7, 8]])

c = a @ b                    # or mx.matmul(a, b)
d = mx.transpose(a)
e = mx.linalg.inv(a)
```

**Rust:**
```rust
use mlx_rs::{array, ops, linalg};

let a = array!([[1, 2], [3, 4]]);
let b = array!([[5, 6], [7, 8]]);

let c = ops::matmul(&a, &b)?;
let d = ops::transpose(&a, &[1, 0])?;
let e = linalg::inv(&a)?;
```

## Neural Network Layers

| Python (`mlx.nn`) | Rust (`mlx_rs::nn`) | Notes |
|-------------------|---------------------|-------|
| `Linear(in_dims, out_dims)` | `Linear::new(in_dims, out_dims)` | Fully connected layer |
| `Conv1d(in_channels, out_channels, kernel_size)` | `Conv1d::new(in_channels, out_channels, kernel_size)` | 1D convolution |
| `Conv2d(in_channels, out_channels, kernel_size)` | `Conv2d::new(in_channels, out_channels, kernel_size)` | 2D convolution |
| `LayerNorm(dims)` | `LayerNorm::new(dims)` | Layer normalization |
| `BatchNorm(num_features)` | `BatchNorm::new(num_features)` | Batch normalization |
| `RMSNorm(dims)` | `RMSNorm::new(dims)` | RMS normalization |
| `GroupNorm(num_groups, dims)` | `GroupNorm::new(num_groups, dims)` | Group normalization |
| `Embedding(num_embeddings, dims)` | `Embedding::new(num_embeddings, dims)` | Embedding layer |
| `MultiHeadAttention(dims, num_heads)` | `MultiHeadAttention::new(dims, num_heads)` | Multi-head attention |
| `Dropout(p)` | `Dropout::new(p)` | Dropout layer |
| `ReLU()` | `nn::relu(&x)` | ReLU activation (function) |
| `GELU()` | `nn::gelu(&x)` | GELU activation (function) |
| `SiLU()` | `nn::silu(&x)` | SiLU/Swish activation |
| `Softmax()` | `ops::softmax(&x, axis)` | Softmax activation |

### Examples

**Python:**
```python
import mlx.core as mx
import mlx.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        return self.fc2(x)
```

**Rust:**
```rust
use mlx_rs::{Array, module::Module, nn};

#[derive(Debug, Module)]
struct MLP {
    fc1: nn::Linear,
    fc2: nn::Linear,
}

impl MLP {
    fn new() -> Self {
        Self {
            fc1: nn::Linear::new(784, 128).unwrap(),
            fc2: nn::Linear::new(128, 10).unwrap(),
        }
    }

    fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let x = nn::relu(&self.fc1.forward(x)?)?;
        self.fc2.forward(&x)
    }
}
```

## Optimizers

| Python (`mlx.optimizers`) | Rust (`mlx_rs::optimizers`) | Notes |
|---------------------------|----------------------------|-------|
| `SGD(learning_rate)` | `Sgd::new(learning_rate)` | Stochastic gradient descent |
| `Adam(learning_rate)` | `Adam::new(learning_rate)` | Adam optimizer |
| `AdamW(learning_rate)` | `AdamW::new(learning_rate)` | Adam with weight decay |
| `AdaGrad(learning_rate)` | `Adagrad::new(learning_rate)` | AdaGrad optimizer |
| `RMSprop(learning_rate)` | `RmsProp::new(learning_rate)` | RMSProp optimizer |
| `Lion(learning_rate)` | `Lion::new(learning_rate)` | Lion optimizer |
| `optimizer.update(model, grads)` | `optimizer.update(&mut model, grads)` | Apply gradients |

### Examples

**Python:**
```python
import mlx.optimizers as optim

optimizer = optim.Adam(learning_rate=0.001)
optimizer.update(model, gradients)
```

**Rust:**
```rust
use mlx_rs::optimizers::{Adam, Optimizer};

let mut optimizer = Adam::new(0.001)?;
optimizer.update(&mut model, gradients)?;
```

## Transforms

| Python (`mlx.core`) | Rust (`mlx_rs::transforms`) | Notes |
|---------------------|----------------------------|-------|
| `grad(fn, argnums)` | `grad(fn)` | Compute gradient |
| `value_and_grad(fn, argnums)` | `value_and_grad(fn)` | Compute value and gradient |
| `eval(x)` | `x.eval()` | Force evaluation |
| `compile(fn)` | `compile(fn)` | Compile function |

### Examples

**Python:**
```python
import mlx.core as mx

def loss_fn(x):
    return mx.sum(x ** 2)

grad_fn = mx.grad(loss_fn)
x = mx.array([1.0, 2.0, 3.0])
gradient = grad_fn(x)
```

**Rust:**
```rust
use mlx_rs::{array, ops, transforms};

fn loss_fn(x: &Array) -> Result<Array, Exception> {
    Ok(ops::sum_all(&ops::square(x)?)?)
}

let mut grad_fn = transforms::grad(loss_fn);
let x = array!([1.0, 2.0, 3.0]);
let gradient = grad_fn(&x)?;
```

## Utilities

| Python (`mlx.core`) | Rust (`mlx_rs`) | Notes |
|---------------------|-----------------|-------|
| `load(path)` | `Array::load_numpy(path)` | Load from .npy file |
| `save(path, x)` | `x.save_numpy(path)` | Save to .npy file |
| `save_safetensors(path, arrays)` | `Array::save_safetensors(path, arrays)` | Save multiple arrays |
| `load_safetensors(path)` | `Array::load_safetensors(path)` | Load multiple arrays |
| `metal.is_available()` | `metal::is_available()` | Check Metal availability |
| `metal.get_active_memory()` | `memory::get_active_memory()` | Get active GPU memory |
| `metal.get_peak_memory()` | `memory::get_peak_memory()` | Get peak GPU memory |
| `metal.set_memory_limit(limit)` | `memory::set_memory_limit(limit)` | Set GPU memory limit |

### Examples

**Python:**
```python
import mlx.core as mx

# Save and load
x = mx.array([1, 2, 3])
mx.save("array.npy", x)
y = mx.load("array.npy")

# Memory monitoring
import mlx.core.metal as metal
active = metal.get_active_memory()
peak = metal.get_peak_memory()
```

**Rust:**
```rust
use mlx_rs::{array, Array, memory};

// Save and load
let x = array!([1, 2, 3]);
x.save_numpy("array.npy")?;
let y = Array::load_numpy("array.npy")?;

// Memory monitoring
let active = memory::get_active_memory();
let peak = memory::get_peak_memory();
```

## Common Patterns

### Training Loop

**Python:**
```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

model = MyModel()
optimizer = optim.Adam(learning_rate=0.001)
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

for batch in dataset:
    loss, grads = loss_and_grad_fn(model, batch)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), loss)
```

**Rust:**
```rust
use mlx_rs::{nn, optimizers::{Adam, Optimizer}, transforms};

let mut model = MyModel::new();
let mut optimizer = Adam::new(0.001)?;
let mut loss_and_grad_fn = transforms::value_and_grad(loss_fn);

for batch in dataset {
    let (loss, grads) = loss_and_grad_fn(&mut model, batch)?;
    optimizer.update(&mut model, grads)?;
    model.eval_params()?;
    loss.eval()?;
}
```

## Missing Features

Some Python MLX features are not yet available in mlx-rs:

- Standard deviation reduction (`std`)
- Tensor contraction (`tensordot`)
- Some advanced slicing patterns
- Learning rate schedulers
- Some specialized layers

Check the [mlx-rs issues](https://github.com/oxideai/mlx-rs/issues) for the current status and contribute if interested!

## Contributing

Found a missing or incorrect mapping? Please submit a pull request or open an issue at [github.com/oxideai/mlx-rs](https://github.com/oxideai/mlx-rs).

## Additional Resources

- [MLX Python Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [mlx-rs Documentation](https://docs.rs/mlx-rs)
- [MLX Swift Converting Python Guide](https://swiftpackageindex.com/ml-explore/mlx-swift/documentation/mlx/converting-python)
