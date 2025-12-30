//! Thread safety stress tests
//!
//! These tests validate whether mlx-rs is safe for concurrent operations.
//! They are designed to catch race conditions in:
//! - eval() operations (#259)
//! - Compiled functions (#258, #224)
//! - Mixed concurrent operations
//!
//! ## CRITICAL NOTE
//!
//! **Upstream mlx-rs does NOT implement `Sync` for `Array`!**
//!
//! These tests require Oxyide-MLX fork which has:
//! ```rust
//! unsafe impl Sync for Array {}
//! ```
//!
//! See PR #302: https://github.com/oxideai/mlx-rs/pull/302
//!
//! ## Running Tests
//!
//! ```bash
//! # Run sequentially (MLX has Metal GPU threading restrictions)
//! cargo test --test thread_safety -- --test-threads=1
//!
//! # With thread sanitizer (if available)
//! RUSTFLAGS="-Z sanitizer=thread" cargo +nightly test --test thread_safety -- --test-threads=1
//! ```
//!
//! ## Expected Behavior
//!
//! If these tests:
//! - **PASS**: mlx-rs may be thread-safe (but more testing recommended)
//! - **FAIL/CRASH**: mlx-rs needs locking mechanisms like mlx-swift
//! - **HANG**: Likely deadlock or race condition
//!
//! See docs/THREAD_SAFETY_INVESTIGATION.md for full analysis.

use mlx_rs::{
    array,
    ops::ones,
    transforms::{compile::compile, eval},
    Array,
};
use std::sync::{Arc, Mutex};
use std::thread;

/// Test 1: Concurrent eval() calls on different arrays
///
/// This tests whether MLX's lazy evaluation graph construction is thread-safe.
/// Multiple threads calling eval() on different arrays simultaneously.
///
/// **mlx-swift uses NSRecursiveLock for this** - does mlx-rs need it?
#[test]
fn test_concurrent_eval_different_arrays() {
    let thread_count = 10;
    let iterations = 100;

    let handles: Vec<_> = (0..thread_count)
        .map(|thread_id| {
            thread::spawn(move || {
                for iter in 0..iterations {
                    // Create unique arrays for this thread
                    let value = (thread_id * 1000 + iter) as f32;
                    let arr = array!([value, value + 1.0, value + 2.0]);

                    // Evaluate
                    arr.eval().unwrap();

                    // Verify correctness
                    let result: Vec<f32> = arr.as_slice().to_vec();
                    assert_eq!(result, vec![value, value + 1.0, value + 2.0]);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test 2: Concurrent eval() on shared arrays (via Arc)
///
/// Tests whether eval() is safe when called on the same array from multiple threads.
/// Array is Send+Sync, so this should be allowed by the type system.
#[test]
fn test_concurrent_eval_shared_arrays() {
    let arrays: Vec<Arc<Array>> = (0..20)
        .map(|i| Arc::new(array!([i as f32, (i + 1) as f32, (i + 2) as f32])))
        .collect();

    let thread_count = 10;
    let iterations = 50;

    let handles: Vec<_> = (0..thread_count)
        .map(|_| {
            let arrays = arrays.clone();
            thread::spawn(move || {
                for _ in 0..iterations {
                    for arr in &arrays {
                        // Multiple threads evaluating the same array
                        arr.eval().unwrap();
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test 3: Concurrent eval() with operations (lazy graph construction)
///
/// Tests whether building and evaluating computation graphs is thread-safe.
/// This is the most likely to trigger race conditions.
#[test]
fn test_concurrent_eval_with_operations() {
    let base_arrays: Vec<Arc<Array>> = (0..10)
        .map(|i| {
            Arc::new(
                ones::<f32>(&[100])
                    .unwrap()
                    .multiply(&array!(i as f32))
                    .unwrap(),
            )
        })
        .collect();

    let thread_count = 10;
    let iterations = 50;

    let handles: Vec<_> = (0..thread_count)
        .map(|_| {
            let arrays = base_arrays.clone();
            thread::spawn(move || {
                for _ in 0..iterations {
                    for arr in &arrays {
                        // Create computation graph
                        let result = arr.add(&ones::<f32>(&[100]).unwrap()).unwrap();

                        // Evaluate lazily constructed graph
                        result.eval().unwrap();
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test 4: Concurrent transforms::eval() on array slices
///
/// Tests the batch eval() function that takes multiple arrays.
#[test]
fn test_concurrent_batch_eval() {
    let arrays: Vec<Arc<Array>> = (0..50)
        .map(|i| Arc::new(ones::<f32>(&[10]).unwrap().multiply(&array!(i as f32)).unwrap()))
        .collect();

    let thread_count = 10;
    let iterations = 20;

    let handles: Vec<_> = (0..thread_count)
        .map(|_| {
            let arrays = arrays.clone();
            thread::spawn(move || {
                for _ in 0..iterations {
                    // Batch evaluate multiple arrays
                    let arr_refs: Vec<&Array> = arrays.iter().map(|a| a.as_ref()).collect();
                    eval(&arr_refs).unwrap();
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test 5: Concurrent compiled function calls (SAME function)
///
/// **This is the critical test for #258/#224.**
///
/// mlx-swift requires per-function locking because:
/// - mlx::core::compile maintains a cache that isn't thread-safe
/// - Tracer arguments have shared mutable state
///
/// If this test crashes or produces wrong results, we need locks.
#[test]
fn test_concurrent_same_compiled_function() {
    // Create a simple function
    let f = |x: &Array| x.square().unwrap();

    // Compile it ONCE
    let compiled = compile(f, None);

    // Share it between threads (need Mutex since compiled function takes &mut self)
    let compiled = Arc::new(Mutex::new(compiled));

    let thread_count = 10;
    let iterations = 100;

    let handles: Vec<_> = (0..thread_count)
        .map(|thread_id| {
            let compiled = Arc::clone(&compiled);
            thread::spawn(move || {
                for iter in 0..iterations {
                    // Each thread uses different input
                    let value = (thread_id * 1000 + iter) as f32;
                    let input = array!([value, value + 1.0, value + 2.0]);

                    // Call the SAME compiled function from multiple threads
                    let result = {
                        let mut compiled = compiled.lock().unwrap();
                        compiled(&input).unwrap()
                    };

                    result.eval().unwrap();

                    // Verify correctness
                    let expected = vec![value * value, (value + 1.0) * (value + 1.0), (value + 2.0) * (value + 2.0)];
                    let actual: Vec<f32> = result.as_slice().to_vec();

                    for (a, e) in actual.iter().zip(expected.iter()) {
                        assert!((a - e).abs() < 1e-5, "thread {} iter {}: expected {}, got {}", thread_id, iter, e, a);
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test 6: Concurrent DIFFERENT compiled functions
///
/// Tests whether different compiled functions can run in parallel.
/// This should be safe even if same-function calls aren't.
#[test]
fn test_concurrent_different_compiled_functions() {
    let thread_count = 10;
    let iterations = 50;

    let handles: Vec<_> = (0..thread_count)
        .map(|thread_id| {
            thread::spawn(move || {
                // Each thread gets its OWN compiled function
                let f = |x: &Array| {
                    x.multiply(&array!(thread_id as f32 + 1.0)).unwrap()
                };
                let mut compiled = compile(f, None);

                for iter in 0..iterations {
                    let value = iter as f32;
                    let input = array!([value, value + 1.0, value + 2.0]);

                    // Call this thread's compiled function
                    let result = compiled(&input).unwrap();
                    result.eval().unwrap();

                    // Verify
                    let multiplier = thread_id as f32 + 1.0;
                    let expected = vec![value * multiplier, (value + 1.0) * multiplier, (value + 2.0) * multiplier];
                    let actual: Vec<f32> = result.as_slice().to_vec();

                    for (a, e) in actual.iter().zip(expected.iter()) {
                        assert!((a - e).abs() < 1e-5);
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test 7: Mixed concurrent operations (eval + compiled functions)
///
/// Tests realistic scenario: some threads doing eval(), others using compiled functions.
#[test]
fn test_mixed_concurrent_operations() {
    let shared_arrays: Vec<Arc<Array>> = (0..20)
        .map(|i| Arc::new(ones::<f32>(&[50]).unwrap().multiply(&array!(i as f32)).unwrap()))
        .collect();

    let compiled_square = Arc::new(Mutex::new(compile(
        |x: &Array| x.square().unwrap(),
        None,
    )));

    let thread_count = 10;
    let iterations = 50;

    let handles: Vec<_> = (0..thread_count)
        .map(|thread_id| {
            let arrays = shared_arrays.clone();
            let compiled = Arc::clone(&compiled_square);

            thread::spawn(move || {
                for iter in 0..iterations {
                    if thread_id % 2 == 0 {
                        // Even threads: eval operations
                        for arr in &arrays {
                            arr.eval().unwrap();
                        }
                    } else {
                        // Odd threads: compiled function calls
                        let value = iter as f32;
                        let input = array!([value; 50]);
                        let result = {
                            let mut c = compiled.lock().unwrap();
                            c(&input).unwrap()
                        };
                        result.eval().unwrap();
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test 8: Stress test with larger arrays and more threads
///
/// Push the limits to try to trigger race conditions.
#[test]
#[ignore] // Run explicitly with: cargo test test_stress -- --ignored --test-threads=1
fn test_stress_concurrent_operations() {
    let thread_count = 20;
    let iterations = 500;
    let array_size = 1000;

    let base_arrays: Vec<Arc<Array>> = (0..50)
        .map(|i| Arc::new(ones::<f32>(&[array_size]).unwrap().multiply(&array!(i as f32)).unwrap()))
        .collect();

    let handles: Vec<_> = (0..thread_count)
        .map(|_| {
            let arrays = base_arrays.clone();
            thread::spawn(move || {
                for _ in 0..iterations {
                    // Randomly pick arrays and eval them
                    for arr in arrays.iter().take(10) {
                        let result = arr.add(&ones::<f32>(&[array_size]).unwrap()).unwrap();
                        result.eval().unwrap();
                    }
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test 9: Concurrent compiled functions with state
///
/// Tests compile_with_state which adds another layer of complexity.
#[test]
fn test_concurrent_compiled_with_state() {
    use mlx_rs::transforms::compile::compile_with_state;

    // Simple stateful operation
    let f = |state: &mut i32, x: &Array| {
        *state += 1;
        x.add(&array!(*state as f32)).unwrap()
    };

    let compiled = Arc::new(Mutex::new(compile_with_state(f, None)));

    let thread_count = 10;
    let iterations = 50;

    let handles: Vec<_> = (0..thread_count)
        .map(|thread_id| {
            let compiled = Arc::clone(&compiled);
            thread::spawn(move || {
                let mut state = 0i32;

                for iter in 0..iterations {
                    let value = (thread_id * 100 + iter) as f32;
                    let input = array!([value; 10]);

                    let result = {
                        let mut c = compiled.lock().unwrap();
                        c(&mut state, &input).unwrap()
                    };

                    result.eval().unwrap();
                }

                // Each thread should have incremented its own state
                assert_eq!(state, iterations);
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }
}

/// Test 10: Re-entrant eval (if possible)
///
/// Tests whether eval() can be called from within a closure that's being evaluated.
/// This is why mlx-swift uses a *recursive* lock.
#[test]
#[ignore] // May not be directly testable without custom ops
fn test_reentrant_eval() {
    // This would require a way to trigger eval() from within a closure
    // that's being traced/compiled. Not directly possible with current API.
    //
    // If we find this is needed, we'll need to implement a recursive lock
    // like mlx-swift does.
}

// Helper function for debugging - not a test
#[allow(dead_code)]
fn print_test_info(test_name: &str) {
    println!("\n=== {} ===", test_name);
    println!("Thread count: {:?}", thread::current().id());
    println!("Starting test...\n");
}
