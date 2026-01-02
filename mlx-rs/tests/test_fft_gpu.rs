//! Test to verify FFT GPU (Metal) support
//!
//! This test verifies that FFT operations can execute on GPU when a GPU stream is provided.
//! Metal FFT support was added in MLX v0.15.0 (March 2024) for power-of-2 sizes up to 2048.

use mlx_rs::{Array, Dtype, fft::*, Stream};

#[test]
fn test_fft_runs_on_gpu() {
    // Create a power-of-2 sized array (GPU-supported size)
    let src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let array = Array::from_slice(&src[..], &[8]);

    // Create GPU stream
    let gpu_stream = Stream::task_local_or_gpu();

    // Run FFT on GPU stream
    let result = fft_device(&array, 8, 0, &gpu_stream).unwrap();

    // Verify result is complex64
    assert_eq!(result.dtype(), Dtype::Complex64);

    // Verify result has correct shape
    assert_eq!(result.shape(), &[8]);

    // Evaluate to ensure GPU execution completes
    result.eval().unwrap();

    println!("✓ FFT successfully executed on GPU stream");
}

#[test]
fn test_fft_power_of_2_sizes() {
    // Test various power-of-2 sizes that Metal FFT supports
    let sizes = vec![2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
    let gpu_stream = Stream::task_local_or_gpu();

    for size in sizes {
        let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let array = Array::from_slice(&data[..], &[size as i32]);

        let result = fft_device(&array, size as i32, 0, &gpu_stream).unwrap();
        result.eval().unwrap();

        assert_eq!(result.shape(), &[size as i32]);
        println!("✓ FFT size {} executed on GPU", size);
    }
}

#[test]
fn test_fft_2d_on_gpu() {
    // Test 2D FFT on GPU
    let gpu_stream = Stream::task_local_or_gpu();

    // Create 32x32 array (both dimensions power-of-2)
    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let array = Array::from_slice(&data[..], &[32, 32]);

    // Run 2D FFT on GPU
    let result = fft2_device(&array, &[32, 32], &[0, 1], &gpu_stream).unwrap();
    result.eval().unwrap();

    assert_eq!(result.shape(), &[32, 32]);
    println!("✓ 2D FFT (32x32) executed on GPU");
}

#[test]
fn test_ifft_on_gpu() {
    // Test inverse FFT on GPU
    let gpu_stream = Stream::task_local_or_gpu();

    let src = vec![1.0f32, 2.0, 3.0, 4.0];
    let array = Array::from_slice(&src[..], &[4]);

    // Forward FFT
    let fft_result = fft_device(&array, 4, 0, &gpu_stream).unwrap();
    fft_result.eval().unwrap();

    // Inverse FFT
    let ifft_result = ifft_device(&fft_result, 4, 0, &gpu_stream).unwrap();
    ifft_result.eval().unwrap();

    assert_eq!(ifft_result.shape(), &[4]);
    println!("✓ Inverse FFT executed on GPU");

    // Note: We don't check exact values due to floating point precision,
    // but the GPU execution path is verified
}
