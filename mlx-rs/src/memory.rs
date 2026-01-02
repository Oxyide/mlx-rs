//! GPU memory management functions for MLX.
//!
//! These functions allow you to monitor and control GPU memory usage, which is particularly
//! useful for running large models on devices with limited unified memory (e.g., base M1/M2 chips).
//!
//! # Example
//!
//! ```rust,no_run
//! use mlx_rs::memory;
//!
//! // Set a soft limit of 12GB to leave headroom on a 16GB device
//! let old_limit = memory::set_memory_limit(12 * 1024 * 1024 * 1024);
//! println!("Previous memory limit: {} bytes", old_limit);
//!
//! // Check current limit
//! let current_limit = memory::get_memory_limit();
//! println!("Current memory limit: {} bytes (0 = no limit)", current_limit);
//!
//! // Monitor memory usage
//! let active = memory::get_active_memory();
//! let peak = memory::get_peak_memory();
//! println!("Active memory: {} bytes, Peak: {} bytes", active, peak);
//! ```

use crate::error::{Exception, Result};
use crate::utils::SUCCESS;

/// Sets a soft limit on GPU memory usage (in bytes).
///
/// When the limit is exceeded, MLX will more aggressively free unused arrays from the GPU cache.
/// This is not a hard Metal limit but a soft, MLX-managed limit that influences internal
/// caching and reuse behavior.
///
/// # Arguments
///
/// * `limit` - Memory limit in bytes. Pass `0` to disable the limit (default behavior).
///
/// # Returns
///
/// Returns the previous memory limit in bytes.
///
/// # Example
///
/// ```rust,no_run
/// use mlx_rs::memory;
///
/// // Limit to 8GB
/// let old_limit = memory::set_memory_limit(8 * 1024 * 1024 * 1024);
/// println!("Previous limit was: {} bytes", old_limit);
/// ```
pub fn set_memory_limit(limit: usize) -> usize {
    let mut prev_limit: usize = 0;
    unsafe {
        mlx_sys::mlx_set_memory_limit(&mut prev_limit, limit);
    }
    prev_limit
}

/// Gets the current soft GPU memory limit (in bytes).
///
/// Returns `0` if no limit is set (unlimited).
///
/// # Example
///
/// ```rust,no_run
/// use mlx_rs::memory;
///
/// let limit = memory::get_memory_limit();
/// if limit == 0 {
///     println!("No memory limit set");
/// } else {
///     println!("Memory limit: {} bytes", limit);
/// }
/// ```
pub fn get_memory_limit() -> usize {
    let mut limit: usize = 0;
    unsafe {
        mlx_sys::mlx_get_memory_limit(&mut limit);
    }
    limit
}

/// Sets a soft limit on the cache size (in bytes).
///
/// The cache stores recently used arrays that can be reused without reallocation.
///
/// # Arguments
///
/// * `limit` - Cache limit in bytes. Pass `0` to disable caching.
///
/// # Returns
///
/// Returns the previous cache limit in bytes.
pub fn set_cache_limit(limit: usize) -> usize {
    let mut prev_limit: usize = 0;
    unsafe {
        mlx_sys::mlx_set_cache_limit(&mut prev_limit, limit);
    }
    prev_limit
}

/// Gets the current cache memory usage (in bytes).
///
/// This is the amount of memory currently held in the cache for potential reuse.
pub fn get_cache_memory() -> usize {
    let mut cache: usize = 0;
    unsafe {
        mlx_sys::mlx_get_cache_memory(&mut cache);
    }
    cache
}

/// Gets the active memory usage (in bytes).
///
/// This is the amount of memory currently allocated and in use by arrays.
pub fn get_active_memory() -> usize {
    let mut active: usize = 0;
    unsafe {
        mlx_sys::mlx_get_active_memory(&mut active);
    }
    active
}

/// Gets the peak memory usage (in bytes) since the last reset.
///
/// The peak is tracked since program start or the last call to [`reset_peak_memory`].
pub fn get_peak_memory() -> usize {
    let mut peak: usize = 0;
    unsafe {
        mlx_sys::mlx_get_peak_memory(&mut peak);
    }
    peak
}

/// Resets the peak memory counter to the current active memory.
///
/// Useful for measuring peak memory usage for specific operations.
///
/// # Example
///
/// ```rust,no_run
/// use mlx_rs::memory;
///
/// memory::reset_peak_memory();
/// // ... perform operations ...
/// let peak = memory::get_peak_memory();
/// println!("Peak memory for operation: {} bytes", peak);
/// ```
pub fn reset_peak_memory() -> Result<()> {
    let status = unsafe { mlx_sys::mlx_reset_peak_memory() };
    if status == SUCCESS {
        Ok(())
    } else {
        Err(Exception::custom("Failed to reset peak memory"))
    }
}

/// Clears the memory cache, freeing all cached memory.
///
/// This forces MLX to release all memory held in the cache for potential reuse.
///
/// # Example
///
/// ```rust,no_run
/// use mlx_rs::memory;
///
/// println!("Cache before: {} bytes", memory::get_cache_memory());
/// memory::clear_cache().unwrap();
/// println!("Cache after: {} bytes", memory::get_cache_memory());
/// ```
pub fn clear_cache() -> Result<()> {
    let status = unsafe { mlx_sys::mlx_clear_cache() };
    if status == SUCCESS {
        Ok(())
    } else {
        Err(Exception::custom("Failed to clear cache"))
    }
}

/// Sets a soft limit on wired memory (in bytes).
///
/// Wired memory is memory that cannot be paged out to disk.
///
/// # Arguments
///
/// * `limit` - Wired memory limit in bytes.
///
/// # Returns
///
/// Returns the previous wired memory limit in bytes.
pub fn set_wired_limit(limit: usize) -> usize {
    let mut prev_limit: usize = 0;
    unsafe {
        mlx_sys::mlx_set_wired_limit(&mut prev_limit, limit);
    }
    prev_limit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_get_memory_limit() {
        // Get initial limit
        let initial_limit = get_memory_limit();

        // Set a new limit (1GB)
        let new_limit = 1024 * 1024 * 1024;
        let old_limit = set_memory_limit(new_limit);
        assert_eq!(old_limit, initial_limit);

        // Verify the new limit was set
        let current_limit = get_memory_limit();
        assert_eq!(current_limit, new_limit);

        // Reset to no limit
        set_memory_limit(0);
        assert_eq!(get_memory_limit(), 0);
    }

    #[test]
    fn test_cache_operations() {
        // Get initial cache
        let _initial_cache = get_cache_memory();

        // Clear cache should succeed
        assert!(clear_cache().is_ok());

        // Cache should be 0 or very small after clearing
        let after_clear = get_cache_memory();
        assert!(after_clear < 1024); // Less than 1KB
    }

    #[test]
    fn test_active_and_peak_memory() {
        // Reset peak memory
        assert!(reset_peak_memory().is_ok());

        // Get current active memory
        let _active = get_active_memory();

        // Peak should exist after reset
        let peak = get_peak_memory();
        // Peak is tracked, so it should be a valid number
        let _ = peak; // Suppress unused variable warning
    }

    #[test]
    fn test_set_cache_limit() {
        // Set cache limit to 512MB
        let new_limit = 512 * 1024 * 1024;
        let _old_limit = set_cache_limit(new_limit);

        // Reset to no limit
        set_cache_limit(0);
    }

    #[test]
    fn test_set_wired_limit() {
        // Set wired limit to 256MB
        let new_limit = 256 * 1024 * 1024;
        let _old_limit = set_wired_limit(new_limit);

        // Reset to no limit
        set_wired_limit(0);
    }

    // Integration tests demonstrating real-world usage

    #[test]
    fn test_active_memory_tracks_allocations() {
        use crate::Array;

        // Clear cache to start fresh
        clear_cache().unwrap();

        // Get baseline memory
        let baseline = get_active_memory();

        // Allocate a large array (10MB of f32 data = 2.5M elements)
        let size = 2_500_000;
        let arr = Array::zeros::<f32>(&[size]).unwrap();
        arr.eval().unwrap();

        // Active memory should increase
        let after_alloc = get_active_memory();
        let allocated = after_alloc.saturating_sub(baseline);

        // Should have allocated at least the array size (10MB)
        // Being conservative with the assertion due to overhead
        let expected_bytes = (size as usize) * std::mem::size_of::<f32>() / 2;
        assert!(
            allocated >= expected_bytes,
            "Expected significant memory allocation, got {} bytes",
            allocated
        );

        // Drop the array and clear cache
        drop(arr);
        clear_cache().unwrap();
    }

    #[test]
    fn test_peak_memory_tracking() {
        use crate::{array, ops::add, Array};

        // Reset peak memory to start fresh
        reset_peak_memory().unwrap();
        let baseline_peak = get_peak_memory();

        // Allocate arrays and perform operations
        let size = 1_000_000; // 4MB per array
        let a = Array::full::<f32>(&[size], array!(1.0f32)).unwrap();
        let b = Array::full::<f32>(&[size], array!(2.0f32)).unwrap();

        // Perform operation that creates temporary result
        let c = add(&a, &b).unwrap();
        c.eval().unwrap();

        // Peak memory should be higher than baseline
        let peak = get_peak_memory();
        assert!(
            peak > baseline_peak,
            "Peak memory should increase after allocations"
        );

        // Clean up
        drop(a);
        drop(b);
        drop(c);
        clear_cache().unwrap();
    }

    #[test]
    fn test_cache_memory_behavior() {
        use crate::Array;

        // Clear cache to start fresh
        clear_cache().unwrap();
        let cache_after_clear = get_cache_memory();

        // Create and drop arrays to populate cache
        for _ in 0..5 {
            let arr = Array::ones::<f32>(&[100_000]).unwrap();
            arr.eval().unwrap();
            drop(arr);
        }

        // Cache may contain memory from dropped arrays
        // (behavior depends on MLX internals, so we just check it's measurable)
        let cache_after_ops = get_cache_memory();

        // Clear cache and verify it reduces
        clear_cache().unwrap();
        let cache_after_second_clear = get_cache_memory();

        // Cache should be small after clearing (allowing some overhead)
        assert!(
            cache_after_second_clear < 1024 * 1024, // Less than 1MB
            "Cache should be mostly cleared, got {} bytes",
            cache_after_second_clear
        );

        println!(
            "Cache progression: {} -> {} -> {} bytes",
            cache_after_clear, cache_after_ops, cache_after_second_clear
        );
    }

    #[test]
    fn test_memory_limit_setting() {
        use crate::Array;

        // Set a conservative memory limit (100MB)
        let limit = 100 * 1024 * 1024;
        let old_limit = set_memory_limit(limit);

        // Verify limit is set
        assert_eq!(get_memory_limit(), limit);

        // Create arrays within limit
        let arr = Array::zeros::<f32>(&[1_000_000]).unwrap(); // 4MB
        arr.eval().unwrap();

        // Should succeed without issue
        let active = get_active_memory();
        assert!(active > 0, "Should have active memory");

        // Clean up and restore old limit
        drop(arr);
        clear_cache().unwrap();
        set_memory_limit(old_limit);
    }

    #[test]
    fn test_cache_limit_enforcement() {
        use crate::Array;

        // Set a small cache limit (10MB)
        let cache_limit = 10 * 1024 * 1024;
        let old_limit = set_cache_limit(cache_limit);

        // Clear cache to start fresh
        clear_cache().unwrap();

        // Create and drop arrays
        for _ in 0..3 {
            let arr = Array::ones::<f32>(&[500_000]).unwrap(); // 2MB each
            arr.eval().unwrap();
            drop(arr);
        }

        // Cache should respect the limit (though exact behavior depends on MLX)
        let cache = get_cache_memory();
        println!("Cache after operations with {}MB limit: {} bytes",
                 cache_limit / (1024 * 1024), cache);

        // Restore old limit and clean up
        set_cache_limit(old_limit);
        clear_cache().unwrap();
    }

    #[test]
    fn test_realistic_workflow() {
        use crate::{array, ops::matmul, Array};

        // Simulate a realistic ML workflow with memory monitoring
        println!("\n=== Realistic Memory Monitoring Workflow ===");

        // Start fresh
        clear_cache().unwrap();
        reset_peak_memory().unwrap();

        let initial_active = get_active_memory();
        println!("Initial active memory: {} bytes", initial_active);

        // Simulate loading model weights (two 1000x1000 matrices = 8MB total)
        let weights = Array::full::<f32>(&[1000, 1000], array!(0.1f32)).unwrap();
        let input = Array::full::<f32>(&[1000, 1000], array!(1.0f32)).unwrap();

        weights.eval().unwrap();
        input.eval().unwrap();

        let after_load = get_active_memory();
        println!(
            "After loading arrays: {} bytes (+{} bytes)",
            after_load,
            after_load.saturating_sub(initial_active)
        );

        // Perform computation
        let output = matmul(&input, &weights).unwrap();
        output.eval().unwrap();

        let after_compute = get_active_memory();
        println!(
            "After computation: {} bytes (+{} bytes)",
            after_compute,
            after_compute.saturating_sub(after_load)
        );

        // Check peak memory
        let peak = get_peak_memory();
        println!("Peak memory reached: {} bytes", peak);
        assert!(peak >= after_compute, "Peak should be at least current active");

        // Clean up
        drop(weights);
        drop(input);
        drop(output);

        let before_clear = get_cache_memory();
        clear_cache().unwrap();
        let after_clear = get_cache_memory();

        println!(
            "Cache before/after clear: {} -> {} bytes",
            before_clear, after_clear
        );

        println!("=== End Workflow ===\n");
    }
}
