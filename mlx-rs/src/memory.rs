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
}
