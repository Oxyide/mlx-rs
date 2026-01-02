//! Learning rate schedulers for optimizers.
//!
//! This module provides various learning rate scheduling strategies that can be used
//! to adjust the learning rate during training.
//!
//! # Example
//!
//! ```rust
//! use mlx_rs::optimizers::schedulers::{CosineDecay, Scheduler};
//!
//! let mut scheduler = CosineDecay::new(0.1, 1000, 0.0);
//!
//! // Get initial learning rate
//! assert_eq!(scheduler.get_lr(), 0.1);
//!
//! // Step through training
//! for _ in 0..10 {
//!     let lr = scheduler.step();
//!     // Use lr with optimizer: optimizer.lr = array!(lr);
//! }
//! ```

use std::f32::consts::PI;

/// Trait for learning rate schedulers.
///
/// Schedulers maintain an internal step counter and compute the learning rate
/// based on the current step.
pub trait Scheduler: std::fmt::Debug {
    /// Get the current learning rate without advancing the step counter.
    fn get_lr(&self) -> f32;

    /// Advance the step counter and return the new learning rate.
    fn step(&mut self) -> f32;

    /// Get the current step count.
    fn get_step(&self) -> usize;

    /// Reset the scheduler to its initial state.
    fn reset(&mut self);
}

/// Cosine decay scheduler.
///
/// The learning rate is decayed from `init` to `end` following a cosine curve over
/// `decay_steps` steps. After `decay_steps`, the learning rate remains constant at `end`.
///
/// The formula is:
/// ```text
/// lr = end + (init - end) * 0.5 * (1 + cos(π * step / decay_steps))
/// ```
/// for `step < decay_steps`, and `lr = end` for `step >= decay_steps`.
///
/// # Example
///
/// ```rust
/// use mlx_rs::optimizers::schedulers::{CosineDecay, Scheduler};
///
/// let mut scheduler = CosineDecay::new(0.1, 1000, 0.0);
/// let lr = scheduler.step();
/// ```
#[derive(Debug, Clone)]
pub struct CosineDecay {
    init: f32,
    decay_steps: usize,
    end: f32,
    step: usize,
}

impl CosineDecay {
    /// Create a new cosine decay scheduler.
    ///
    /// # Arguments
    ///
    /// * `init` - The initial learning rate
    /// * `decay_steps` - Number of steps over which to decay
    /// * `end` - The final learning rate (default: 0.0)
    pub fn new(init: f32, decay_steps: usize, end: f32) -> Self {
        Self {
            init,
            decay_steps,
            end,
            step: 0,
        }
    }
}

impl Scheduler for CosineDecay {
    fn get_lr(&self) -> f32 {
        if self.step >= self.decay_steps {
            self.end
        } else {
            let progress = self.step as f32 / self.decay_steps as f32;
            let cosine_decay = 0.5 * (1.0 + (PI * progress).cos());
            self.end + (self.init - self.end) * cosine_decay
        }
    }

    fn step(&mut self) -> f32 {
        let lr = self.get_lr();
        self.step += 1;
        lr
    }

    fn get_step(&self) -> usize {
        self.step
    }

    fn reset(&mut self) {
        self.step = 0;
    }
}

/// Exponential decay scheduler.
///
/// The learning rate is multiplied by `decay_rate` at each step.
///
/// The formula is:
/// ```text
/// lr = init * decay_rate^step
/// ```
///
/// # Example
///
/// ```rust
/// use mlx_rs::optimizers::schedulers::{ExponentialDecay, Scheduler};
///
/// let mut scheduler = ExponentialDecay::new(0.1, 0.9);
/// let lr = scheduler.step();  // 0.09
/// ```
#[derive(Debug, Clone)]
pub struct ExponentialDecay {
    init: f32,
    decay_rate: f32,
    step: usize,
}

impl ExponentialDecay {
    /// Create a new exponential decay scheduler.
    ///
    /// # Arguments
    ///
    /// * `init` - The initial learning rate
    /// * `decay_rate` - The multiplicative decay factor (e.g., 0.9 for 10% decay per step)
    pub fn new(init: f32, decay_rate: f32) -> Self {
        Self {
            init,
            decay_rate,
            step: 0,
        }
    }
}

impl Scheduler for ExponentialDecay {
    fn get_lr(&self) -> f32 {
        self.init * self.decay_rate.powi(self.step as i32)
    }

    fn step(&mut self) -> f32 {
        let lr = self.get_lr();
        self.step += 1;
        lr
    }

    fn get_step(&self) -> usize {
        self.step
    }

    fn reset(&mut self) {
        self.step = 0;
    }
}

/// Step decay scheduler.
///
/// The learning rate is multiplied by `decay_rate` every `step_size` steps.
///
/// The formula is:
/// ```text
/// lr = init * decay_rate^(floor(step / step_size))
/// ```
///
/// # Example
///
/// ```rust
/// use mlx_rs::optimizers::schedulers::{StepDecay, Scheduler};
///
/// let mut scheduler = StepDecay::new(0.1, 0.9, 10);
/// // Learning rate stays at 0.1 for steps 0-9
/// // Then drops to 0.09 at step 10
/// ```
#[derive(Debug, Clone)]
pub struct StepDecay {
    init: f32,
    decay_rate: f32,
    step_size: usize,
    step: usize,
}

impl StepDecay {
    /// Create a new step decay scheduler.
    ///
    /// # Arguments
    ///
    /// * `init` - The initial learning rate
    /// * `decay_rate` - The multiplicative decay factor
    /// * `step_size` - Number of steps between decay applications
    pub fn new(init: f32, decay_rate: f32, step_size: usize) -> Self {
        Self {
            init,
            decay_rate,
            step_size,
            step: 0,
        }
    }
}

impl Scheduler for StepDecay {
    fn get_lr(&self) -> f32 {
        let decay_count = self.step / self.step_size;
        self.init * self.decay_rate.powi(decay_count as i32)
    }

    fn step(&mut self) -> f32 {
        let lr = self.get_lr();
        self.step += 1;
        lr
    }

    fn get_step(&self) -> usize {
        self.step
    }

    fn reset(&mut self) {
        self.step = 0;
    }
}

/// Linear schedule scheduler.
///
/// The learning rate changes linearly from `init` to `end` over `steps` steps.
///
/// The formula is:
/// ```text
/// lr = init + (end - init) * (step / steps)
/// ```
/// for `step < steps`, and `lr = end` for `step >= steps`.
///
/// # Example
///
/// ```rust
/// use mlx_rs::optimizers::schedulers::{LinearSchedule, Scheduler};
///
/// // Warmup from 0.0 to 0.1 over 100 steps
/// let mut scheduler = LinearSchedule::new(0.0, 0.1, 100);
/// ```
#[derive(Debug, Clone)]
pub struct LinearSchedule {
    init: f32,
    end: f32,
    steps: usize,
    step: usize,
}

impl LinearSchedule {
    /// Create a new linear schedule scheduler.
    ///
    /// # Arguments
    ///
    /// * `init` - The initial learning rate
    /// * `end` - The final learning rate
    /// * `steps` - Number of steps over which to transition
    pub fn new(init: f32, end: f32, steps: usize) -> Self {
        Self {
            init,
            end,
            steps,
            step: 0,
        }
    }
}

impl Scheduler for LinearSchedule {
    fn get_lr(&self) -> f32 {
        if self.step >= self.steps {
            self.end
        } else {
            let progress = self.step as f32 / self.steps as f32;
            self.init + (self.end - self.init) * progress
        }
    }

    fn step(&mut self) -> f32 {
        let lr = self.get_lr();
        self.step += 1;
        lr
    }

    fn get_step(&self) -> usize {
        self.step
    }

    fn reset(&mut self) {
        self.step = 0;
    }
}

/// Join multiple schedulers with boundaries.
///
/// This scheduler switches between multiple sub-schedulers at specified step boundaries.
/// Each scheduler is used for a range of steps before switching to the next one.
///
/// # Example
///
/// ```rust
/// use mlx_rs::optimizers::schedulers::{JoinSchedules, LinearSchedule, CosineDecay, Scheduler};
///
/// // Warmup for 100 steps, then cosine decay for 900 steps
/// let warmup = Box::new(LinearSchedule::new(0.0, 0.1, 100));
/// let decay = Box::new(CosineDecay::new(0.1, 900, 0.0));
///
/// let mut scheduler = JoinSchedules::new(vec![warmup, decay], vec![100]);
/// ```
#[derive(Debug)]
pub struct JoinSchedules {
    schedulers: Vec<Box<dyn Scheduler>>,
    boundaries: Vec<usize>,
    step: usize,
    current_scheduler_idx: usize,
}

impl JoinSchedules {
    /// Create a new joined scheduler.
    ///
    /// # Arguments
    ///
    /// * `schedulers` - Vector of schedulers to use
    /// * `boundaries` - Step boundaries at which to switch schedulers.
    ///   Must have length `schedulers.len() - 1`.
    ///
    /// # Panics
    ///
    /// Panics if `boundaries.len() != schedulers.len() - 1` or if boundaries are not
    /// in ascending order.
    pub fn new(schedulers: Vec<Box<dyn Scheduler>>, boundaries: Vec<usize>) -> Self {
        assert_eq!(
            boundaries.len(),
            schedulers.len() - 1,
            "boundaries must have length schedulers.len() - 1"
        );

        // Check boundaries are in ascending order
        for i in 1..boundaries.len() {
            assert!(
                boundaries[i] > boundaries[i - 1],
                "boundaries must be in ascending order"
            );
        }

        Self {
            schedulers,
            boundaries,
            step: 0,
            current_scheduler_idx: 0,
        }
    }

    /// Update the current scheduler index based on the current step.
    fn update_current_scheduler(&mut self) {
        for (idx, &boundary) in self.boundaries.iter().enumerate() {
            if self.step < boundary {
                self.current_scheduler_idx = idx;
                return;
            }
        }
        self.current_scheduler_idx = self.schedulers.len() - 1;
    }
}

impl Scheduler for JoinSchedules {
    fn get_lr(&self) -> f32 {
        self.schedulers[self.current_scheduler_idx].get_lr()
    }

    fn step(&mut self) -> f32 {
        let lr = self.get_lr();
        self.schedulers[self.current_scheduler_idx].step();
        self.step += 1;
        self.update_current_scheduler();
        lr
    }

    fn get_step(&self) -> usize {
        self.step
    }

    fn reset(&mut self) {
        self.step = 0;
        self.current_scheduler_idx = 0;
        for scheduler in &mut self.schedulers {
            scheduler.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::Sgd;

    #[test]
    fn test_scheduler_with_optimizer() {
        // Example: Using a scheduler with an SGD optimizer
        let mut scheduler = StepDecay::new(0.1, 0.5, 10);
        let mut optimizer = Sgd::new(0.1);

        // Simulate training loop
        for step in 0..25 {
            // Update learning rate from scheduler
            let new_lr = scheduler.step();
            optimizer.lr = new_lr;

            // Expected learning rates:
            // Steps 0-9: 0.1
            // Steps 10-19: 0.05
            // Steps 20-24: 0.025
            let expected_lr = if step < 10 {
                0.1
            } else if step < 20 {
                0.05
            } else {
                0.025
            };

            float_eq::assert_float_eq!(optimizer.lr, expected_lr, abs <= 1e-6);
        }
    }

    #[test]
    fn test_warmup_then_decay() {
        // Example: Warmup for 100 steps, then cosine decay
        let warmup = Box::new(LinearSchedule::new(0.0, 0.1, 100));
        let decay = Box::new(CosineDecay::new(0.1, 900, 0.0));
        let mut scheduler = JoinSchedules::new(vec![warmup, decay], vec![100]);

        let mut optimizer = Sgd::new(0.0);

        // Check warmup phase (first few steps)
        for _ in 0..10 {
            optimizer.lr = scheduler.step();
        }
        // Should be warming up
        assert!(optimizer.lr > 0.0 && optimizer.lr < 0.1);

        // Skip to end of warmup
        for _ in 10..100 {
            optimizer.lr = scheduler.step();
        }
        // Should be close to 0.1 at end of warmup
        float_eq::assert_float_eq!(optimizer.lr, 0.1, abs <= 1e-2);

        // Start decay phase
        for _ in 100..200 {
            optimizer.lr = scheduler.step();
        }
        // Should be decaying
        assert!(optimizer.lr < 0.1);
    }

    #[test]
    fn test_cosine_decay() {
        let mut scheduler = CosineDecay::new(1.0, 100, 0.0);

        // Initial learning rate
        assert_eq!(scheduler.get_lr(), 1.0);
        assert_eq!(scheduler.get_step(), 0);

        // Step and check
        let lr = scheduler.step();
        assert_eq!(lr, 1.0);
        assert_eq!(scheduler.get_step(), 1);

        // After decay_steps, should be at end value
        for _ in 0..99 {
            scheduler.step();
        }
        assert_eq!(scheduler.get_step(), 100);
        let final_lr = scheduler.get_lr();
        float_eq::assert_float_eq!(final_lr, 0.0, abs <= 1e-6);

        // Reset
        scheduler.reset();
        assert_eq!(scheduler.get_step(), 0);
        assert_eq!(scheduler.get_lr(), 1.0);
    }

    #[test]
    fn test_exponential_decay() {
        let mut scheduler = ExponentialDecay::new(1.0, 0.9);

        assert_eq!(scheduler.get_lr(), 1.0);

        let lr1 = scheduler.step();
        assert_eq!(lr1, 1.0);

        let lr2 = scheduler.step();
        float_eq::assert_float_eq!(lr2, 0.9, abs <= 1e-6);

        let lr3 = scheduler.step();
        float_eq::assert_float_eq!(lr3, 0.81, abs <= 1e-6);
    }

    #[test]
    fn test_step_decay() {
        let mut scheduler = StepDecay::new(1.0, 0.5, 10);

        // First 10 steps should have lr = 1.0
        for i in 0..10 {
            let lr = scheduler.step();
            assert_eq!(lr, 1.0, "Step {}", i);
        }

        // Next 10 steps should have lr = 0.5
        for i in 0..10 {
            let lr = scheduler.step();
            float_eq::assert_float_eq!(lr, 0.5, abs <= 1e-6, "Step {}", i + 10);
        }

        // Next 10 steps should have lr = 0.25
        for i in 0..10 {
            let lr = scheduler.step();
            float_eq::assert_float_eq!(lr, 0.25, abs <= 1e-6, "Step {}", i + 20);
        }
    }

    #[test]
    fn test_linear_schedule() {
        let mut scheduler = LinearSchedule::new(0.0, 1.0, 10);

        assert_eq!(scheduler.get_lr(), 0.0);

        // Should increase linearly
        for i in 0..10 {
            let expected = i as f32 / 10.0;
            let lr = scheduler.step();
            float_eq::assert_float_eq!(lr, expected, abs <= 1e-6);
        }

        // After steps, should stay at end value
        let lr = scheduler.get_lr();
        assert_eq!(lr, 1.0);
    }

    #[test]
    fn test_join_schedules() {
        // Warmup for 10 steps, then decay for 10 steps
        let warmup = Box::new(LinearSchedule::new(0.0, 1.0, 10));
        let decay = Box::new(LinearSchedule::new(1.0, 0.0, 10));

        let mut scheduler = JoinSchedules::new(vec![warmup, decay], vec![10]);

        // First 10 steps: warmup
        for i in 0..10 {
            let expected = i as f32 / 10.0;
            let lr = scheduler.step();
            float_eq::assert_float_eq!(lr, expected, abs <= 1e-6, "Warmup step {}", i);
        }

        // Next 10 steps: decay
        for i in 0..10 {
            let expected = 1.0 - (i as f32 / 10.0);
            let lr = scheduler.step();
            float_eq::assert_float_eq!(lr, expected, abs <= 1e-6, "Decay step {}", i);
        }
    }

    #[test]
    #[should_panic(expected = "boundaries must have length schedulers.len() - 1")]
    fn test_join_schedules_invalid_boundaries() {
        let s1 = Box::new(LinearSchedule::new(0.0, 1.0, 10));
        let s2 = Box::new(LinearSchedule::new(1.0, 0.0, 10));

        // Wrong number of boundaries
        JoinSchedules::new(vec![s1, s2], vec![10, 20]);
    }

    #[test]
    #[should_panic(expected = "boundaries must be in ascending order")]
    fn test_join_schedules_unordered_boundaries() {
        let s1 = Box::new(LinearSchedule::new(0.0, 1.0, 10));
        let s2 = Box::new(LinearSchedule::new(1.0, 0.5, 10));
        let s3 = Box::new(LinearSchedule::new(0.5, 0.0, 10));

        // Boundaries not in ascending order
        JoinSchedules::new(vec![s1, s2, s3], vec![20, 10]);
    }
}
