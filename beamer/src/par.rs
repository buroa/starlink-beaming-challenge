//! Parallelism shim for the solver's hot paths.
//!
//! With the `parallel` feature (default) this re-exports rayon's prelude, so the
//! `par_iter` / `into_par_iter` call sites in [`crate::assign`] and
//! [`crate::feasibility`] run on rayon's work-stealing pool — OS threads
//! natively, or Web Workers in the browser when built with `wasm-bindgen-rayon`.
//!
//! Without the feature the *same* call sites resolve to sequential iterators, so
//! the solver compiles with no thread/atomics requirement (e.g. stable
//! `wasm32-unknown-unknown`, or any `--no-default-features` build). Every
//! down-stream adapter the solver uses (`map`, `enumerate`, `collect`) is a
//! method common to both `Iterator` and rayon's `ParallelIterator`, so nothing
//! at the call sites changes. The solver is deterministic, so the serial and
//! parallel builds produce bit-identical solutions.

#[cfg(feature = "parallel")]
pub use rayon::prelude::*;

#[cfg(not(feature = "parallel"))]
pub use serial::*;

#[cfg(not(feature = "parallel"))]
mod serial {
    /// Sequential stand-in for `rayon::iter::IntoParallelIterator`. Forwards to
    /// `IntoIterator`, so e.g. `(0..n).into_par_iter()` becomes `(0..n).into_iter()`.
    pub trait IntoParallelIterator {
        type Iter: Iterator<Item = Self::Item>;
        type Item;
        fn into_par_iter(self) -> Self::Iter;
    }

    impl<T: IntoIterator> IntoParallelIterator for T {
        type Iter = T::IntoIter;
        type Item = T::Item;
        fn into_par_iter(self) -> Self::Iter {
            self.into_iter()
        }
    }

    /// Sequential stand-in for `rayon::iter::IntoParallelRefIterator`. Forwards to
    /// the slice/Vec `iter()`, so `xs.par_iter()` becomes `xs.iter()`.
    pub trait IntoParallelRefIterator<'a> {
        type Iter: Iterator<Item = Self::Item>;
        type Item: 'a;
        fn par_iter(&'a self) -> Self::Iter;
    }

    impl<'a, T: 'a> IntoParallelRefIterator<'a> for [T] {
        type Iter = std::slice::Iter<'a, T>;
        type Item = &'a T;
        fn par_iter(&'a self) -> Self::Iter {
            self.iter()
        }
    }

    impl<'a, T: 'a> IntoParallelRefIterator<'a> for Vec<T> {
        type Iter = std::slice::Iter<'a, T>;
        type Item = &'a T;
        fn par_iter(&'a self) -> Self::Iter {
            self.iter()
        }
    }
}
