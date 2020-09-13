#![allow(unused_variables)]
#[macro_use]
extern crate serial_test;

pub mod hyperparams;

pub mod sliding_2d;
pub mod sliding_3d;

pub mod mm_2d;
pub mod mm_3d;

pub mod utils;

mod tests;

pub mod prelude {

    pub use crate::sliding_2d::*;
    pub use crate::sliding_3d::*;

    pub use crate::mm_2d::*;
    pub use crate::mm_3d::*;

    pub use crate::hyperparams::*;
    pub use crate::utils::*;
}
