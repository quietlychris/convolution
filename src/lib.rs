use ndarray::prelude::*;

pub mod two_dimensional;

pub struct ConvHyperParam {
    padding: usize,
    stride: (usize, usize),
    kernel: Array2<f32>,
}

impl ConvHyperParam {
    pub fn new(padding: usize, stride: (usize, usize), kernel: Array2<f32>) -> Self {
        ConvHyperParam {
            padding: padding,
            stride: stride,
            kernel: kernel,
        }
    }

    pub fn default(kernel: Array2<f32>) -> Self {
        ConvHyperParam::new(0, (1, 1), kernel)
    }
}
