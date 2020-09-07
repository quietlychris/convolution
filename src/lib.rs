use ndarray::prelude::*;

pub mod two_dimensional;
pub mod three_dimensional;

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

    pub fn padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    pub fn stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    pub fn kernel(mut self, kernel: Array2<f32>) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn build(self) -> ConvHyperParam {
        ConvHyperParam {
            padding: self.padding,
            stride: self.stride,
            kernel: self.kernel
        }
    }

}
