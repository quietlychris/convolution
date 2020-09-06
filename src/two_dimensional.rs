use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

use image::*;
use std::error::Error;

use crate::ConvHyperParam;

pub fn convolution_2d(input: Array2<f32>, hp: &ConvHyperParam) -> Result<Array2<f32>, Box<dyn Error>> {
    let padding = 1;
    let input = pad_2d(input, padding);

    let (stride_n, stride_m) = (hp.stride.0, hp.stride.1);
    let (i_n, i_m) = (input.nrows(), input.ncols());
    let (k_n, k_m) = (hp.kernel.nrows(), hp.kernel.ncols());
    // These o_n,m terms don't include the padding values, since we're already taking those
    // into account when we calculuate the input (n,m) values here
    // Otherwise, we'd have an extra (2 * padding) term in the numerator
    let o_n = ((i_n - k_n) as f32 / stride_n as f32).floor() as usize + 1;
    let o_m = ((i_m - k_m) as f32 / stride_m as f32).floor() as usize + 1;

    let mut output: Array2<f32> = Array2::zeros((o_n, o_m));
    run_convolution_2d(&hp.kernel, &input, &mut output);

    Ok(output)
}

fn run_convolution_2d(kernel: &Array2<f32>, input: &Array2<f32>, output: &mut Array2<f32>) {
    let (i_n, i_m) = (input.shape()[0], input.shape()[1]);
    let (k_n, k_m) = (kernel.shape()[0], kernel.shape()[1]);
    let (o_n, o_m) = (output.shape()[0], output.shape()[1]);

    // println!("{:#?}", output);
    for y in 0..o_n {
        for x in 0..o_m {
            let temp = &input.slice(s![y..(y + k_n), x..(x + k_m)]) * kernel;
            output[[y, x]] = temp.sum();
            // output.slice_mut(s![x..x+k_m, y..y+k_n]).assign(&temp);
        }
    }
}

fn pad_2d(input: Array2<f32>, padding: usize) -> Array2<f32> {
    let (n, m) = (input.nrows(), input.ncols());

    let mut out: Array2<f32> = Array2::zeros((n + (padding * 2), m + (padding * 2)));
    out.slice_mut(s![padding..n + padding, padding..m + padding])
        .assign(&input);
    out
}

pub fn open_grayimage_and_convert_to_ndarray2(path: &str) -> Result<Array2<f32>, ImageError> {
    let img = image::open(&path)?.to_luma();

    let (w, h) = img.dimensions();
    let (w, h) = (w as usize, h as usize);
    println!("img dimensions: ({},{})", w, h);

    let mut array = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            array[[y, x]] = img.get_pixel(x as u32, y as u32)[0] as f32;
        }
    }

    Ok(array)
}
