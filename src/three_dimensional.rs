use ndarray::prelude::*;

use image::*;
use std::error::Error;

use crate::ConvHyperParam;

pub fn convolution_3d(input: Array3<f32>, hp: &ConvHyperParam) -> Result<Array3<f32>, Box<dyn Error>> {
    let input = pad_3d(input, hp.padding);

    let (stride_n, stride_m) = (hp.stride.0, hp.stride.1);
    let i_dims = input.dim();
    let (i_n, i_m) = (i_dims.1, i_dims.2);
    let (k_n, k_m) = (hp.kernel.nrows(), hp.kernel.ncols());
    // These o_n,m terms don't include the padding values, since we're already taking those
    // into account when we calculuate the input (n,m) values here
    // Otherwise, we'd have an extra (2 * padding) term in the numerator
    let o_n = ((i_n - k_n) as f32 / stride_n as f32).floor() as usize + 1;
    let o_m = ((i_m - k_m) as f32 / stride_m as f32).floor() as usize + 1;

    let mut output: Array3<f32> = Array3::zeros((3, o_n, o_m));
    run_convolution_3d(&hp.kernel, &input, &mut output);

    Ok(output)
}

fn run_convolution_3d(kernel: &Array2<f32>, input: &Array3<f32>, output: &mut Array3<f32>) {
    let i_dims = input.dim();
    let (i_n, i_m) = (i_dims.1, i_dims.2);
    let (k_n, k_m) = (kernel.nrows(), kernel.ncols());
    let o_dims = output.dim();
    let (o_n, o_m) = (o_dims.1, o_dims.2);

    // println!("{:#?}", output);
    for c in 0..i_dims.0 { // c is for "color"
        for y in 0..o_n {
            for x in 0..o_m {
                let temp = &input.slice(s![c, y..(y + k_n), x..(x + k_m)]) * kernel;
                output[[c, y, x]] = temp.sum();
            }
        }
    }
}

// Expects a (3,n,m) dimensioned input
pub fn pad_3d(input: Array3<f32>, padding: usize) -> Array3<f32> {
    let dims = input.dim();
    println!("input dimensions are: {:?}",dims);
    let (n, m) = (dims.1, dims.2);
    println!("with an (n,m) of ({},{})",n,m);

    let mut out: Array3<f32> = Array3::zeros((3, n + (padding * 2), m + (padding * 2)));
    // Can this be done in parallel using iterators + Rayon?
    for i in 0..3 {
        out.slice_mut(s![i,padding..n + padding, padding..m + padding])
        .assign(&input.slice(s![i,..,..]));
    }

    out
}


/// Helper function for transitioning between an `Image::RgbImage` input and an NdArray3<u8> structure
pub fn open_rgb_image_and_convert_to_ndarray3(path: &str) -> Result<Array3<f32>, Box<dyn Error>> {

    let img = image::open(&path)?;
    let (w, h) = img.dimensions();

    let mut arr = Array3::<f32>::zeros((3, h as usize, w as usize));
    for y in 0..h {
        for x in 0..w {
            let pixel = img.get_pixel(x, y);
            arr[[0usize, y as usize, x as usize]] = pixel[0] as f32;
            arr[[1usize, y as usize, x as usize]] = pixel[1] as f32;
            arr[[2usize, y as usize, x as usize]] = pixel[2] as f32;
        }
    }
    Ok(arr)
}
