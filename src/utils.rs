use ndarray::prelude::*;
use ndarray_stats::QuantileExt;

use image::*;
use std::error::Error;

//--- TWO-DIMENSIONAL ---

pub fn pad_2d(input: Array2<f32>, padding: usize) -> Array2<f32> {
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

//--- THREE-DIMENSIONAL ---

// Expects a (3,n,m) dimensioned input
pub fn pad_3d(input: Array3<f32>, padding: usize) -> Array3<f32> {
    let dims = input.dim();
    println!("input dimensions are: {:?}", dims);
    let (n, m) = (dims.1, dims.2);
    println!("with an (n,m) of ({},{})", n, m);

    let mut out: Array3<f32> = Array3::zeros((3, n + (padding * 2), m + (padding * 2)));
    // Can this be done in parallel using iterators + Rayon?
    for i in 0..3 {
        out.slice_mut(s![i, padding..n + padding, padding..m + padding])
            .assign(&input.slice(s![i, .., ..]));
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
