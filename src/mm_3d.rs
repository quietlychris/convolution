use ndarray::prelude::*;
use std::iter::FromIterator;

use std::error::Error;

use crate::utils::*;
use crate::ConvHyperParam;

pub fn kernel_to_weights_matrix(hp: &ConvHyperParam, input: &Array2<f32>) -> Result<Array2<f32>, Box<dyn Error>> {
    let (stride_n, stride_m) = (hp.stride.0, hp.stride.1);
    let (i_n, i_m) = (input.nrows(), input.ncols());
    let (k_n, k_m) = (hp.kernel.nrows(), hp.kernel.ncols());
    let o_n = ((i_n - k_n) as f32 / hp.stride.0 as f32).floor() as usize + 1;
    let o_m = ((i_m - k_m) as f32 / hp.stride.1 as f32).floor() as usize + 1;
    // Build the blank weights matrix
    let (w_n, w_m) = (o_m * o_n, (i_n * i_m) + stride_m);
    let mut weights = Array2::zeros((w_n, w_m));

    let kernel_subunit_length = k_m + stride_m; // Length needed to assign each row of the kernel to the flattened unit
    let flat_kernel_length = kernel_subunit_length * k_n;
    let mut flat_kernel = Array2::zeros((1, flat_kernel_length));
    for kernel_row in 0..k_n {
        flat_kernel
            .slice_mut(s![0, (k_m + stride_m) * kernel_row..(k_m + stride_m) * kernel_row + k_m])
            .assign(&hp.kernel.slice(s![kernel_row, 0..k_m]));
    }
    // println!("flat_kernel:\n{:#?}", flat_kernel);

    let mut weight_row = 0;
    for row in 0..o_n {
        for slide in 0..o_m {
            // println!("in iteration for Row {}, Slide {}",row,slide);
            // println!("dropping on weight row #{}",weight_row);
            // QUARTER VIEW
            //weights.slice_mut(s![weight_row, (row*i_m + slide)..(row*i_m + slide) + flat_kernel_length]).assign(&flat_kernel.slice(s![0,0..flat_kernel_length]));
            // HALF VIEW
            // weights.slice_mut(s![weight_row, (row*i_m + (slide*stride_m))..(row*i_m + (slide*stride_m)) + flat_kernel_length]).assign(&flat_kernel.slice(s![0,0..flat_kernel_length]));
            //
            // FULL VIEW FOR SQUARE STRIDES
            weights
                .slice_mut(s![
                    weight_row,
                    (row * i_m * stride_m + (slide * stride_m))..(row * i_m * stride_m + (slide * stride_m)) + flat_kernel_length
                ])
                .assign(&flat_kernel.slice(s![0, 0..flat_kernel_length]));
            //
            // TO_DO: When not presented with a square value for strides, the convolution doesn't scale nicely
            // weights.slice_mut(s![weight_row, (row*i_m*stride_n + (slide*stride_m))..(row*i_m*stride_n + (slide*stride_m)) + flat_kernel_length]).assign(&flat_kernel.slice(s![0,0..flat_kernel_length]));
            weight_row += 1;
        }
    }

    // println!("Intermediate weights:\n{:#?}",weights);
    // TO_DO: Not sure we're splitting at the right point
    let (weights, _) = weights.view().split_at(Axis(1), w_m - stride_m);

    Ok(weights.to_owned())
}

fn run_mm_convolution_3d(hp: &ConvHyperParam, input: &Array3<f32>, output: &mut Array3<f32>) -> Result<(), Box<dyn Error>> {
    let weights = kernel_to_weights_matrix(&hp, &input.slice(s![0, .., ..]).to_owned()).expect("Error creating the weights matrix");
    // println!("weights:\n{:#?}", weights);

    let i_dims = input.dim();
    let (i_n, i_m) = (i_dims.1, i_dims.2);
    let (k_n, k_m) = (hp.kernel.shape()[0], hp.kernel.shape()[1]);

    let o_n = ((i_n - k_n) as f32 / hp.stride.0 as f32).floor() as usize + 1;
    let o_m = ((i_m - k_m) as f32 / hp.stride.1 as f32).floor() as usize + 1;

    for c in 0..3 {
        let flat_input = Array::from_iter(input.slice(s![c, .., ..]).iter().cloned());
        let mm_output = weights.dot(&flat_input).into_shape((o_n, o_m)).unwrap();
        for y in 0..o_n {
            for x in 0..o_m {
                output[[c, y, x]] = mm_output[[y, x]];
            }
        }
    }
    Ok(())
}

pub fn mm_convolution_3d(input: Array3<f32>, hp: &ConvHyperParam) -> Result<Array3<f32>, Box<dyn Error>> {
    let input = pad_3d(input, hp.padding);
    // println!("padded mm_input:\n{:#?}",input);

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
    run_mm_convolution_3d(&hp, &input, &mut output)?;

    Ok(output)
}
