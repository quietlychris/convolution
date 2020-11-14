use ndarray::prelude::*;
use std::error::Error;

use crate::prelude::*;


pub fn native_mm_convolution_3d(input: Array3<f32>, hp: &ConvHyperParam) -> Result<Array3<f32>, Box<dyn Error>> {
    // Assuming number of channels is 3
    let input = pad_3d(input, hp.padding);
    let channels = 3;

    let (i_n, i_m) = (input.dim().0, input.dim().1);
    let (k_n, k_m) = (hp.kernel.nrows(), hp.kernel.ncols());

    let flat_kernel = return_flat_kernel_3d(hp)?;

    dbg!("sucessfully built flat kernel");

    let o_n = ((i_n - k_n) as f32 / hp.stride.0 as f32).floor() as usize + 1;
    let o_m = ((i_m - k_m) as f32 / hp.stride.1 as f32).floor() as usize + 1;

    let mut altered_input: Array2<f32> = Array2::zeros((k_n * k_m * channels, o_m * o_n));
   
    for c in 0..channels {
        for y in 0..o_n {
            for x in 0..o_m {
                let (i_y, i_x) = (y * hp.stride.0, x * hp.stride.1);
                let temp = input
                    .slice(s![i_y..(i_y + k_n), i_x..(i_x + k_m), c])
                    .to_owned()
                    .into_shape((k_n * k_m, 1))?;
                // println!("temp with shape {:?}:\n{:#?}\n", temp.shape(), temp);
                let y_start = c * (k_n * k_m);
                let y_end = y_start.clone() + (k_n * k_m);
                altered_input.slice_mut(s![y_start..y_end, (y * o_m) + x]).assign(&temp.slice(s![..,0]));
            }
        }
    }

    dbg!("successfully build altered output");   
    
    println!("trying to multiply flat kernel with dims {:?} by altered input of dims {:?}",flat_kernel.dim(), altered_input.dim());

    let output = flat_kernel.dot(&altered_input).into_shape((o_n, o_m, channels)).expect("Couldn't put the output into the assigned shape");
    Ok(output)

}

/// This the three-dimensional variation
fn return_flat_kernel_3d(hp: &ConvHyperParam) -> Result<Array2<f32>, Box<dyn Error>> {
    // Note: this is the three-dimensional

    let (k_n, k_m) = (hp.kernel.nrows(), hp.kernel.ncols());
    let flat_kernel_base = hp.kernel.clone().into_shape((1, k_n * k_m))?;
    /*
    let mut flat_kernel = Array2::zeros((1, k_n * k_m * 3));
    for i in 0..3 {
        let start = i * (k_n * k_m);
        let end = start + (k_n * k_m);
        flat_kernel.slice_mut(s![0, start..end]).assign(&flat_kernel_base.slice(s![..,0]));
    }*/
    let mut flat_kernel = Array2::zeros((3, k_n * k_m));
    for i in 0..3 {
        flat_kernel.slice_mut(s![i, 0..(k_n * k_m)]).assign(&flat_kernel_base.slice(s![..,0]));
    }

    Ok(flat_kernel)
}

pub fn mm_convolution_3d(input: Array3<f32>, hp: &ConvHyperParam) -> Result<Array3<f32>, Box<dyn Error>> {
    
    let (i_n, i_m) = (input.dim().1, input.dim().2);
    println!("the input is of shape {:?}, leading to an (i_n, i_m) of {:?}",input.shape(),(i_n,i_m));

    let (k_n, k_m) = (hp.kernel.nrows(), hp.kernel.ncols());

    let o_n = ((i_n - k_n) as f32 / hp.stride.0 as f32).floor() as usize + 1 + hp.padding;
    let o_m = ((i_m - k_m) as f32 / hp.stride.1 as f32).floor() as usize + 1 + hp.padding;

    let channels = 3;

    // let mut temp_channel_out = Array2::zeros((o_n, o_m));
    let mut output = Array3::zeros((channels, o_n, o_m));
    // println!()
    for c in 0..channels {
        let channel_input = input.slice(s![c, .., ..]).to_owned().into_shape((i_n, i_m))?;
        println!("the channel input shape is: {:?}",channel_input.shape());
        let channel_output = mm_convolution_2d(channel_input, hp)?;
        println!("the channel output shape is: {:?}",channel_output.shape());
        println!("which we want to put into a shape of: {:?}",output.shape());
        output.slice_mut(s![c, .., ..]).assign(&channel_output.slice(s![..,..]));
    }
    Ok(output)

}

#[test]
fn small_mm_3d_convolution() {

    let input = array![
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0]
            ],
            [
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0]
            ],
            [
                [3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0]
            ]
        ];

        let kernel_v = array![[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]];
        let hp_vert = ConvHyperParam::default(kernel_v).stride((1, 1)).build();

        let output = mm_convolution_3d(input, &hp_vert).unwrap();

        println!("output: {:?}",output);
}

