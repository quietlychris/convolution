use ndarray::prelude::*;
use std::error::Error;

use crate::prelude::*;

pub fn mm_convolution_3d(input: Array3<f32>, hp: &ConvHyperParam) -> Result<Array3<f32>, Box<dyn Error>> {
    
    let (i_n, i_m) = (input.dim().1, input.dim().2);
    println!("the input is of shape {:?}, leading to an (i_n, i_m) of {:?}",input.shape(),(i_n,i_m));

    let (k_n, k_m) = (hp.kernel.nrows(), hp.kernel.ncols());

    let o_n = ((i_n - k_n) as f32 / hp.stride.0 as f32).floor() as usize + 1 + hp.padding;
    let o_m = ((i_m - k_m) as f32 / hp.stride.1 as f32).floor() as usize + 1 + hp.padding;

    let channels = 3;

    // let mut temp_channel_out = Array2::zeros((o_n, o_m));
    let mut output = Array3::zeros((channels, o_n, o_m));

    for c in 0..channels {
        let channel_input = input.slice(s![c, .., ..]).to_owned().into_shape((i_n, i_m))?;
        // println!("the channel input shape is: {:?}",channel_input.shape());
        let channel_output = mm_convolution_2d(channel_input, hp)?;
        // println!("the channel output shape is: {:?}",channel_output.shape());
        // println!("which we want to put into a shape of: {:?}",output.shape());
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

