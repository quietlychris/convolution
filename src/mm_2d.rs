use ndarray::prelude::*;
use std::error::Error;

use crate::prelude::*;

pub fn mm_convolution_2d(input: Array2<f32>, hp: &ConvHyperParam) -> Result<Array2<f32>, Box<dyn Error>> {
    let input = pad_2d(input, hp.padding);

    let (i_n, i_m) = (input.nrows(), input.ncols());
    let (k_n, k_m) = (hp.kernel.nrows(), hp.kernel.ncols());

    let flat_kernel = return_flat_kernel_2d(hp)?;
    let o_n = ((i_n - k_n) as f32 / hp.stride.0 as f32).floor() as usize + 1;
    let o_m = ((i_m - k_m) as f32 / hp.stride.1 as f32).floor() as usize + 1;

    let altered_input = return_alternate_input(input, &hp, (o_n, o_m))?;
    

    let output = flat_kernel.dot(&altered_input).into_shape((o_n, o_m))?;
    Ok(output)

}

fn return_alternate_input(input: Array2<f32>, hp: &ConvHyperParam, (o_n, o_m): (usize, usize)) -> Result<Array2<f32>, Box<dyn Error>> {
    let (k_n, k_m) = (hp.kernel.nrows(), hp.kernel.ncols());
    
    let mut alternate_input: Array2<f32> = Array2::zeros((k_n * k_m, o_m * o_n));
   
    for y in 0..o_n {
        for x in 0..o_m {
            let (i_y, i_x) = (y * hp.stride.0, x * hp.stride.1);
            let temp = input
                .slice(s![i_y..(i_y + k_n), i_x..(i_x + k_m)])
                .to_owned()
                .into_shape((k_n * k_m, 1))?;
            // println!("temp with shape {:?}:\n{:#?}\n", temp.shape(), temp);
            alternate_input.slice_mut(s![0..(k_n * k_m), (y * o_m) + x]).assign(&temp.slice(s![..,0]));
        }
    }

    Ok(alternate_input)
}

pub fn return_flat_kernel_2d(hp: &ConvHyperParam) -> Result<Array2<f32>, Box<dyn Error>> {
    let (k_n, k_m) = (hp.kernel.nrows(), hp.kernel.ncols());
    let flat_kernel = hp.kernel.clone().into_shape((1, k_n * k_m))?;
    Ok(flat_kernel)
}

#[test]
fn test_mm_convolution_2d() {
    #[rustfmt::skip]
    let input = array![
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 2.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 3.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 4.0, 0.0]
    ];

    #[rustfmt::skip]
    let kernel = array![
        [1.0, 1.0, 1.0], 
        [0.0, 0.0, 0.0], 
        [-1.0, -1.0, -1.0]
    ];

    let hp = ConvHyperParam::default(kernel).stride((1, 1)).padding(0).build();

    let mm_output = mm_convolution_2d(input.clone(), &hp).expect("Error occurred while running matrix-multiplied convolution");
    println!("mm_output:\n{:#?}",mm_output);
    let sliding_output = convolution_2d(input, &hp).unwrap();
    println!("sliding_output:\n{:#?}",sliding_output);

    assert_eq!(mm_output, sliding_output);
}


#[test]
fn test_return_flat_kernel() {
    #[rustfmt::skip]
    let kernel: Array2<f32> = array![
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -1.0, -1.0]
    ];
    let hp = ConvHyperParam::default(kernel).stride((1, 1)).padding(0).build();

    let flat_kernel = return_flat_kernel_2d(&hp).unwrap();
    println!("flat_kernel: {:#?}", flat_kernel);

    let ideal = array![[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0]];
    assert_eq!(ideal, flat_kernel);
}
