use ndarray::prelude::*;
use std::iter::FromIterator;

use std::error::Error;

use crate::prelude::*;

pub fn kernel_to_weights_matrix(hp: &ConvHyperParam, input: &Array2<f32>) -> Result<Array2<f32>, Box<dyn Error>> {
    let (stride_n, stride_m) = (hp.stride.0, hp.stride.1);
    let (i_n, i_m) = (input.nrows(), input.ncols());
    let (k_n, k_m) = (hp.kernel.nrows(), hp.kernel.ncols());
    let o_n = ((i_n - k_n) as f32 / stride_n as f32).floor() as usize + 1;
    let o_m = ((i_m - k_m) as f32 / stride_n as f32).floor() as usize + 1;
    println!("output size should be: {:?}", (o_n, o_m));

    let kernel_subunit_length = k_m + stride_m; // Length needed to assign each row of the kernel to the flattened unit
    let flat_kernel_length = kernel_subunit_length * k_n;
    let mut flat_kernel = Array2::zeros((1, flat_kernel_length));
    for kernel_row in 0..k_n {
        flat_kernel
            .slice_mut(s![0, (k_m + stride_m) * kernel_row..(k_m + stride_m) * kernel_row + k_m])
            .assign(&hp.kernel.slice(s![kernel_row, 0..k_m]));
    }
    println!("flat_kernel:\n{:#?}", flat_kernel);

    let meta_kernel_width = flat_kernel_length + (stride_m * o_n);
    let mut meta_kernel = Array2::zeros((o_m, meta_kernel_width));
    for mk_row in 0..meta_kernel.nrows() {
        let start_x = mk_row * stride_m;
        meta_kernel
            .slice_mut(s![mk_row, start_x..start_x + flat_kernel_length])
            .assign(&flat_kernel.slice(s![0, 0..flat_kernel_length]))
    }
    println!("meta_kernel:\n{:#.1?}", meta_kernel);

    // Build the blank weights matrix
    println!("in the weight building function: (o_n, o_m) = {:?}", (o_n, o_m));
    let (w_n, w_m) = (o_n * o_m, (i_n * i_m) + (o_n * stride_m) + stride_m);
    println!("size of the weights matrix will be: {:?}", (w_n, w_m));
    let mut weights = Array2::zeros((w_n, w_m));
    println!("blank weights:\n{:#.1?}", &weights);

    //  TO_DO: Constructing the weights in a generalized way
    for pass in 0..o_n {
        //for pass in 0..o_n {
        let start_x = if pass == 0 { 0 } else { 
            // I think most of the work needs to be done here
            pass * o_n +  
        };
        let start_y = if pass == 0 {
            0
        } else {
            // Remember the metakernel has o_n rows
            // I think this part is correct
            pass * o_m
        };
        weights
            .slice_mut(s![start_y..start_y + o_m, start_x..(start_x + meta_kernel_width)])
            .assign(&meta_kernel.slice(s![0..o_m, 0..meta_kernel_width]));
    }

    // TO_DO: Not sure we're splitting at the right point
    // let (weights, _) = weights.view().split_at(Axis(1), w_m - (o_n + stride_m));
    let (weights, _) = weights.view().split_at(Axis(1), i_m * i_n);
    println!("weights:\n{:#.1?}", &weights);

    Ok(weights.to_owned())
}

fn run_mm_convolution_2d(hp: &ConvHyperParam, input: &Array2<f32>, output: &mut Array2<f32>) {
    let flat_input = Array::from_iter(input.iter().cloned());
    println!("Shape of flat_input: {:?}", flat_input.shape());
    let weights = kernel_to_weights_matrix(&hp, &input).expect("Error creating the weights matrix");
    // println!("Shape of flat_input: {:?}",flat_input.shape());
    println!("Shape of weights matrix: {:?}", weights.shape());
    //println!("weights:\n{:#.2?}", weights);

    let (i_n, i_m) = (input.nrows(), input.ncols());
    let (k_n, k_m) = (hp.kernel.shape()[0], hp.kernel.shape()[1]);

    let o_n = ((i_n - k_n) as f32 / hp.stride.0 as f32).floor() as usize + 1;
    let o_m = ((i_m - k_m) as f32 / hp.stride.1 as f32).floor() as usize + 1;
    println!("output should have shape: {:?}", (o_n, o_m));

    let mm_output = weights.dot(&flat_input);
    let mm_output = mm_output.clone().into_shape((o_n, o_m)).expect(&format!(
        "Error building mm_output of shape {:?} into ({},{})",
        mm_output.shape(),
        o_n,
        o_m
    ));
    for y in 0..o_n {
        for x in 0..o_m {
            output[[y, x]] = mm_output[[y, x]];
        }
    }
}

pub fn mm_convolution_2d(input: Array2<f32>, hp: &ConvHyperParam) -> Result<Array2<f32>, Box<dyn Error>> {
    let input = pad_2d(input, hp.padding);
    // println!("padded mm_input:\n{:#?}",input);

    let (i_n, i_m) = (input.nrows(), input.ncols());
    let (k_n, k_m) = (hp.kernel.nrows(), hp.kernel.ncols());
    // These o_n,m terms don't include the padding values, since we're already taking those
    // into account when we calculuate the input (n,m) values here
    // Otherwise, we'd have an extra (2 * padding) term in the numerator
    let o_n = ((i_n - k_n) as f32 / hp.stride.0 as f32).floor() as usize + 1;
    let o_m = ((i_m - k_m) as f32 / hp.stride.1 as f32).floor() as usize + 1;
    println!("output size should be: {:?}", (o_n, o_m));

    let mut output: Array2<f32> = Array2::zeros((o_n, o_m));
    run_mm_convolution_2d(&hp, &input, &mut output);

    Ok(output)
}

#[test]
#[serial]
fn build_the_right_weights_matrix_i33_k22() {
    #[rustfmt::skip]
    let input = array![
        [1.0, 2.0, 3.0], 
        [4.0, 5.0, 6.0], 
        [7.0, 8.0, 9.0]
    ];

    #[rustfmt::skip]
    let kernel = array![
        [1.0, 2.0],
        [3.0, 4.0]
    ];
    let hp = ConvHyperParam::new(0, (1, 1), kernel);
    println!("convolution hyperparams:\n{:#?}", &hp);
    let weights = kernel_to_weights_matrix(&hp, &input).unwrap();
    let ideal = array![
        [1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 4.0]
    ];
    let eps = 1e-5;
    //let diff = (ideal - weights).sum().abs();
    //println!("diff: {}",diff);
    // assert!(diff < eps);

    let output = mm_convolution_2d(input.clone(), &hp).unwrap();
    let sliding_output = convolution_2d(input, &hp).unwrap();
    let diff = (sliding_output - output).sum().abs();
    dbg!(diff);
    assert!(diff < eps);
}

#[test]
#[serial]
fn build_the_right_weights_matrix_i44_k33() {
    #[rustfmt::skip]
    let input = array![
        [1.0, 2.0, 3.0, 4.0], 
        [5.0, 6.0, 7.0, 8.0], 
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0]
    ];
    /*
    #[rustfmt::skip]
    let kernel = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ];
    */
    #[rustfmt::skip]
    let kernel = array![
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ];
    let hp = ConvHyperParam::new(0, (1, 1), kernel);
    //println!("convolution hyperparams:\n{:#?}",&hp);

    let weights = kernel_to_weights_matrix(&hp, &input).unwrap();

    let eps = 1e-5;
    let mm_output = mm_convolution_2d(input.clone(), &hp).unwrap();
    let sliding_output = convolution_2d(input, &hp).unwrap();

    println!("mm_output:\n{:#.1?}", mm_output);
    println!("sliding_output:\n{:#.1?}", sliding_output);
    let diff = (sliding_output - mm_output).sum().abs();
    dbg!(diff);
    assert!(diff < eps);
    println!("\n\n\n\n.")
}

#[test]
#[serial]
fn build_the_right_weights_matrix_i56_k33() {
    #[rustfmt::skip]
    let input = array![
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5], 
        [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
        [3.0, 3.1, 3.2, 3.3, 3.4, 3.5],
        [4.0, 4.1, 4.2, 4.3, 4.4, 4.5]
    ];

    /*
    #[rustfmt::skip]
    let kernel = array![
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
    ];
    */
    #[rustfmt::skip]
    let kernel = array![
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ];
    let hp = ConvHyperParam::new(0, (1, 1), kernel);
    //println!("convolution hyperparams:\n{:#?}",&hp);

    // let weights = kernel_to_weights_matrix(&hp, &input).unwrap();

    let eps = 1e-5;
    let mm_output = mm_convolution_2d(input.clone(), &hp).unwrap();
    let sliding_output = convolution_2d(input, &hp).unwrap();

    println!("mm_output:\n{:#.1?}", mm_output);
    println!("sliding_output:\n{:#.1?}", sliding_output);
    let diff = (sliding_output - mm_output).sum().abs();
    dbg!(diff);
    assert!(diff < eps);
}
