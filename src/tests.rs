#![allow(unused_imports)]

use crate::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::iter::FromIterator;

#[test]
#[serial]
fn sliding_2d_k22s11p0() {
    #[rustfmt::skip]
    let input = array![
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 2.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 3.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 4.0, 0.0]
    ];

    #[rustfmt::skip]
    let kernel = array![
        [1.0, 2.0],
        [3.0, 4.0]
    ];

    let hp = ConvHyperParam::default(kernel).stride((1, 1)).padding(0).build();
    let sliding_output = convolution_2d(input,&hp).unwrap();
    println!("sliding output:\n{:?}",sliding_output);

    #[rustfmt::skip]
    let ideal = array![
        [14.0, 13.0, 10.0, 4.0],
        [12.0, 19.0, 16.0, 4.0],
        [10.0, 14.0, 24.0, 13.0],
    ];
    let eps = 1e-5;
    let diff = (ideal - sliding_output).sum().abs();
    println!("diff: {}",&diff);
    assert!(diff < eps);
}

#[test]
#[serial]
fn sliding_2d_k33s11p0() {
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
    let sliding_output = convolution_2d(input, &hp).unwrap();
    println!("3x3 kernel sliding output:\n{:?}",sliding_output);

    
    #[rustfmt::skip]
    let ideal = array![
        [-2.0, -2.0, -2.0],
        [1.0, -2.0, -3.0]
    ];
    let eps = 1e-5;
    let diff = (ideal - sliding_output).sum().abs();
    println!("diff: {}",&diff);
    assert!(diff < eps);
    
}


#[test]
#[serial]
fn small_mm_2d_test() {
    /*
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
    // let kernel = array![[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]];
    */

    let input = Array::random((3,4), Uniform::new(0., 1.));
    let kernel = Array::random((2, 2), Uniform::new(0., 1.));

    let hp = ConvHyperParam::default(kernel).stride((1, 1)).padding(0).build();

    let sliding_output = convolution_2d(input.clone(), &hp).unwrap();
    println!("output from sliding convolution:\n{:#?}", sliding_output);

    let mm_output = mm_convolution_2d(input, &hp).unwrap();
    println!("mm_output:\n{:#?}", mm_output);

    let eps = 1e-5;
    let diff = (sliding_output - mm_output).sum();
    println!("diff: {}",&diff);
    assert!(diff.abs() < eps);


}
