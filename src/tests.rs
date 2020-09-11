#![allow(unused_imports)]
use crate::prelude::*;
use ndarray::prelude::*;
use std::iter::FromIterator;

#[test]
fn test_sliding_2d() {
    let input = array![
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 2.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 3.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 4.0, 0.0]
    ];

    let kernel = array![[-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]];
}

#[test]
fn small_mm_test() {
    let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

    let kernel = array![[1.0, 2.0], [3.0, 4.0]];
    // let kernel = array![[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]];
    let hp = ConvHyperParam::default(kernel).stride((1, 1)).build();

    let sliding_output = convolution_2d(input.clone(), &hp).unwrap();
    println!("output from sliding convolution:\n{:#?}", sliding_output);

    let flat_input = Array::from_iter(input.iter().cloned());
    let weights = kernel_to_weights_matrix(&hp, &input).expect("Error creating the weights matrix");
    println!("weights:\n{:#?}", weights);

    let mm_output = weights.dot(&flat_input).into_shape((2, 2)).unwrap();
    println!("mm_output:\n{:#?}", mm_output);
    assert!(sliding_output == mm_output);
}
