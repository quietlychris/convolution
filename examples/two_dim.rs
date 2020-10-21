use convolution::prelude::*;
use minifb::{Key, ScaleMode, Window, WindowOptions};
use ndarray::prelude::*;

fn main() {

    let kernel_h = array![[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]];
    let hp_hori = ConvHyperParam::default(kernel_h).stride((1, 1)).build();
    let kernel_v = array![[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]];
    let hp_vert = ConvHyperParam::default(kernel_v).stride((1, 1)).padding(3).build();
    // let input = open_grayimage_and_convert_to_ndarray2("examples/ferris_ml.png").unwrap();
    let input = open_grayimage_and_convert_to_ndarray2("examples/grand_canyon_trees.png").unwrap();
    display_img(input.clone());

    let output = convolution_2d(input, &hp_hori).unwrap();
    let output = convolution_2d(output, &hp_vert).unwrap();
    display_img(output.to_owned());
}

pub fn display_img(input: Array2<f32>) {
    let (n, m) = (input.nrows(), input.ncols());
    println!("input shape: ({}, {})", n, m);
    let input = input.into_shape(n * m).expect("Error flattening input");
    let img_vec: Vec<u8> = input.to_vec().iter().map(|x| *x as u8).collect();
    // println!("img_vec: {:?}",img_vec);
    let mut buffer: Vec<u32> = Vec::with_capacity(input.len());
    for px in 0..input.len() {
        let temp: [u8; 4] = [img_vec[px], img_vec[px], img_vec[px], 255u8];
        // println!("temp: {:?}",temp);
        buffer.push(u32::from_le_bytes(temp));
    }

    let (window_width, window_height) = (n * 2, m * 2);
    let mut window = Window::new(
        "Test - ESC to exit",
        window_width,
        window_height,
        WindowOptions {
            resize: true,
            scale_mode: ScaleMode::Center,
            ..WindowOptions::default()
        },
    )
    .unwrap_or_else(|e| {
        panic!("{}", e);
    });

    // Limit to max ~60 fps update rate
    window.limit_update_rate(Some(std::time::Duration::from_micros(16600)));

    while window.is_open() && !window.is_key_down(Key::Escape) && !window.is_key_down(Key::Q) {
        // We unwrap here as we want this code to exit if it fails. Real applications may want to handle this in a different way
        window.update_with_buffer(&buffer, m, n).unwrap();
    }
}
