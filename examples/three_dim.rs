use minifb::{Key, ScaleMode, Window, WindowOptions};
use ndarray::prelude::*;

use convolution::prelude::*;

fn main() {
    /*let input = array![
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
    println!("input:\n{:#?}",input);
    //let padding = 1;
    //let padded = pad_3d(input, padding);
    //println!("padded:\n{:#?}",padded);
    */
    let input = open_rgb_image_and_convert_to_ndarray3("examples/ferris_ml.png").unwrap();
    //let input = open_rgb_image_and_convert_to_ndarray3("examples/grand_canyon_trees.png").unwrap();
    display_img(&input);

    let kernel_h = array![[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]];
    let hp_hori = ConvHyperParam::default(kernel_h).stride((2, 2)).build();
    let kernel_v = array![[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]];
    let hp_vert = ConvHyperParam::default(kernel_v).stride((1, 1)).build();

    let output = convolution_3d(input, &hp_hori).unwrap();
    let output = convolution_3d(output, &hp_vert).unwrap();
    display_img(&output);
}

fn display_img(data: &Array3<f32>) {
    // let dims = data.dim();
    let (n, m) = (data.dim().1, data.dim().2);
    println!("image is of size: ({},{})", n, m);

    let mut buffer: Vec<u32> = Vec::with_capacity(m * n);
    let (w, h) = (m, n);
    for y in 0..h {
        for x in 0..w {
            let temp: [u8; 4] = [data[[2, y, x]] as u8, data[[1, y, x]] as u8, data[[0, y, x]] as u8, 255u8];
            // println!("temp: {:?}", temp);
            buffer.push(u32::from_le_bytes(temp));
        }
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
