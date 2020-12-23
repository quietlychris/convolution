use convolution::prelude::*;
use ndarray::prelude::*;
use show_image::{make_window_full, Event, WindowOptions};

fn main() {
    // let input = open_grayimage_and_convert_to_ndarray2("examples/ferris_ml.png").unwrap();
    let input = open_grayimage_and_convert_to_ndarray2("examples/grand_canyon_trees.png").unwrap();

    let kernel_h = array![[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]];
    let hp_hori = ConvHyperParam::default(kernel_h).stride((1, 1)).build();
    let kernel_v = array![[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]];
    let hp_vert = ConvHyperParam::default(kernel_v).stride((1, 1)).padding(3).build();

    let output = mm_convolution_2d(input, &hp_hori).unwrap();
    let output = mm_convolution_2d(output, &hp_vert).unwrap();
    display_img(output.to_owned());
}

pub fn display_img(input: Array2<f32>) {
    let output_img = bw_ndarray2_to_image(input);
    let window_options = WindowOptions {
        name: "image".to_string(),
        size: [300, 300],
        resizable: true,
        preserve_aspect_ratio: true,
    };
    println!("\nPlease hit [ ESC ] to quit window:");
    let window = make_window_full(window_options).unwrap();
    window.set_image(output_img, "test_result").unwrap();

    for event in window.events() {
        if let Event::KeyboardEvent(event) = event {
            if event.key == show_image::KeyCode::Escape {
                break;
            }
        }
    }

    show_image::stop().unwrap();
}
