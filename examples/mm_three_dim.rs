use convolution::prelude::*;
use ndarray::prelude::*;
use show_image::{make_window_full, Event, WindowOptions};
use image;

fn main() {
    //let input = open_rgb_image_and_convert_to_ndarray3("examples/ferris_ml.png").unwrap();
    let mut input = open_rgb_image_and_convert_to_ndarray3("examples/grand_canyon_trees.png").unwrap();

    let kernel_h = array![[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]];
    let hp_hori = ConvHyperParam::default(kernel_h).stride((1, 1)).padding(0).build();
    let kernel_v = array![[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]];
    let hp_vert = ConvHyperParam::default(kernel_v).stride((1, 1)).build();

    // let output = mm_convolution_3d(input, &hp_hori).unwrap();
    // let output = mm_convolution_3d(output, &hp_vert).unwrap();

    #[rustfmt::skip]
    let kernel_edge = array![
        [-1.0, -1.0, -1.0],
        [-1.0, 8.0, -1.0],
        [-1.0, -1.0, -1.0]
    ];
    let hp_edge = ConvHyperParam::default(kernel_edge).stride((1,1)).padding(0).build();

    let output = single_mult_mm_convolution_3d(input.clone(), &hp_edge).unwrap();

    let output_img = rgb_ndarray3_to_rgb_image(output);
    
    let window_options = WindowOptions {
        name: "image".to_string(),
        size: [800, 800],
        resizable: true,
        preserve_aspect_ratio: true,
    };
    println!("\nPlease hit [ ESC ] to quit window:");
    let window = make_window_full(window_options).unwrap();
    window.set_image(image::open("examples/grand_canyon_trees.png").unwrap(), "orig_image").unwrap();

    for event in window.events() {
        if let Event::KeyboardEvent(event) = event {
            if event.key == show_image::KeyCode::Escape {
                break;
            }
        }
    }

    window.set_image(output_img, "test_result").unwrap();

    for event in window.events() {
        if let Event::KeyboardEvent(event) = event {
            if event.key == show_image::KeyCode::Escape {
                break;
            }
        }
    }
    
    std::thread::sleep(std::time::Duration::from_secs(10));
    show_image::stop().unwrap();


}
