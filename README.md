## Work-in-progress library for composable convolution primitives of 2/3D ndarray structs**
*No, really, this library is pretty unstable*

The following is an example of an RGB image converted to `Array3<f32>` and 
run through an edge-detection kernel filter through a matrix-multiplication convolution.

Several different methods are available, including only-2D convolution, repeated 2D convolution,
and single matrix-multiplication operation for 3D images. Arbitrary kernels can be used for any 
of these operations. 

```rust
use convolution::prelude::*;
use ndarray::prelude::*;

fn main() {
    let mut input = open_rgb_image_and_convert_to_ndarray3("examples/grand_canyon_trees.png").unwrap();

    // https://en.wikipedia.org/wiki/Kernel_(image_processing)#Details Look under "Edge Detection"
    #[rustfmt::skip]
    let kernel_edge = array![
        [-1.0, -1.0, -1.0],
        [-1.0, 8.0, -1.0],
        [-1.0, -1.0, -1.0]
    ];
    let hp_edge = ConvHyperParam::default(kernel_edge).stride((1,1)).padding(0).build();
    let output = mm_convolution_3d(input, &hp_edge).unwrap(); // or single_mult_mm_convolution_3d()
    
    display_image(output); // Do this however you want, the show_image library is pretty nice
}
```

**before:**
<p align="left"><img src="/examples/grand_canyon_trees.png" width="500" height="400" /></p>

**after:**
<p align="left"><img src="/examples/filtered_canyon.png" width="500" height="400" /></p>

### License

This work is licensed under the MIT License, a permissive open source license. 

However, if you are intending to use this library for machine learning, please educate yourself 
on the potential downstream effects of your work, and review some of the guidelines and 
discussions surrounding it. Some resources that have been collected by the Institute for Ethical 
Machine Learning [here](https://github.com/EthicalML/awesome-artificial-intelligence-guidelines).
