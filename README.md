**Work-in-progress library for doing arbitrary convolution of 2/3D ndarray structs**

The following is an example of an RGB image converted to `Array3<f32>` and 
run through a horizontal and vertical line filter using a sliding-window kernel.
Dimensionality is reduced depending on the defined padding and stride values.   

**before:**
<p align="left"><img src="/examples/grand_canyon_trees.png" width="400" height="300" /></p>

**after:**
<p align="left"><img src="/examples/filtered_canyon.png" width="400" height="300" /></p>