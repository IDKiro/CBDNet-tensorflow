# CBDNet-tensorflow 

A unofficial implementation of CBDNet by Tensorflow.

[CBDNet in MATLAB](https://github.com/GuoShi28/CBDNet)

## Network Structure

![Image of Network](figs/CBDNet_v13.png)

## Realistic Noise Model
Given a clean image `x`, the realistic noise model can be represented as:

![](http://latex.codecogs.com/gif.latex?\\textbf{y}=f(\\textbf{DM}(\\textbf{L}+n(\\textbf{L}))))

![](http://latex.codecogs.com/gif.latex?n(\\textbf{L})=n_s(\\textbf{L})+n_c)

Where `y` is the noisy image, `f(.)` is the CRF function and the irradiance ![](http://latex.codecogs.com/gif.latex?\\textbf{L}=\\textbf{M}f^{-1}(\\textbf{x})) , `M(.)` represents the function that convert sRGB image to Bayer image and `DM(.)` represents the demosaicing function.

If considering denosing on compressed images, 

![](http://latex.codecogs.com/gif.latex?\\textbf{y}=JPEG(f(\\textbf{DM}(\\textbf{L}+n(\\textbf{L})))))

# Result

![](figs/results.png)

# Quick Start

Use followed command:

```
python train.py
```