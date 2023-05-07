---
title: Variational Autoencoder Color Balance
---

<br>
<br>  


![cie_scatter](diagrams/model2.png)

<br>
<br>

### author: [John (Jack) Messerly]([https://link-url-here.org](https://www.linkedin.com/in/jack-messerly-567b9b96/))

# Manipulating Colors and Lighting with VAEs

By using Variational Autoencoders, Stability AI has made working with generative AI much easier. These autoencoders compress images into small (28 x 28 x 28 x 8) vectors that can be easily stored and processed on servers and standard GPUs. The compressed space is more organized than the original, meaning that when these vectors are manipulated with other algorithms, the resulting images will be coherent. For example, a minor distortion in the compressed space will result in a hairstyle change in the original image space.

A big disadvantage of this scheme is that the encoded representations can't be decoded perfectly back into the originals. The "phase" of the image, a term I abuse here to describe the shape, structure and details, will be distorted. This is why AI art often contains "messed up faces". Below is an encoded and decoded image of a barista. Clearly, the decoded image is unusable. Also, the the fact that our VAE can only encode/decode small images (224 x 224) seems to negate its use in actual image enhancement pipelines.

<br>

![barista](diagrams/barista.png)

<br>

What if we only want to change the colors and lighting of our image instead of altering its "phase"? In the graphics and film industries, "Look Up Tables" are commonly used to adjust colors in images by mapping old colors to new ones. These tables are small in size (33 x 33 x 33 x 3) and can be applied to images of any resolution. You can follow this logic in reverse: if you want to learn a color and exposure mapping that "fixes" a high-resolution image, you shouldn't need to use the original full-sized image to obtain this mapping.                                    
 
<br>
<br>
     
# Project Structure

In this blog post, I discuss the challenge of removing intense colored lighting from images, which cannot be done by simple "white balance" algorithms, and takes expert skill to fix in manual retouching tools. While the abstract above advertises the use of Variational Autoencoders, the real difficulty was creating an image dataset that accurately portrayed these distortions. The post is divided into three parts for clarity.

<br>

## [Part 1: Results Gallery](https://messy-bytes.github.io/Advanced-ML-Color-Fixes/2023/04/29/P1.html)

I've dedicated the first section to explaining the problem's scope, and demonstrating results in a small gallery. A lot of the images I focused on contain challenging color and lighting distortions that are hard to eliminate even with Photoshop tools. Some of this page is dedicated to explaining the general problem of color balance, but the writing does assume you are somewhat familiar with computational photography already.

<br>

## [Part 2: Dataset Curation](https://messy-bytes.github.io/Advanced-ML-Color-Fixes/2023/05/03/P2.html)

<br>

![3DLUT](diagrams/3dlut.png)

<br>

This section covers dataset curation with 1D and 3D LUTs, and describes the limitations of generating a distortion dataset "randomly". It also includes a tutorial on how to create 3D LUTs using GIMP that can be applied to any image you like. There are probably radiometric and colorimetric distortions you'd like to be able to neutralize that aren't covered here, and learning how to build custom 3D LUTs is a good step forward for solving those problems.

<br>


## [Part 3: Denoising Algorithm Overview](https://messy-bytes.github.io/Advanced-ML-Color-Fixes/2023/05/05/P3.html)

This last section covers the actual machine learning involved, and the other parts of the denoising algorithm. Most of it is intuitive, but there seem to be some advantages of working in a latent space when it comes to "recursively applying" image filters. I've tried my best to explain why my technique works as well as it does when you apply it iteratively, but at the end of the day it's anyone's guess.

<br>

![recursive](diagrams/recursive.png)

<br>
<br>

# Sample Results

<br>
<br>

![natales](diagrams/natales.jpg)

<br>
<br>

![coffee](diagrams/coffee.png)

<br>
<br>

![garage](diagrams/garage.png)

<br>
<br>

![skates](diagrams/skates.png)

<br>
<br>

![vsco](diagrams/vsco.png)

<br>
<br>

![vsco](diagrams/greenlight.png)

<br>
<br>

![patagonia1](diagrams/patagonia1.png)

<br>
<br>

![patagonia2](diagrams/patagonia2.png)

<br>
<br>

![patagonia3](diagrams/patagonia3.png)
