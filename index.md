---
title: Colored Lighting Removal with VAEs
---

### [Author: John (Jack) Messerly](https://www.linkedin.com/in/jack-messerly-567b9b96/)

<br>
<br>  


![cie_scatter](diagrams/model2.png)

<br>
<br>

I created this ML algorithm during my spare time, initially to re-color old film photographs my dad took in Patagonia, which he captured with the  [Malefic 617 camera](https://www.maleficcameras.com/pagina-prodotto/malefic-m617). He forgot to get a color calibration measurement in the field, and the filmroll got jammed, so they became physically damaged in the development process. GIMP / photoshop was not enough to fix these photos, some of which are at the bottom of this page. I tried some basic ML color balancers available online, but they didn't give me the results I needed. I thought I'd have a crack at designing a better a better one. Below is a sample result of fixing a damaged photograph of a street band taken in Puerto Natales, Chile.

<br>

![natales](diagrams/natales.jpg)


# Abstract

"Color balance" aims to enhance images with suboptimal color quality, typically resulting from colored ambient lighting, like the fluorescent lighting in a parking garage. The desired outcome in traditional color balancing is to replicate the colors that would be present under perfectly white light, a process known as "white balancing". However, the ideal color adjustment often depends on personal preference and psychological factors, leading modern techniques to extend beyond mere white balance to overall color enhancement. For instance, we generally prefer warmer tones in human faces, regardless of the actual lighting conditions.

The distinction between "color balance" and "image enhancement" can be vague. Data-driven machine learning methods address both simultaneously by adjusting the pixels of a poorly colored image to resemble those in more aesthetically pleasing references. A common machine learning approach involves using a fully convolutional network (FCN) to modify colors directly. This method, while effective, has limitations, such as potential loss of detail, inconsistent recoloring across similar areas, and slow training for large images.

However, powerful pre-trained varational autoencoders let you bypass these issues. By training a color balancer in Stable Diffusion's latent space, you can extract a color map (old color -> new color) to apply to your original images, which is far more efficient and effective than trying to train an AI to color each pixel in an image individually. Some advantages of using variational autoencoder representation's to achieve color balance: 

1. Protect edges and textures through simple color remapping.
2. Ensure uniform recoloring across the image, avoiding irregularities in similar areas.
3. Enhance efficiency and training speed due to the small, fixed-size latent representations, independent of the original image's resolution.

<br>
<br>


# HuggingFace's toolbox

Autoencoders (and variational autoencoder), are just data compression algorithms. They've become a hot topic recently because its been shown that a lot of AI art algorithms (including stable diffusion) can operate in the smaller compressed space. The VAE's trained by Stability AI that are available on HuggingFace compress images into small (28 x 28 x 28 x 8) vectors that can be easily stored and processed on servers and standard GPUs. The compressed space is more organized than the original, meaning that when these vectors are manipulated with other algorithms, the resulting images will be coherent. For example, a minor distortion in the compressed space will result in a hairstyle change in the original image space.

A big disadvantage of this scheme is that the "encoded representations" can't be decoded perfectly back into the originals. So while you can fuse together several picture of horses in a the "compressed" space, when you go and decode it, it will have weird ears and a distorted mouth. The "phase" of the image, a term I abuse here to describe the shape, structure and details, will be distorted. This is why AI art often contains "messed up faces". Below is an encoded and decoded image of a barista. Clearly, the decoded image is unusable. Also, the the fact that our VAE can only encode/decode small images (224 x 224) seems to negate its use in actual image enhancement pipelines.

<br>

![barista](diagrams/barista.png)

<br>

What if we only want to change the colors and lighting of our image instead of altering its "phase"? In the graphics and film industries, "Look Up Tables" are commonly used to adjust colors in images by mapping old colors to new ones. These tables are small in size (33 x 33 x 33 x 3) and can be applied to images of any resolution. You can follow this logic in reverse: if you want to learn a color and exposure mapping that "fixes" a high-resolution image, you shouldn't need to use the original full-sized image to obtain this mapping.                                    
 
<br>
<br>
     
# Project Structure

The goal of this (informal) post is to show how Stability AI's pre-trained VAEs can be used to re-color and re-light images more efficiently than current approaches. The post is divided into three parts for clarity.

<br>

## [Part 1: Results Gallery](https://messy-bytes.github.io/Advanced-ML-Color-Fixes/2023/04/29/Introduction.html)

"Colored lighting removal" is very similar to the concept of "white balance", but it is not the same thing. I've dedicated the first section to explaining the problem's scope, and demonstrating results in a small gallery. A lot of the images I focused on contain challenging color and lighting distortions that you generally won't see on other color balance articles. Some of this page is dedicated to explaining the general problem of color balance, but the writing does assume you are somewhat familiar with computational photography already.

<br>

## [Part 2: Dataset Curation](https://messy-bytes.github.io/Advanced-ML-Color-Fixes/2023/05/03/Dataset-Curation.html)

<br>

![3DLUT](diagrams/3dlut.png)

<br>

Dataset generation is its own large problem for tasks like this. This section covers dataset curation with 1D and 3D LUTs, and describes the limitations of generating a distortion dataset "randomly". It also includes a tutorial on how to create 3D LUTs using GIMP that can be applied to any image you like. There are probably radiometric and colorimetric distortions you'd like to be able to neutralize that aren't covered here, and learning how to build custom 3D LUTs is a good step forward for solving those problems.

<br>


## [Part 3: Denoising Algorithm Overview](https://messy-bytes.github.io/Advanced-ML-Color-Fixes/2023/05/05/The-Denoising-Algorithm.html)

This last section covers the actual machine learning involved, and the other parts of the denoising algorithm. Most of it is intuitive, but there seem to be some advantages of working in a latent space when it comes to "recursively applying" image filters. I've tried my best to explain why my technique works as well as it does when you apply it iteratively. This section assumes familiarity with Variational Autoencoders.

<br>

![recursive](diagrams/recursive.png)

<br>
<br>

# Sample Results

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
