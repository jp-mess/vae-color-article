---
title: Advanced Color Fixes
author: John Patrick Messerly 
---

In this multi-part post, we're going to thoroughly review the state of color balance and enhancement. 

1. We're going to go beyond based color perturbations, and learn how to design complex color distortion datasets with our own Look Up Tables (LUTs). This will help us balance more complex scenes, like those with **multi-source colored lighting** and **harsh artificial lighting**

2. We're going to use HuggingFace's incredible diffusers library to take our ML algorithms into the latent space. We'll learn how to train a fully convolutional network in this condensed space to save on memory and time, and show a how a "color sampling" trick will let us recolor the original high resolution image with what our model learned in "compressed space"

3. We're going to deep dive into the recent paper "**Learning Image-adaptive 3D Lookup Tables for High Performance Photo Enhancement in Real-time**", and learn how to use pytorch's "grid sample" function, as well as GANs, to learn LUTs differentiably

### This project assumes that...

*   You have a strong understanding of basic color theory, including HSV coordinates, as well as basic white balance concepts (the gray world algorithm)
*   You're comfortable with pytorch and fully convolutional neural networks, and you've at least heard of GANs 
*   You're familiar with variational autoencoders, the latent space, and programming with the HuggingFace diffusers library. If not, then you've heard of them, and you'd like to learn more

Although it does not assume you know GIMP, I suggest you download it (for free) and try it out! It will get you through most of your image data needs when you don't feel like writing an entire python script to do something.
