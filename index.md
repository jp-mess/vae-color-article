---
title: Advanced Color Fixes 
---

This three part article covers:

1. How to design an ML dataset using 1D and 3D LUTs, in order to fix more complex color descriptions than "white balance". We 
will show how to remove specular highlights, instagram filters, and artificial lighting

2. How to train a convolutional neural network to fix these color distortions in the "latent space" of one of the more popular
Variational Autoencoder's available on HuggingFace

3. We will review the state of the art in image enhancement by showing how to learn a "LUT" using pytorch's differentiable 
grid sample function  

