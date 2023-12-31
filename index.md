---
title: Latent Space Color Balance
---

### [Author: John (Jack) Messerly](https://www.linkedin.com/in/jack-messerly-567b9b96/)

### [show me the code](https://github.com/messy-bytes/VAE-Color-Balance)

<br>
<br>  


![cie_scatter](diagrams/model2.png)

<br>
<br>

# About This Project

I created this ML algorithm in my spare time, initially to re-color old film photographs my dad took in Mexico, Chile and Argentina, which he captured with the  [Malefic 617 camera](https://www.maleficcameras.com/pagina-prodotto/malefic-m617). The camera isn't digital, so it doesn't have an auto white balance system, and to make matters worse, the film roll got jammed, so the photos were physically damaged before being developed. They also appear to have creases and chemical burns. GIMP (free photoshop) was not enough to fix these photos, some of which are at the bottom of this page. I tried some basic ML color balancers available online, which directly regress new colors from bad ones, but they didn't give me the results I needed. I thought I'd have a crack at designing a better one, since I sit around and work on image processing problems all day anyway. What I came up with worked very well, even on color balance problems that are extremely difficult (with mixed lighting effects, or harsh post-processing filters), so I decided to write it up. Below is a sample result of fixing a damaged photograph of a street band taken in Santiago de Queretaro. 

**More background info:** [I've added a detailed results gallery here](https://messy-bytes.github.io/Advanced-ML-Color-Fixes/2023/04/29/Introduction.html), which includes a lengthy introduction to basic computational photography, and a comparison of "easy" versus "difficult" lighting problems.
<br>
<br>

![natales](diagrams/natales.jpg)

<br>
<br>

# I Want To Try It

[Here's a git repository](https://github.com/messy-bytes/VAE-Color-Balance/tree/main) with directions you can follow for trying this algorithm on your own images. It's pretty straightforward, but you need a linux terminal to use it, and the repository is fairly large (about 100MB) due to the size of the latent color balancing network.

<br>
<br>

# How does it work

### [Too much text, I want an illustrated guide](#step-by-step-illustration)
<br>

There are really three networks involved here. [A pre-trained VAE from StabilityAI](https://huggingface.co/docs/diffusers/api/models/autoencoderkl), a "latent color balancer", which was trained on KL Divergence, and a "color mapper", which is trained on MSE in pixel space. You can see how these networks work together in the diagram at the top of this page. The VAE comes pre-trained, the [KL Divergence network I trained on a custom dataset made image enhancement LUTs](https://messy-bytes.github.io/Advanced-ML-Color-Fixes/2023/05/03/Dataset-Curation.html), and the color mapper is trained online every time the network is used.

**Network #1, The Latent Balancer:** Overall, this algorithm works by fixing images in the "latent space" of a pre-trained Variational Autoencoder (VAE). In simpler terms, a pre-trained network maps the original image to very small dimensional representation (28 x 28 x 8), and all my color fixing happens in this smaller dimensional space. An "intermediate" network (that I trained) modifies these encodings into new encodings that can be decoded into prettier, "fixed" images. The architecture of this intermediate network is not very important (I made a Unet, which is often used for these applications<sup>1</sup>). It needs to be noted that the decoded image from this process will be small, and distorted. A second network is needed to map the colors to the high-resolution original image, which is trained online.

**Network #2, The Color Mapper:** The second network is just a single layer MLP, trained with MSE. It only exists to allow the latent color balancer to recolor high-resolution images. It learns a color mapping (old color -> new color) from the above input/output pair. Since this simpler network operates on single pixels, it can be applied to images of arbitrary resolution. Apply this network to the original high-resolution image to get the final output.

**Putting it all together:** This algorithm uses a VAE (which has both an encoder and decoder network), a latent color balancing network, and another network for representing a color map. A high-resolution "distorted image" image (the one you want to fix) is cropped and compressed into a low-resolution image (of dimension [224 x 224 x 3]) that StabilityAI's VAE can encode. We'll call this artifact the "distorted subimage", because it is cropped and will have terrible resolution. The VAE encodes this low-resolution image as a [28 x 28 x 8] sized vector. We'll call this the "distorted encoding".

1. The first network (which is large, and was trained offline) accepts as input this "distorted encoding", and outputs a "fixed encoding" (also of dimension [28 x 28 x 8]). The loss function for this network is KL divergence, because the encodings are technically the mean and variance of a gaussian distribution, although this isn't important for understanding the algorithm. This "fixed encoding" is decoded, using the VAE's decoder, to get back a prettier, albeit low-resolution, new image of size [224 x 224 x 3]. We'll call this preliminary output the "fixed subimage".
2. A second network is trained (every time the algorithm runs) to map the colors in the "distorted subimage" to the "fixed subimage", using MSE loss. Since this network maps colors (3 vectors) to new colors (3 vectors) it can be applied to images of arbitrary resolution. Apply it to the original, high-resolution image to get a new high-resolution image with "fixed" colors.

**How does the VAE help us?** A standard ML methodology for image enhancement is to train a CNN (convolutional neural network) to directly turn "bad pixels" into "balanced pixels". These networks look at thousands of "bad images", and their "corrected version", and learn to "fix" images through regression. This works pretty well, but on larger images, these networks take a very, very long time to train and are prone to catastrophic forgetting. These methods are also not agnostic to the target image resolution. If you do all your color fixing in the smaller dimensional space of the output of a Variational Autoencoder, you fix most of these problems. The "encoded" output of a VAE is small (the one I used has latent dimension of fixed size [28 x 28 x 8]), which speeds up training tremendously. The latent space is also enforced to be "regular", meaning that images with similar content are closer grouped together in this space, than they are in the original iamge space. I did not find that I needed any regularization strategies (like dropout or weight decay) when training a color balancer in the encoded space, but I did find that I needed them when training networks on the images directly.

**Prove it.** I dislike it when people add complexity or "novel techniques" to algorithms that don't need them. So, I will do my best to convince you that I have not done that by introducing a VAE into an algorithm that does not need it. What if I just trained a Unet to balance the images directly, without this convoluted 3 network system? For comparison's sake, trained a "direct Unet" on the same dataset, which simply rebalances the images in their original format. Here were my observations:

1. **It did work.** When you skip the VAE stuff, and just run a Unet to recolor images, you get similar results on certain photos. 
2. **It was slower.** By using compressed latent vectors, you can significantly speed up your training process due to the smaller size of the data. This results in the ability to train a high-performing network that can generalize effectively in under 20 minutes, which is not achievable when using sized (224x224) images. In my implementation without a VAE, each epoch took 220 seconds, while with VAE it only took 30 seconds. However, it’s not a direct comparison since I had to reduce the number of parameters in the “regular CNN” implementation significantly to make it run without being killed by the OS. This isn't exactly a point in the favor of the "direct" approach. I used the NVIDIA GeForce RTX 3070 for training.
3. **It required training regularization.** In order to not make this direct network not overfit, I needed to add both weight decay and dropout. Training not only took longer, but tuning it was more of a headache. The VAE implementation did not need regularization, almost certainly because the latent space it operates in was forced to be regular.
4. **It couldn't be recursively applied.** Some photos, like the image of the girl below, require multiple applications to "dredge out". The VAE balancer can be applied to the same image multiple times, for iterative benefit. The direct Unet did not exhibit this property. Interestingly, it flopped between similar images. I believe this is also because the latent space is highly regular, so iterative improvements could be found easily, whereas this is not the case in the original image space. Observe, on the top is some recursive applications using a denoising system with a VAE, and on the bottom is the same thing, but using a system that denoises images with a Unet directly.

<br>

<p align="center">
  <img src="diagrams/recursive_comparison.png" alt="Recursive Comparison">
</p>

<br>


**I'd like to learn more about VAEs**:  I did not provide a background section on Variational Autoencoders as explaining it from scratch would be difficult.One excellent article on this topic was written by Joseph Rocca and can be found on TowardsDataScience. It is worth mentioning that he was a full-time employee at the site when he wrote it. [The article is linked here](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)

<br>

<sup>1</sup> *In Stable Diffusion art generation, this "intermediate" network doesn't just modify the colors of the image, but also the composition. You can view the algorithm on this page as an AI art algorithm that just modifies the colors.*

<br>
<br>

# Step-by-Step Illustration

## Step 1: compress / crop a full sized image a 224x224 sized tile 

The VAE for StableDiffusion actually requires images to be quite small ([224 x 224 x 3]), but this compression does not affect the final result due to our ability to remap the colors at full resolution with the MSE network (step 5-6).

<p align="center">
  <img src="diagrams/crop_compress.jpg" alt="Crop and Compress">
</p>

<br>

## Step 2: encode this image as a 28x28x8 sized distribution vector

The low-resolution image is encoded with the pre-trained VAE's encoder. The "top four" [28 x 28] images are the mean of the latent representation, while the bottom four are the variance. In Stable Diffusion, this latent distribution is sampled to get randomized versions of the original image, but we aren't going to use the encoding like that. We are just going to plug it into another network as a [28 x 28 x 8] sized input vector.

<p align="center">
  <img src="diagrams/initial_encoding.jpg" alt="Initial Encoding">
</p>

<br>

## Step 3: balance the encoding

The primary balancing network, trained with KL divergence, adjusts the encoding to be one that, in theory, should decode to a fixed image. The differences between the two encodings are interesting to ponder (why does the fixed encoding have higher variance values, and what do the channels mean in RGB space?). Ultimately, this encoding is uninterpretable, and we just have to treat it as data.

<p align="center">
  <img src="diagrams/latent_color_balancer.jpg" alt="Latent Color Balancer">
</p>

<br>

## Step 4: decode to recover a distorted/fixed pair

We decode both the distorted vector, and the fixed vector. Why not just the fixed and call it a day? Notice how the face of the girl in these images looks wonky and weird. That is because Stable Diffusion's VAE does not decode details perfectly. We notice faces are often "off" in AI art because as humans, we are sensitive to facial distortions. Ultimately, we only care about the "color map" that can be learned between these two images. We want to take the original distorted image, and line it up with the "fixed" one that we made with the balancer network, pixel for pixel, so we can figure out how to remap colors. We can't quite line up the cropped/compressed image from Step 1 with the "fixed" decoded output, because the image from step 1 does not have an "ugly face" that can be matched pixel-for-pixel with the fixed one. How do we get an "ugly face" version of the original image? Run the original encoding through the decoder as well.

<p align="center">
  <img src="diagrams/decoding_stage.jpg" alt="Decoding Stage">
</p>

<br>

## Step 5: train a new MSE classifier to map bad colors to good colors

This network is trained online, meaning that every time the algorithm runs, it needs to train a new color mapping network. However, a single layer MLP can be trained in a few seconds.

<p align="center">
  <img src="diagrams/color_map_learning.jpg" alt="Color Map Learning">
</p>


<br>

## Step 6: apply this map to the original, full resolution image

<p align="center">
  <img src="diagrams/one_iter.png" alt="One Iteration">
</p>


<br>

## Step N: repeat as necessary

In the latent space, representations of the same subject matter with different colors are closely grouped, exhibiting regularity. This proximity implies that the algorithm maintains consistency and stability when repeatedly applied to the same image.

![recursive](diagrams/recursive.png)

<p align="center">
  <img src="diagrams/five_iter.png" alt="Five Iterations">
</p>


<br>
<br>


# How Did You Train It

It was actually difficult to make a training data set that could address the issues in my dad's damaged photos. I made some Look Up Table (LUTs) to mimic a lot of complex distortions and gamma curves, and trained the latent balancer to remove them in the encoded space (by using the LUTs to distort the images, and then treat the original image as an output). [A full guide on generating these datasets is here](https://messy-bytes.github.io/Advanced-ML-Color-Fixes/2023/05/03/Dataset-Curation.html). I also took some "image enhancement" LUTs (from Kodak, Nikon and Colorist Factory), and used them to make training examples where the network could take normal images, and enhance them. This helped address the network's tendancy to output "dead-body" skin tones when the color neutralization was too strong. I think these extra examples overall made the output images look more visually appealing. Recall that of the 3 networks used (the VAE, the latent balancer, and the color mapper), only the latent balancer required offline training to recolor images.

<br>

![3DLUT](diagrams/3dlut.png)

<br>
<br>
<br>
<br>

# Torres del Paine Gallery

<br>

![patagonia1](diagrams/patagonia1.png)

<br>
<br>

![patagonia2](diagrams/patagonia2.png)

<br>
<br>

![patagonia3](diagrams/patagonia3.png)
