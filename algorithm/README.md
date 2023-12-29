# Latent Space Color Correction

This project implements an image color correction algorithm using a neural network approach. It learns a color mapping from a distorted image to a fixed image and iteratively applies this correction to enhance image quality.

## Installation

To use this project, clone the repository and install the required packages:

```bash
git clone https://your-repository-url.git
cd your-repository-directory
pip install -r requirements.txt
```

## Usage

### Using the Script from the Command Line

The `test.py` script can be used directly from the command line. It requires two arguments: `--image`, the path to the image file, and `--iterations`, the number of iterations for the recursive algorithm.

```bash
python test.py --image path/to/your/image.png --iterations 1
```

Replace `path/to/your/image.png` with the path to the image you want to process and specify the number of iterations as needed.

### Image Loader

To download and save an image:

```python
# Replace with your actual URL
image_url = 'http://example.com/image.jpg'  
save_image_from_url(image_url, 'images')
```

This script downloads the image from the given URL and saves it in the `images` directory with the filename derived from the variable name.

### Color Correction Algorithm

To apply the color correction algorithm in a Python script:

```python
from PIL import Image
# Other necessary imports

# Load your image
original_image = Image.open('path/to/your/image.png')

# Define other necessary components like transformations, VAE model, etc.

# Apply the recursive color correction
iterations = 1  # Default number of iterations
adjusted_image = recursive_color_correction(original_image, iterations, transformations, vae_model, balancer)

# Display or save the adjusted image
adjusted_image.show()  # or adjusted_image.save('path/to/save/image.png')
```

Replace `'path/to/your/image.png'` with the path to the image you want to process.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

[MIT License](LICENSE)

