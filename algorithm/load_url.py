import os
import requests
from PIL import Image
from io import BytesIO
import inspect

def save_image_from_url(url, folder='images'):
    """
    Saves an image from a URL to a specified folder.
    
    Args:
    - url (str): The URL of the image.
    - folder (str): The folder where the image will be saved.
    """

    # Create the folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Load the image from the URL
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))

    # Get the variable name used for the URL
    for name, val in inspect.currentframe().f_back.f_locals.items():
        if val is url:
            file_name = name
            break

    # Save the image
    image_path = os.path.join(folder, file_name + '.png')
    image.save(image_path)
    print(f"Image saved as {image_path}")

if __name__ == "__main__":
  vsco_girl = "https://images.unsplash.com/photo-1603632710913-ba23139de531?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=735&q=80"
  sabers = "https://images.unsplash.com/photo-1603757981609-9aa146d91fb0?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1332&q=80"
  garage = "https://images.unsplash.com/photo-1665973605924-3db7ed60ffa5?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=687&q=80"
  skating = "https://images.pexels.com/photos/5788775/pexels-photo-5788775.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
  save_image_from_url(vsco_girl)
