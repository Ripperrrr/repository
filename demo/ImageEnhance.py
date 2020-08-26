from PIL import Image
from PIL import ImageEnhance
import  matplotlib.pyplot as plt
import numpy as np

# provide some image enhancement function
# brightness enhancement
def brightness(image, brightness = 1.5):
    output = ImageEnhance.Brightness(image)
    output = output.enhance(brightness)
    return output

# Sharpness enhancement
def sharpness(image, sharpness = 3):
    output = ImageEnhance.Sharpness(image)
    output = output.enhance(sharpness)
    return output

# Contrast enhancement
def contrast(image, contrast = 1.5):
    output = ImageEnhance.Contrast(image)
    output = output.enhance(contrast)
    return output

