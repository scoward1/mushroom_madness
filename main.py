# testing feature extraction in pictures with mushrooms
# the same features will be extracted from non-mushroom pictures
# the features will be put into a dataframe that includes whether it is a shroom or not
# all of that will be exported to a .csv file
# that is the creation of the dataset
# this was start on sept 10/2022

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, show
from skimage.filters import prewitt_h,prewitt_v

# import image
image = imread('mushroom.jpg', as_gray=True)

#calculating horizontal edges using prewitt kernel
edges_prewitt_horizontal = prewitt_h(image)
#calculating vertical edges using prewitt kernel
edges_prewitt_vertical = prewitt_v(image)

imshow(edges_prewitt_vertical, cmap='gray')
show()
imshow(edges_prewitt_horizontal, cmap='gray')
show()

# import features
# extract features, and turn them into a dataframe
