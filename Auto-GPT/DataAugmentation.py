import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from skimage import io 
from skimage import transform
from skimage.transform import rotate, AffineTransform
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage import util

image = io.imread('https://github.com/VALENCEML/eBOOK/raw/main/EN/07/flower.jpg')
#plt.imshow(image)
#plt.axis('off')

augmentation2=np.flipud(image)
fig=plt.figure(tight_layout='auto', figsize=(8,4))
fig.add_subplot(121)
plt.imshow(image)
plt.axis('off')
fig.add_subplot(122)
plt.imshow(augmentation2)
plt.axis('off')
