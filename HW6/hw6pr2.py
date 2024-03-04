"""
Starter file for hw6pr2 of Big Data Summer 2017

Before attemping the helper functions, please familiarize with pandas and numpy
libraries. Tutorials can be found online:
http://pandas.pydata.org/pandas-docs/stable/tutorials.html
https://docs.scipy.org/doc/numpy-dev/user/quickstart.html

Please COMMENT OUT any steps in main driver before you finish the corresponding
functions for that step. Otherwise, you won't be able to run the program
because of errors.

Note:
1. When filling out the functions below, note that
	1) Let k be the rank for approximation

2. Please read the instructions and hints carefully, and use the name of the
variables we provided, otherwise, the function may not work.

3. Remember to comment out the TODO comment after you finish each part.
"""

"""
I was not able to get this code up and running :( I was haing problems loading in the 
image, and even after referring to the answer key, the loading was not working.
I think the answer key uses an outdated version of scipy imread, 
because when looking up errors, it says this function has been demoted...sorry!"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.misc
from scipy import ndimage
import urllib.request
import imageio
from matplotlib.pyplot import imread
import cv2 as cv
#from skimage import io

if __name__ == '__main__':

	# =============STEP 0: LOADING DATA=================
	print('==> Loading image data...')
	img = imread(urllib.request.urlopen('https://variety.com/wp-content/uploads/2021/12/doctor-strange.jpg?w=681&h=383&crop=1&resize=681%2C383'))
	
	shuffle_img = img.copy().flatten()
	np.random.shuffle(shuffle_img)
	# reshape the shuffled image
	shuffle_img = shuffle_img.reshape(img.shape)

	# =============STEP 1: RUNNING SVD ON IMAGES=================
	print('==> Running SVD on images...')

	U,S,V = np.linalg.svd(img)
	U_s, S_s, V_s = np.linalg.svd(shuffle_img)

	# =============STEP 2: SINGULAR VALUE DROPOFF=================
	print('==> Singular value dropoff plot...')
	k = 100
	plt.style.use('ggplot')
	orig_S_plot, = plt.plot(S[:k], 'b')
	shuf_S_plot, = plt.plot(S_s[:k], 'r')

	plt.legend((orig_S_plot, shuf_S_plot), \
		('original', 'shuffled'), loc = 'best')
	plt.title('Singular Value Dropoff for Clown Image')
	plt.ylabel('singular values')
	plt.savefig('dropoff.png', format='png')
	plt.close()

	# =============STEP 3: RECONSTRUCTION=================
	print('==> Reconstruction with different ranks...')
	rank_list = [2, 10, 20]
	plt.subplot(2, 2, 1)
	plt.imshow(img, cmap='Greys_r')
	plt.axis('off')
	plt.title('Original Image')

	
	for index in range(len(rank_list)):
		k = rank_list[index]
		plt.subplot(2, 2, 2 + index)

		img_recons = U[:, :k] @ np.diag(S)[:k, :k] @ V[:k, :]
		plt.imshow(img_recons, cmap='Greys_r')

		plt.title('Rank {} Approximation'.format(k))
		plt.axis('off')

	plt.tight_layout()
	plt.savefig('reconstruction.png', format='png')
	plt.close()
