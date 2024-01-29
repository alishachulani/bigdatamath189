
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


###########################################
#	    	Main Driver Function       	  #
###########################################


if __name__ == '__main__':

	# =============part c: Plot data and the optimal linear fit=================
	# NOTE: to finish this part, you need to finish the part(a) of this part
	# 		first

	# load the four data points of tihs problem
	X = np.array([0, 2, 3, 4])
	y = np.array([1, 3, 6, 8])

	# plot four data points on the plot
	plt.style.use('ggplot')
	plt.plot(X, y, 'ro')


	# 		part (a), note that y = mx + b
	m_opt = 62./35
	b_opt = 18./35



	# HINT:
	#	1) Use np.linspace to get the x-coordinate of 100 points
	#	2) Calculate the y-coordinate of those 100 points with the m_opt and
	#	   b_opt, remember y = mx+b.
	#	3) Use a.reshape(-1,1), where a is a np.array, to reshape the array
	#	   to appropraite shape for generating plot

	X_space = np.linspace(-1, 5, num=100).reshape(-1, 1)
	y_space = (m_opt * X_space + b_opt).reshape(-1, 1)

	# plot the optimal learn fit you obtained and save it to your current
	# folder
	plt.plot(X_space, y_space)
	plt.savefig('hw1pr2c.png', format='png')
	plt.close()


	# =============part d: Optimal linear fit with random data points=================

	# variables to start with
	mu, sigma, sampleSize = 0, 1, 100

	noise = np.random.normal(mu, sigma, sampleSize).reshape(-1, 1)
	y_space_rand = m_opt * X_space + b_opt + noise
	X_space_stacked = np.hstack((np.ones_like(y_space), X_space))
	W_opt = np.linalg.solve(X_space_stacked.T @ X_space_stacked,
	X_space_stacked.T @ y_space_rand)
	b_rand_opt, m_rand_opt = W_opt.item(0), W_opt.item(1)
	# calculate predicted values
	y_pred_rand = np.array([m_rand_opt * x + b_rand_opt for x in X_space]).reshape(-1, 1)


	# generate plots with legend
	plt.plot(X, y, 'ro')
	orig_plot, = plt.plot(X_space, y_space, 'r')
	plt.plot(X_space, y_space_rand, 'bo')
	rand_plot, = plt.plot(X_space, y_pred_rand, 'b')
	plt.legend((orig_plot, rand_plot), \
	('original fit', 'fit with noise'), loc = 'best')
	plt.savefig('hw1pr2d.png', format='png')
	plt.close()
