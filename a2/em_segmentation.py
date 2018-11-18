from io_data import read_data, write_data
import os
import numpy as np
from scipy.stats import multivariate_normal
np.random.seed(5)

def run_em_algorithm(X, min_em_iterations):

	'''
	This function runs the EM algorithm on the image to find 2 segments and returns the mask.

	Input: 
	'X' is the numpy array representing the pre-processed (standardized) image of size (N,M), where
	N = number of pixels
	M = number of features. 
	The features used are (x, y, L, a, b) which are standardised beforehand (subtract the mean, divide by standard deviation)
	
	'min_em_iterations' is the minimum number of iterations that the EM algorithm will run for even if it converges.

	Output:
	'mask' - an array of size M which contains values either 0 or 1 representing the segment. 
	'''


	#EM initialisation is of utmost importance to prevent convergence to local maxima
	'''
	initialise the means in the following manner:
	1) Segment the image into 2 'windows', the center of the image and the background. Refer to report for the specifics.
	2) take the mean of the (x,y,L,a,b) values as the initial mean. 

	'''
	N, M = X.shape[0], X.shape[1]
	half_range = max(X[:,0])/2
	X0 = list()
	X1 = list()
	for i in range(X.shape[0]):
		if abs(X[i,0]) < half_range and abs(X[i,1]) < half_range:
			X0.append(list(X[i,:]))
		else:
			X1.append(list(X[i,:]))
	X0 = np.array(X0)
	X1 = np.array(X1)
	means = np.zeros(shape=(2,5))
	means[0,:] = np.mean(X0, axis=0)
	means[1,:] = np.mean(X1, axis=0)

	#Initialise the covariances to be diagonal matrices with unit values
	covariances = np.zeros(shape=(2,M,M))
	np.fill_diagonal(covariances[0,:,:],1)
	np.fill_diagonal(covariances[1,:,:],1)

	# Gamma intialised to zero
	gamma = np.zeros(shape=(N,2))

	#Mixing coefficients initialised to 0.5 each
	mixers = np.array([0.5,0.5])

	denominator = multivariate_normal.pdf(X,means[0,:],covariances[0,:,:]) + multivariate_normal.pdf(X,means[1,:],covariances[1,:,:])
	s0 = multivariate_normal.pdf(X,means[0,:],covariances[0,:,:])/denominator
	s1 = multivariate_normal.pdf(X,means[1,:],covariances[1,:,:])/denominator

	log_likelihood = np.log(mixers[0]*s0 + mixers[1]*s1)
	sum_log_likelihood = np.sum(log_likelihood)

	previous_log_likelihood = sum_log_likelihood
	convergence = False
	n = 0
	while not convergence:
		n += 1

		#E-Step
		denominator = mixers[0]*multivariate_normal.pdf(X,means[0,:],covariances[0,:,:]) + mixers[1]*multivariate_normal.pdf(X,means[1,:],covariances[1,:,:])
		gamma[:,0] = mixers[0]*multivariate_normal.pdf(X,means[0,:],covariances[0,:,:])/denominator
		gamma[:,1] = mixers[1]*multivariate_normal.pdf(X,means[1,:],covariances[1,:,:])/denominator

		#M-Step
		Nk = np.sum(gamma, axis=0)

		means = np.matmul(np.transpose(gamma),X)/Nk[:,None]

		gamma0 = gamma[:,0]
		gamma0 = gamma0[:,np.newaxis]
		covariances[0,:,:] = np.matmul(np.transpose(gamma0*(X-means[0,:])),(gamma0*(X-means[0,:])))/Nk[0]
		gamma1 = gamma[:,1]
		gamma1 = gamma1[:,np.newaxis]
		covariances[1,:,:] = np.matmul(np.transpose(gamma1*(X-means[1,:])),(gamma1*(X-means[1,:])))/Nk[1]

		mixers = Nk/N

		denominator = multivariate_normal.pdf(X,means[0,:],covariances[0,:,:]) + multivariate_normal.pdf(X,means[1,:],covariances[1,:,:])
		s0 = multivariate_normal.pdf(X,means[0,:],covariances[0,:,:])/denominator
		s1 = multivariate_normal.pdf(X,means[1,:],covariances[1,:,:])/denominator	
		log_likelihood = np.log(mixers[0]*s0 + mixers[1]*s1)
		sum_log_likelihood = np.sum(log_likelihood)
		
		if (previous_log_likelihood/sum_log_likelihood) - 1 <0.00001 and n>=min_em_iterations:
			convergence = True
			print ("Iteration : {:d}. Convergence at log likelihood : {:.2f}.".format(n,sum_log_likelihood))

		previous_log_likelihood = sum_log_likelihood

	mask = np.array(gamma[:,0]<gamma[:,1]).astype(int)

	return mask



def pre_process(data):
	# Standardize the data. Subtract the mean and divide by standard deviation

	processed_data = np.zeros(shape=data.shape)
	for i in range(data.shape[1]):
		processed_data[:,i] = (data[:,i] - np.mean(data[:,i]))/np.std(data[:,i])
	return processed_data


def segment_image(data):

	processed_data = pre_process(data)
	min_em_iterations = 10
	mask = run_em_algorithm(processed_data, min_em_iterations)

	inverse_mask = mask == 0
	inverse_mask = inverse_mask.astype(int)

	mask_image = np.zeros(shape=data.shape)
	mask_image[:,0:2] = data[:,0:2]
	mask_image[:,2] = mask*100

	foreground = np.zeros(shape=data.shape)
	foreground[:,0:2] = data[:,0:2]
	for i in range(2,5):
		foreground[:,i] = np.multiply(data[:,i], inverse_mask)



	background = np.zeros(shape=data.shape)
	background[:,0:2] = data[:,0:2]
	for i in range(2,5):
		background[:,i] = np.multiply(data[:,i], mask)

	return mask_image, foreground, background

if __name__=='__main__':
	values = ['fox','owl','zebra','cow']
	for i in values:
		print ("Now Processing : " + str(i) + ".txt")
		filename = "data" + os.sep + str(i) +".txt"
		data, image = read_data(filename, is_RGB = False)
		mask, foreground, background = segment_image(data)
		output_filename = "results" + os.sep + str(i) + "_mask.txt"
		write_data(mask, output_filename)
		data, image = read_data(output_filename, is_RGB=False, save=True)
		output_filename =  "results" + os.sep + str(i) + "_foreground.txt"
		write_data(foreground, output_filename)
		data, image = read_data(output_filename, is_RGB=False, save=True)
		output_filename = "results" + os.sep + str(i) + "_background.txt"
		write_data(background, output_filename)
		data, image = read_data(output_filename, is_RGB=False, save=True)