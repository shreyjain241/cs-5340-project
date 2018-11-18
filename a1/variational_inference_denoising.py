from io_data import read_data, write_data
import os
import numpy as np
from scipy.stats import norm
from scipy.special import expit


def run_variational_inference(X,params):
	'''
	This function runs the variational inference algorithm on the data input X and returns all iterations
	Input:
	'X' is a numpy array of size (image_width, image_height) with all values either 1 or -1
	'params' is a dict containing values for coupling_strength, c, and iterations

	Output:
	Returns the result of each iteration of the sampling in a numpy matrix of size (image_width, image_height, iterations)
	'''
	J = params['coupling_strength']
	c = params['c']
	n_iters = params['iterations']
	processed_data = np.zeros(shape=(X.shape[0],X.shape[1],n_iters+1))

	logodds = norm.logpdf(X,1,c) - norm.logpdf(X,-1,c)
	p1 = expit(logodds)
	mu = 2*p1 - 1
	processed_data[:,:,0] = np.sign(mu)

	for n in range(1,n_iters+1):
		print ("Running Iteration : {:d} of {:d}".format(n,n_iters))
		mu_new = mu
		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				sum_nn = 0
				if i > 0:
					sum_nn = sum_nn + mu[i-1,j]
				if i < X.shape[0] - 1:
					sum_nn = sum_nn + mu[i+1,j]
				if j > 0:
					sum_nn = sum_nn + mu[i,j-1]
				if j < X.shape[1] - 1:
					sum_nn = sum_nn + mu[i,j+1]
				coupled_sum = J*sum_nn
				mu_new[i,j] = np.tanh(coupled_sum + 0.5*logodds[i,j])
		mu = mu_new
		processed_data[:,:,n] = np.sign(mu)

	return processed_data

def get_mrf(data):
	# function to convert raw image data into a numpy matrix of size (image_width, image_height) and values 1 and -1.
	rows = max(data[:,0].astype(int)) + 1
	cols = max(data[:,1].astype(int)) + 1
	X = data[:,2].astype(int).reshape(rows,cols)
	X = np.divide(np.multiply(X,2),255).astype(int) - 1
	return X

def mrf_to_txt(data):
	# function to convert the mrf numpy matrix into the format of the orginal image.
	txt_data = np.zeros((data.shape[0]*data.shape[1],3,data.shape[2]))
	for k in range(data.shape[2]):
		n = 0
		for i in range(data.shape[0]):
			for j in range(data.shape[1]):
				txt_data[n][0][k] = i
				txt_data[n][1][k] = j
				txt_data[n][2][k] = int(127.5*data[i][j][k] + 127.5)
				n += 1
	return txt_data

def denoise_data(data):
	X = get_mrf(data)
	var_inf_params = {
	'coupling_strength' : 3,
	'c' : 0.5,
	'iterations' : 5
	}
	denoised_iterations = run_variational_inference(X,var_inf_params)
	txt_data = mrf_to_txt(denoised_iterations)
	return txt_data

if __name__=='__main__':
	for i in range(1,5):
		print ("Now Processing : " + str(i) + "_noise.txt")
		filename = "data" + os.sep + str(i) +"_noise.txt"
		data, image = read_data(filename, is_RGB = True)
		denoised_data = denoise_data(data)
		for j in range(denoised_data.shape[2]):
			if j == 0:
				output_filename = "variational_inference_denoising_results" + os.sep + str(i) + "_original.txt"
				write_data(denoised_data[:,:,j], output_filename)
				data, image = read_data(output_filename, is_RGB=True, save=True)
			else:
				output_filename = "variational_inference_denoising_results" + os.sep + str(i) + "_denoise_iter{:d}.txt".format(j)
				write_data(denoised_data[:,:,j], output_filename)
				data, image = read_data(output_filename, is_RGB=True, save=True)			
