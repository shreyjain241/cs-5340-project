from io_data import read_data, write_data
import os
import numpy as np
from scipy.stats import norm


def run_gibbs_sampling(X,params):
	n_iters = params['iterations']
	sigma = params['sigma']
	J = params['coupling_strength']
	samples = np.zeros((X.shape[0],X.shape[1],n_iters+1),dtype=int)

	prob_equal = norm.pdf(1,1,sigma)/(norm.pdf(1,1,sigma) + norm.pdf(-1,1,sigma))
	prob_unequal = norm.pdf(-1,1,sigma)/(norm.pdf(1,1,sigma) + norm.pdf(-1,1,sigma))

	#cdf_unequal = norm.cdf(0,1,sigma)
	#cdf_equal = norm.cdf(0,-1,sigma)

	X0 = X
	samples[:,:,0] = X0
	for n in range(1,n_iters+1):
		print ("Running Iteration : {:d} of {:d}".format(n,n_iters))

		randn = np.random.uniform(size=X.shape)

		for i in range(X.shape[0]):
			for j in range(X.shape[1]):
				sum_nn = 0
				if i > 0:
					sum_nn = sum_nn + X0[i-1,j]
				if i < X.shape[0] - 1:
					sum_nn = sum_nn + X0[i+1,j]
				if j > 0:
					sum_nn = sum_nn + X0[i,j-1]
				if j < X.shape[1] - 1:
					sum_nn = sum_nn + X0[i,j+1]

				if X0[i][j] == 1:
					p1 = prob_equal*np.exp(J*sum_nn)/(prob_equal*np.exp(J*sum_nn) + prob_unequal*np.exp(-J*sum_nn))
				if X0[i][j] == -1:
					p1 = prob_unequal*np.exp(J*sum_nn)/(prob_unequal*np.exp(J*sum_nn) + prob_equal*np.exp(-J*sum_nn))

				samples[i][j][n] = np.sign(p1 - randn[i][j]).astype(int)
		X0 = samples[:,:,n]
	return samples

def get_mrf(data):
	rows = max(data[:,0].astype(int)) + 1
	cols = max(data[:,1].astype(int)) + 1
	X = data[:,2].astype(int).reshape(rows,cols)
	X = np.divide(np.multiply(X,2),255).astype(int) - 1
	return X

def mrf_to_txt(data):

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

	gibbs_params = {
	'coupling_strength' : 3,
	'sigma' : 0.5,
	'iterations' : 5
	}
	gibbs_samples = run_gibbs_sampling(X,gibbs_params)
	denoised_data = mrf_to_txt(gibbs_samples)
	return denoised_data


if __name__=='__main__':
	for i in range(1,5):
		print ("Now Processing : " + str(i) + "_noise.txt")
		filename = "data" + os.sep + str(i) +"_noise.txt"
		data, image = read_data(filename, is_RGB = True)
		denoised_data = denoise_data(data)
		for j in range(denoised_data.shape[2]):
			output_filename = "gibbs_denoising_results" + os.sep + str(i) + "_denoise_iter{:d}.txt".format(j)
			write_data(denoised_data[:,:,j], output_filename)
			data, image = read_data(output_filename, is_RGB=True, save=True)

