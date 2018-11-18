import numpy as np
from scipy.stats import norm


sigma = 0.1
J = 2


sum_nn = [4,2,0]


for J in range(1,4):
	for s in range(1, 7):
		sigma = s/10

		prob_equal = norm.pdf(1,1,sigma)/(norm.pdf(1,1,sigma) + norm.pdf(-1,1,sigma))
		prob_unequal = norm.pdf(-1,1,sigma)/(norm.pdf(1,1,sigma) + norm.pdf(-1,1,sigma))
		a = prob_unequal*np.exp(np.multiply(J,sum_nn))
		b = prob_equal*np.exp(np.multiply(-J,sum_nn))

		p_flip = a/(a+b)
		print (p_flip, J, sigma)