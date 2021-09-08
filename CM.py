import numpy as np 
import numpy.matlib
import numpy.linalg
from scipy import special
from matplotlib import pyplot as plt


def Distance(p1, p2):
    return np.sqrt(p1[1]**2 + p2[1]**2 - 2*p1[1]*p2[1]*np.cos(p1[0]-p2[0]))

def Delta(point_array):
	'''
        点与点距离序列
	'''
	num = len(point_array)
	delta = []

	for n in range(num-1):
		phi1, r1 = point_array[n][0], point_array[n][1]
		phi2, r2 = point_array[n+1][0], point_array[n+1][1]
		delta.append(np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(phi1-phi2)))
	phi1, r1 = point_array[-1][0], point_array[-1][1]
	phi2, r2 = point_array[0][0], point_array[0][1]		
	delta.append(np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(phi1-phi2)))

	return delta

def Center(point_array):
    '''
        取中点，返回序列
    '''
    num = len(point_array)
    c_point = []

    for n in range(num-1):
        c_point.append(((point_array[n][0]+point_array[(n+1)][0])/2, (point_array[n][1]+point_array[(n+1)][1])/2))
    c_point.append(((point_array[0][0] + point_array[-1][0])/2 + np.pi, (point_array[0][1]+point_array[-1][1])/2))

    return c_point 

def CM(point, kwave):
	'''

	'''
	euler_gamma = np.euler_gamma
	pi = np.pi

	M = len(point)
	delta_point = Delta(point)
	c_point = Center(point)

	Y0 = numpy.matlib.ones((M, M))
	J0 = numpy.matlib.ones((M, M))

	for n in range(M):
		for m in range(n, M):
			d = Distance(c_point[n], c_point[m])
			J0[n, m] = special.j0(kwave*d)
			J0[m, n] = special.j0(kwave*d)
			if m == n:
				Y0[n, n] = 2/pi * (np.log(kwave*delta_point[n]/4) + euler_gamma - 1)
			else:
				Y0[n, m] = special.y0(kwave*d)
				Y0[m, n] = special.y0(kwave*d)

	return numpy.linalg.eigh(numpy.linalg.inv(J0) * Y0)


if __name__ == '__main__':

	number_of_point = 100
	kwave = 2*np.pi

	phi = np.arange(number_of_point) / number_of_point * 2*np.pi 	
	#rho = np.random.rand(number_of_point) + 0.5
	rho = np.ones(number_of_point)

	point = list(zip(phi, rho))

	#print(special.y0(kwave)/special.j0(kwave))
	print(CM(point, kwave)[0])