'''
    三角基MoM计算二维物体远场散射
'''
import numpy as np 


number_of_point = 500
number_of_Primary_Funcition = 100


class Object:
    '''
        形状类
        包括离散点数N，离散后的r随角度phi的函数
    '''
    def __init__(self, r, N=number_of_point):
        self.N = N
        self.r = r

#基函数的离散化角度，在2pi上平均分
angel_sample = 2*np.pi*np.arange(number_of_Primary_Funcition+2) / (number_of_Primary_Funcition+1)

def primary_function(phi, n, angel=angel_sample):
    '''
        第n个基函数在phi上的值
    '''
    if angel[n-1] <= phi <= angel[n]:
    	return (phi-angel[n-1]) / (angel[n]-angel[n-1])
    else if angel[n] < phi <= angel[n+1]:
    	return (angel[n+1]-phi) / (angel[n+1]-angel[n])
    else
        return 0

def value_R(phi_n, phi_m):
	