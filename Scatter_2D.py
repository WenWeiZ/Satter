'''
    三角基MoM计算二维物体远场散射
'''
import numpy as np 
import numpy.matlib
import numpy.linalg
from scipy import special
from matplotlib import pyplot as plt


class Object:
    '''
        形状类
        包括离散点数N，离散后的r随角度phi的函数
    '''
    def __init__(self, phi_array, r_array):
        self.phi_array = phi_array
        self.r_array = r_array
        self.len = len(phi_array)
        assert self.len == len(r_array), 'length not match'


    def rho(self, phi):
        '''
            返回对应角度的长度，向下取最近的离散角度
        '''
        index = np.sum(phi >= self.phi_array)
        return self.r_array[index]


    def distance(self, phi1, phi2):
        '''
            计算两角度对应长度的距离
        '''
        r1 = self.rho(phi1)
        r2 = self.rho(phi2)
        return np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(phi1-phi2))


class TriangularBasis2D:
    '''
        2D三角基函数类
    '''
    def __init__(self, discret_angel):
        self.discret_angel = np.hstack((discret_angel, discret_angel[0] + 2*np.pi, discret_angel[-1] - 2*np.pi)) #这里有一个很巧妙地设计，-1放在最后一位即第N+2个
        self.len = len(discret_angel)


    def value(self, phi, n):
        '''
        第n个基函数在phi上的值
        n取0，1，2 ... N-1
        '''
        angel = self.discret_angel

        if angel[n-1] <= phi <= angel[n]:
            return (phi-angel[n-1]) / (angel[n]-angel[n-1])
        elif angel[n] < phi <= angel[n+1]:
            return (angel[n+1]-phi) / (angel[n+1]-angel[n])
        else:
            return 0


def value_Z(Object, Basis, k, m, n):
    '''
        Z_{m,n} 
    '''
    phi = Object.phi_array
    phi = np.hstack((phi, phi[0])) #预处理，头尾相接
    delta_phi = np.hstack((np.array([x - y for (x, y) in zip(phi[1:], phi[:-1])]), phi[0]+2*np.pi-phi[-1]))
    end = Object.len  

    primary_function = Basis.value

    value = 0
    gamma = 1.78107 #处理汉克尔函数奇异性的系数

    for k_m in range(0, end, 2):
        delta_phi_m = phi[k_m+2] - phi[k_m]
        Fm = primary_function(phi[k_m+1], m)
        if Fm == 0:
            continue
        inner_value = 0
        for k_n in range(0, end, 2):
            delta_phi_n = phi[k_n+2] - phi[k_n]
            Fn = primary_function(phi[k_n+1], n)
            if Fn == 0:
                continue
            elif k_m == k_n:
                delta_R = abs((Object.rho(phi[k_n+2]) - Object.rho(phi[k_n])))
                distance = Object.distance(phi[k_n+2], phi[k_n])
                inner_value += (1 - 1.j*2/np.pi*(np.log(gamma*k*distance/4) - 1)) * Fn * delta_phi_n * distance / delta_R    #奇异性处理
            else:
                distance = Object.distance(phi[k_m+1], phi[k_n+1])
                inner_value += special.hankel2(0, k * distance) * Fn * delta_phi[k_n]
            '''debug
            print('m:', k_m, 'n:', k_n)
            print('phi:', phi[k_m], phi[k_n])
            print('r:', Object.rho(phi[k_m]), Object.rho(phi[k_n]))
            print('inner_value:', inner_value)
            '''
        value += Fm * inner_value * delta_phi_m

    return value

def matrix_Z(Object, Basis, k):
    '''
        Z矩阵生成，利用value_Z函数
    '''
    M = Basis.len

    matrix = numpy.matlib.ones((M, M), dtype=complex)

    for m in range(M):
        for n in range(M):
            matrix[m, n] = value_Z(Object, Basis, k, m, n)

    return matrix

def value_V(Object, Basis, k, m):
    '''
        V_m
    '''
    phi = Object.phi_array
    delta_phi = np.array([x - y for (x, y) in zip(phi[1:], phi[:-1])])
    step = Object.len - 1 #间隔数是点数减一

    primary_function = Basis.value

    value = 0

    for n in range(step):
        Fm = primary_function(phi[n], m)
        if Fm == 0:
            continue
        value += Fm * np.exp(-1.j * k * Object.rho(phi[n]) * np.cos(phi[n]) * delta_phi[n])

    return value

def matrix_V(Object, Basis, k):
    '''
        V矩阵生成，利用value_V函数
    '''    
    M = Basis.len
    matrix = numpy.matlib.ones((M, 1), dtype=complex)

    for m in range(M):
        matrix[m, 0] = value_V(Object, Basis, k, m)

    return matrix 

def alpha(Object, Basis, n, m, k):
    '''
        最后计算的中间系数，简化计算
    '''
    phi = Object.phi_array
    delta_phi = np.array([x - y for (x, y) in zip(phi[1:], phi[:-1])])
    step = Object.len - 1 #间隔数是点数减一

    primary_function = Basis.value

    value = 0

    for l in range(step):
        value += special.jv(n, k*abs((Object.rho(phi[n]) - Object.rho(phi[l])))) * primary_function(phi[l], m) * np.exp(-1.j * n * phi[l]) * delta_phi[l]

    return value 


def alpha_matrix(Object, Basis, n, k):
    '''
        矩阵Alpha
    '''    
    M = Basis.len
    matrix = numpy.matlib.ones((1, M), dtype=complex)

    for m in range(M):
        matrix[0, m] = alpha(Object, Basis, n, m, k)
    
    return matrix

def Cn(Object, Basis, n, k, I):
    '''
        所求系数
    '''
    alpha_nm = alpha_matrix(Object, Basis, n, k)
    return 1 / (-1.j) ** n * (alpha_nm*I)


if __name__ == '__main__':

    half_number_of_point = 300
    number_of_point = 2 * half_number_of_point #确保偶数，用于积分离散化的时候可以取到中点

    number_of_Primary_Funcition = 100
    k = 2*np.pi / 0.1

    #基函数的离散化角度，在2pi上平均分并创建基函数类basis
    discret_angel = 2*np.pi*np.arange(1, number_of_Primary_Funcition+1) / (number_of_Primary_Funcition+1)
    basis = TriangularBasis2D(discret_angel)

    #生成形状实例
    phi_array = 2*np.pi*np.arange(number_of_point) / number_of_point
    rho_array = 0.5 + np.random.rand(1, number_of_point)[0]

    ob = Object(phi_array, rho_array)

    #print(value_Z(ob, basis, k, 0, 0))
    
    #计算
    matrixz = matrix_Z(ob, basis, k)
    matrixv = matrix_V(ob, basis, k)

    #print(matrixz[4,:])

    I = numpy.linalg.inv(matrixz) * matrixv 
    #print(I)

    c0 = Cn(ob, basis, 1, k, I)
    print(c0)










