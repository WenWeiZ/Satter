import numpy as np 
import pandas as pd

A = np.matrix([[1, 1], [1, 1]])
B = np.matrix([[1, 1], [2, 1]])
assert (A==B).all(), 'J0不是对称矩阵'