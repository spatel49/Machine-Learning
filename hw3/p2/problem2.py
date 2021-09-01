import pandas as pd
import numpy as np
import random as rd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda

D_1 = np.array([[-2, 1], [-5, -4], [-3, 1], [0, -3], [-8, -1]])
D_2 = np.array([[2, 5], [1, 0], [5, -1], [-1, -3], [6, 1]])
x = np.concatenate((D_1, D_2), axis=0)

print("CLASS 1: ", D_1)
print("CLASS 2: ", D_2)

n_1 = len(D_1)
n_2 = len(D_2)

mean_1=np.mean(D_1.T,axis=1).reshape((2,1))
mean_2=np.mean(D_2.T,axis=1).reshape((2,1))

print("\n \nMEAN 1:" , mean_1)
print("MEAN 2:" , mean_2)

S_1 = 4*np.cov(D_1.T)
S_2 = 4*np.cov(D_2.T)

print("\n \nS_1:", S_1)
print("S_2:", S_2)

S_w = S_1 + S_2
print("\n \nWithin class scatter: ", S_w)

Sw_inverse = np.linalg.inv(S_w)
print("\n \nInverse within class scatter:", Sw_inverse)

v = np.dot(Sw_inverse, (mean_1 - mean_2))
print("\n \nOptimal Line Direction: ", v)

D1_p = np.dot(v.T, D_1.T) 
D2_p = np.dot(v.T, D_2.T)

print("\n \nClass 1 Projection: ", D1_p)
print("Class 2 Projection: ", D2_p)