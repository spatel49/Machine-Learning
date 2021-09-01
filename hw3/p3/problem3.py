import pandas as pd
import numpy as np
import random as rd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from statistics import mean
from statistics import stdev

if __name__ == "__main__":
    data  = pd.read_csv("pima-indians-diabetes.csv")
    x = data.iloc[:,0:8]
    y = data.iloc[:,-1]
    x_data, y_data = data.iloc[:,0:8], data.iloc[:,-1]

    data  = pd.read_csv("pima-indians-diabetes.csv")
    D_1 = np.array(data.iloc[0:500,0:8])
    D_2 = np.array(data.iloc[501:768,0:8])

    n_1 = len(D_1)
    n_2 = len(D_2)
    
    print("CLASS 1: ", D_1)
    print("CLASS 2: ", D_2)

    mean_1=np.mean(D_1.T,axis=1).reshape((8,1))
    mean_2=np.mean(D_2.T,axis=1).reshape((8,1))

    print("\n \nMEAN 1:" , mean_1)
    print("MEAN 2:" , mean_2)

    S_1 = 7*np.cov(D_1.T)
    S_2 = 7*np.cov(D_2.T)

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

    lda = LinearDiscriminantAnalysis()

    x_lda = lda.fit_transform(x, y)

    a = []
    gnb = GaussianNB()
    
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(x_lda,y,test_size=0.5)
        gnb.fit(x_train, y_train)
        output = gnb.score(x_test,y_test)
        a += [output]
    
    print("\n List:", a)
    print("\n \n Mean Accuracy= ", mean(a), ", Standard Deviation: ", stdev(a))
    print("\n")