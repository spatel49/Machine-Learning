import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from statistics import mean
from statistics import stdev

if __name__ == "__main__":
    data  = pd.read_csv("pima-indians-diabetes.csv")
    x_data, y_data = data.iloc[:,0:8], data.iloc[:,-1]

    x_std = StandardScaler().fit_transform(x_data)
    
    features = x_std.T
    
    covariance_matrix = np.cov(features)
    print("Covariance Matrix: ", covariance_matrix)
    
    eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
    print("\n \nEigenvectors \n %s" %eig_vecs)
    print("\n \nEigenvalues \n %s" %eig_vals)
    
    print("\n \nSince the first, second, and last eigen values are the greatest we will pick them for 3 dimensions.")
    # print( (eig_vals[0] + eig_vals[1] + eig_vals[7]) / sum(eig_vals))
    
    # projected_X = x_std.dot(eig_vecs.T[0])
    # print(projected_X)    
    # result = pd.DataFrame(projected_X, columns=['PC1'])
    # result['label'] = y_data
    # print(result)
    
    x_data = pd.DataFrame(x_std)

    # make the eigen values and vectors into pairs
    eig_tup = []
    for i in range(len(eig_vals)):
        eig_tup += [(np.abs(eig_vals[i]), eig_vecs[:,i])]
    
    print("Selected eigenvalues: ", eig_vals[0], eig_vals[1], eig_vals[7])
    eig_tup.sort(reverse=True)
    eig_arr = [eig_tup[i][1] for i in range(3)]
    eig_arr=np.array(eig_arr)
    row_data = np.dot(x_data,eig_arr.T)
    
    a = []
    gnb = GaussianNB()
    
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(row_data,y_data,test_size=0.5)
        gnb.fit(x_train, y_train)
        output = gnb.score(x_test,y_test)
        a += [output]
    
    print("\n List:", a)
    print("\n \n Mean Accuracy= ", mean(a), ", Standard Deviation: ", stdev(a))
    print("\n")