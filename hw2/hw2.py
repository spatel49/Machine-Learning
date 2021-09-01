import pandas as pd
import numpy as np
import csv

from statistics import mean
from statistics import stdev

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

from scipy.stats import multivariate_normal as mvn

import math

if __name__ == "__main__":
    data = pd.read_csv("pima-indians-diabetes.csv")
    
    #question 1 -----------------------------------------------------------------------------------------------------
    # To train:
    # 1. Separate training data by class
    # 2. Get mean and covariance for each class
    # 3. Build a Gaussian curve using that mean and covariance
    # To classify:
    # 1. Get the CDF for each Gaussian curve (which tells you the probability inside a given area of your Gaussian curve)
    # 2. Take the data you're classifying and plug it into each CDF
    # 3. The data belongs to the class which gave it the highest probability after being plugged into the CDF
    
    # rows = []
    # with open("pima-indians-diabetes.csv", 'r') as csvfile:
    #     csvreader = csv.reader(csvfile) 
    #     for row in csvreader:
    #         if len(row) >= 9:
    #             rows.append([row[1], row[2], row[3], row[8]])
    
    # col_1 = [[], [], []]
    # col_2 = [[], [], []]
    
    # class0 = row
    # class1 = row
    # for i in range(3):
    #     for j in class0:
    #         col_1[i].append(float(j[i]))
    #     for j in class1:
    #         col_2[i].append(float(j[i]))
                            
    # for i in range(3):
    #     col_1[i] = np.array(col_1[i])
    #     col_2[i] = np.array(col_2[i])
    
    # data1 = np.array(col_1)
    # data2 = np.array(col_2)
    
    # cov1 = np.cov(data1, bias = False)
    # cov2 = np.cov(data2, bias = False)
    gnb = GaussianNB()
    a = []
    
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 1:4], data.iloc[:, -1], test_size=0.5)
        gnb.fit(x_train, y_train)
        a += [gnb.score(x_test, y_test)]
    
    print("\n")
    print("Question 1:----------------------------------------------------")
    print("10 Trials:")
    print("Mean= ", mean(a), ", Standard Deviation: ", stdev(a))
    print("\n")
    
    gnb = GaussianNB()
    a = []
    for i in range(50):
        x_train, x_test, y_train, y_test = train_test_split(data.iloc[:, 1:4], data.iloc[:, -1], test_size=0.5)
        gnb.fit(x_train, y_train)
        a += [gnb.score(x_test, y_test)]
    
    print("50 Trials:")
    print("Mean= ", mean(a), ", Standard Deviation: ", stdev(a))
    print("\n")
    
    #question 2 -----------------------------------------------------------------------------------------------------
    a1 = []
    a5 = []
    a11 = []
    
    for i in range(10):
        x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,1:4], data.iloc[:,-1], test_size=0.5)
        
        model1 = KNeighborsClassifier(1)
        model5 = KNeighborsClassifier(5)
        model11 = KNeighborsClassifier(11)
        
        model1.fit(x_train, y_train)
        model5.fit(x_train, y_train)
        model11.fit(x_train, y_train)
        
        a1 += [model1.score(x_test,y_test)]
        a5 += [model5.score(x_test,y_test)]
        a11 += [model11.score(x_test,y_test)]

    print("Question 2:----------------------------------------------------")
    print("10 Trials:")
    print("k = 1 -> Mean = ", mean(a1), ", Standard Deviation = ", stdev(a1))
    print("k = 5 -> Mean = ", mean(a5), ", Standard Deviation = ", stdev(a5))
    print("k = 11 -> Mean = ", mean(a11), ", Standard Deviation = ", stdev(a11))
    print("\n")

    a1 = []
    a5 = []
    a11 = []
    
    for i in range(50):
        x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,1:4], data.iloc[:,-1], test_size=0.5)
        
        model1 = KNeighborsClassifier(1)
        model5 = KNeighborsClassifier(5)
        model11 = KNeighborsClassifier(11)
        
        model1.fit(x_train, y_train)
        model5.fit(x_train, y_train)
        model11.fit(x_train, y_train)
        
        a1 += [model1.score(x_test,y_test)]
        a5 += [model5.score(x_test,y_test)]
        a11 += [model11.score(x_test,y_test)]

    print("50 Trials:")
    print("k = 1 -> Mean = ", mean(a1), ", Standard Deviation = ", stdev(a1))
    print("k = 5 -> Mean = ", mean(a5), ", Standard Deviation = ", stdev(a5))
    print("k = 11 -> Mean = ", mean(a11), ", Standard Deviation = ", stdev(a11))