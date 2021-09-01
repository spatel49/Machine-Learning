from numpy.random import normal
from numpy import mean as n_mean
from numpy import var as n_var
from math import sqrt

if __name__ == "__main__":
    while True:
        try:
            print("Please enter an integer for mean, variance, and N.")
            mean = float(input("Enter Mean (integer): "))
            variance = float(input("Enter Variance (positive integer): "))
            n = float(input("Enter N (postive integer): "))
        except:
            print("Try again.")
            break

        if variance < 0 or n < 0:
            print("Try again.")
            break

        dataset = []
        while (n>0):
            dataset.append(normal(mean,sqrt(variance)))
            n-=1
        
        print("Data: [",end="")
        for datapoint in dataset:
            if (datapoint == dataset[-1]):
                print("%.2f]"%(datapoint))
            else:
                print("%.2f"%(datapoint),end=", ")
        print("Mean: %.2f, Variance: %.3f"%(n_mean(dataset),n_var(dataset)))
        break
