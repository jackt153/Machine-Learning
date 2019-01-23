#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 13:26:01 2018

@author: jack
"""


import pandas as pd
import numpy as np
import math
import sys
from sklearn.metrics import f1_score

class anomaly_detection:
    
    def __init__(self, df_X):
         self.df_X = df_X
         
         m,n = df_X.values.shape
         
         self.sigma = np.zeros((n,n))
         self.mu = np.zeros((n))

    def Guassion_Parameters(self):
        ##
        #Obtain the mean and covariance matrix need for the Guassian curve
        ##
        X_mean = self.df_X.mean().values
        X = self.df_X.values
        
        m, n = X.shape # number of training examples, number of features
        
        #Creates the covariance Matrix
        Sigma = np.zeros((n,n))
        for i in range(0,m):
    
            Sigma = Sigma +  (X[i]-X_mean).reshape(n,1).dot((X[i]-X_mean).reshape(1,n))
        
        Sigma = Sigma * (1.0/m)
        
        self.mu = X_mean
        self.sigma = Sigma


    def multivariateGaussian(self, df_X):
        ##
        #Calculates the P-Value based on the parameters found above.
        #This is the vectorised form of the equation.
        ##
        
        X = df_X
        
        m, n = X.shape # number of training examples, number of features
    
        X = X.values - self.mu.reshape(1,n) # (X - mu)
    
        # vectorized implementation of calculating p(x) for each m examples: p is m length array
        p = (1.0 / (math.pow((2 * math.pi), n / 2.0) * math.pow(np.linalg.det(self.sigma),0.5))) * np.exp(-0.5 * np.sum(X.dot(np.linalg.pinv(self.sigma)) * X, axis=1))
    
        return p

    def Best_Epsilon(self, Yval ,Pred):
        ##
        #Obtains the best epsilon when the F1 score is maxamised.
        ##
        
        
        bestF1 = 0
        bestEpsilon = 0
    
        stepsize = (max(Pred) - min(Pred)) / 1000
        for epsilon in np.arange(min(Pred), max(Pred), stepsize):
            
            predictions = (Pred < epsilon).astype(int)
            
            F1 = f1_score(Yval, predictions)
            
            if F1 > bestF1:
                bestF1 = F1
                bestEpsilon = epsilon
                
        return(bestEpsilon, bestF1)
    
if __name__ == '__main__':
    




