import scipy.io
import numpy as np
from scipy.stats import multivariate_normal

class MoG:
    
    def __init__(self, X, k):
        
        self.k = k
        
        #m = instances, n= dimensions
        m,n = X.shape
        print("Instances: ", m, "Dimension: ",n)
        
        #Initalise mean of Guassian
        self.mu = np.random.rand(n,k)
        for i in range(0,k):
            self.mu[:,i] = self.mu[:,i]* (X.max(axis=0) - X.min(axis=0))
        
        # mixing proportions
        self.p = (np.ones((1,k))/k).reshape(k,1)
        #print(self.p)
        
        #Creates covariance matrix
        self.Sigma = np.zeros((n,n,k))
        for c in range(0,k):
            
            self.Sigma[:,:,c] = self.Covariance_Matrix(X, self.mu[:,c], np.zeros((n,n)), np.ones((m)), m)
        
        self.Sigma = self.Sigma * 1/k
        
        self.Z = np.zeros((m,k))
        
        self.X = X
        
    def Covariance_Matrix(self, X, mu, Sigma, Z, mc):
        
        m,n = X.shape
        #Intialise Covariance matrix
        #Sigma = np.zeros((n,n,k))
        #initially set to fraction of data covariance
        for i in range(0,m):
        
            Sigma = Sigma  + Z[i] * (X[i]-mu).reshape(n,1).dot((X[i]-mu).reshape(1,n))
        
        Sigma = Sigma * (1.0/mc)
    
        #Set off Diagnoal elements to zero
        Sigma = np.diag(np.diag(Sigma))
        
        return(Sigma)
    
    def Fit(self):
        
        m,n = self.X.shape
        
        llog = []
        
        for iters in range(0,100):
            
            #Step E fix the Mean, Std, P 
            
            for c in range(0,self.k):
                
                 #Z[:,c] = multivariateGaussian(X, Sigma[:,:,c] ,mu[:,c], p[c])
                 self.Z[:,c] = self.p[c] * multivariate_normal.pdf(self.X, self.mu[:,c], self.Sigma[:,:,c])
            
            self.Z = self.Z/self.Z.sum(axis=1)[:,None]
            
            self.mu = np.zeros((n,self.k))
            m_c = np.zeros((self.k))
            self.Simga = np.zeros((n,n,self.k))
            #Step M update mean covariance matrix and P
            for c in range(0,self.k):
                
                m_c[c] = sum(self.Z[:,c])
                
                #Updates the mean
                self.mu[:,c] = np.sum(self.X * self.Z[:,c].reshape(m,1), axis=0)/m_c[c]
                
                #Updates the fraction of the total assigned to cluster c
                self.p[c] = self.Z[:,c].mean()
                
                #Updates the covariance matrix
                self.Sigma[:,:,c] =  self.Covariance_Matrix(self.X, self.mu[:,c], np.zeros((n,n)), self.Z[:,c], m_c[c])
            
            #Calculate loglikihood.    
            S = np.zeros((m,self.k))
            for c in range(0,self.k):
                
                 #Z[:,c] = multivariateGaussian(X, Sigma[:,:,c] ,mu[:,c], p[c])
                 S[:,c] = self.p[c] * multivariate_normal.pdf(self.X, self.mu[:,c], self.Sigma[:,:,c])

            llog.append(np.log(np.sum(S)))
             
        return(self.Sigma, self.mu, self.p, llog)
        
    def Prediction(self, X, pred_type):
    
        m,n = X.shape
        
        Z = np.zeros((m,self.k))
        
        for c in range(0,self.k):
                        
             #Z[:,c] = multivariateGaussian(X, Sigma[:,:,c] ,mu[:,c], p[c])
            Z[:,c] = self.p[c] * multivariate_normal.pdf(X, self.mu[:,c], self.Sigma[:,:,c])
            
        if pred_type == "K":
        
            Z = Z/Z.sum(axis=1)[:,None]
            
            Results = np.argmax(Z, axis=1).reshape(m,1)
            
            return(Results)
        
        if pred_type == "P_Value":
            
            return(np.sum(Z, axis=1))
        
        else:
            
            return(print("No Type Provided"))
            
