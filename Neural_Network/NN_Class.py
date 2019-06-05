# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:36:55 2019

@author: Jack
"""

import numpy as np

class NN:
    
    def __init__(self,
                X,
                y,
                layers,
                alpha,
                batch_size,
                n_epochs):
        
         self.X = X
         self.y = y
         self.layers = layers
         self.batch_size = batch_size
         self.alpha = alpha
         self.n_epochs = n_epochs

    def Tanh(self, X):
        return(np.tanh(X))
    
    def Diff_Tanh(self, X):
        return(1 - np.power(np.tanh(X),2)) 
    
    def Softmax(self, X):
        return np.exp(X) / np.sum(np.exp(X), axis=0)
    
    def initalise(self):
    
        W_dict = {}
        
        for i in range(1,len(self.layers)):
            #print(self.layers[i])
            
            W_dict.update({"W_"+str(i):  np.random.randn(self.layers[i], self.layers[i-1])*0.01})
            W_dict.update({"B_"+str(i):  np.zeros([self.layers[i],1])})
            
        return(W_dict)
    
    def Forward(self, X, W_dict):
        
        cache = {}
        
        cache.update({"A_0" : X})
        
        for i in range(1,len(self.layers)):
            
            W = W_dict.get("W_"+str(i))
            B = W_dict.get("B_"+str(i))
            A = cache.get("A_"+str(i-1))
                
            if( i == len(self.layers)-1):
   
                cache.update({"Z_"+str(i): np.dot(W, A) + B})
                cache.update({"A_"+str(i): self.Softmax(cache.get("Z_"+str(i)))})
         
            else:    
        
                cache.update({"Z_"+str(i): np.dot(W, A) + B})
                cache.update({"A_"+str(i): self.Tanh(cache.get("Z_"+str(i)))})

        return(cache)
    
    def Cost(self, Y_actual, Y_pred):
    
        Cost_list = []
        
        for i in range(0,self.batch_size):
        
            Temp_Cost = np.sum(-1 * np.multiply(Y_actual[:,i].reshape(-1,1), np.log(Y_pred[:,i]).reshape(-1,1)))
        
            Cost_list.append(float(Temp_Cost))
        
        return(np.mean(Cost_list))
        
    def Back_Prop(self, W_dict ,cache, Y_actual, m):
        
        for i in range(len(self.layers)-1,0,-1):
            #print(i)
            if i == len(self.layers)-1:
                
                cache.update({"dZ_"+str(i): cache.get("A_"+str(i)) - Y_actual })
        
                cache.update({"dW_"+str(i): np.multiply( (1.0/m) , np.dot(cache.get("dZ_"+str(i)), cache.get("A_"+str(i-1)).T)) })
                cache.update({"dB_"+str(i): np.multiply( (1.0/m) , np.sum(cache.get("dZ_"+str(i)), axis=1, keepdims=True)) })
                
            else:
                
                cache.update({"dZ_"+str(i): np.dot(W_dict.get("W_"+str(i+1)).T, cache.get("dZ_"+str(i+1))) * self.Diff_Tanh(cache.get("Z_"+str(i)))})
        
                cache.update({"dW_"+str(i): np.multiply( (1.0/m) , np.dot(cache.get("dZ_"+str(i)), cache.get("A_"+str(i-1)).T)) })
                cache.update({"dB_"+str(i): np.multiply( (1.0/m) , np.sum(cache.get("dZ_"+str(i)), axis=1, keepdims=True)) })

        
        return(cache)  
        
    
    def Grad_Decent(self, cache, W_dict):
        
        for i in range(1,len(self.layers)):
            W_dict.update({"W_"+str(i): W_dict.get("W_"+str(i)) - np.multiply(self.alpha, cache.get("dW_"+str(i)))}) 
        
        for i in range(1,len(self.layers)):
            W_dict.update({"B_"+str(i): W_dict.get("B_"+str(i)) - np.multiply(self.alpha, cache.get("dB_"+str(i)))})
    
        return(W_dict)
        
        
    def Fit(self):
        
        Cost_Overall = []
        
        W_dict = self.initalise()
        
        for epoch in range(self.n_epochs):
            Cost_Tr_List = []
            Cost_Val_List = []
            Cost_Val_Acc = []
            
            batchnumber = 0
            
            for i in range(len(self.y)// self.batch_size):
                batchnumber = batchnumber+1
                
                batch_start_idx = (i * self.batch_size) % (self.X.shape[0] - self.batch_size)
                batch_end_idx = batch_start_idx + self.batch_size
                batch_X = self.X[batch_start_idx:batch_end_idx]
                batch_Y = self.y[batch_start_idx:batch_end_idx]
                
                cache = self.Forward(batch_X.T, W_dict)
                
                Cost_Tr_List.append(self.Cost(batch_Y.T, cache.get("A_"+str(len(self.layers)-1))))
                
                cache = self.Back_Prop(W_dict, cache, batch_Y.T, self.batch_size)
                
                W_dict = self.Grad_Decent(cache, W_dict)
                
                self.Final_W_dict = dict(W_dict)
                
            print("Cost at epoch" +str(epoch)+ " Cost:"+str(np.mean(Cost_Tr_List)))
            Cost_Overall.append(np.mean(Cost_Tr_List))
            
        return(Cost_Overall)
        
        
    def Predict(self, X):
        
        Predict = {}
        
        Predict.update({"A_0" : X.T})
        
        for i in range(1,len(self.layers)):
            
            W = self.Final_W_dict.get("W_"+str(i))
            B = self.Final_W_dict.get("B_"+str(i))
            A = Predict.get("A_"+str(i-1))
                
            if( i == len(self.layers)-1):
   
                Predict.update({"Z_"+str(i): np.dot(W, A) + B})
                Predict.update({"A_"+str(i): self.Softmax(Predict.get("Z_"+str(i)))})
         
            else:    
        
                Predict.update({"Z_"+str(i): np.dot(W, A) + B})
                Predict.update({"A_"+str(i): self.Tanh(Predict.get("Z_"+str(i)))})

        return(Predict.get("A_"+str(i)))
        
