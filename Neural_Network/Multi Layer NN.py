# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 21:55:49 2019

@author: Jack
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from NN_Class import NN

def Graph_Confusion_Matrix(CM, labels):
    #np.fill_diagonal(CM,0)
    
    plt.figure(figsize = (8,8))
    sn.set(font_scale=1.4)#for label size
    sn.heatmap(CM, annot=True,annot_kws={"size": 16}, fmt='g'
               ,xticklabels = labels
               ,yticklabels = labels)# font size
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    

Path = "C:\\Users\\Jack\\OneDrive\Kaggle\\Digit_Recogniser\\train.csv"
Path_Test = "C:\\Users\\Jack\\OneDrive\Kaggle\\Digit_Recogniser\\test.csv"

df_train = pd.read_csv(Path)
#This randomise the trainning data (frac = 1 is sample of 100%)
df_train = df_train.sample(frac=1).reset_index(drop=True)

df_test = pd.read_csv(Path_Test)

y_ = df_train["label"].copy()
X_ = df_train.copy()
X_.drop(columns=["label"], inplace=True)

y_ = pd.get_dummies(y_)

#Pandas Dataframe to Numpy array
X_ = X_.values
y_ = y_.values

#Normalisation
X_norm = X_/255.0
    

#Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_norm, y_, test_size=0.20, random_state=42)

#We want to select the number of layers and the nodes in each layer. First layer is number of pixels
# last layer is 10 for number of labels
layers = [X_train.shape[1],100, 512, 256, 128  ,10]       

Neural_N   = NN(X = X_train,
                y = y_train,
                layers = layers,
                alpha = 0.1,
                batch_size = 256,
                n_epochs = 20)

Cost_Train = Neural_N.Fit()

plt.figure(figsize=(7,5))            
plt.plot(Cost_Train)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Cost of Trainning Dataset by Epoch")
plt.show()
        


Y_pred = np.argmax(Neural_N.Predict(X_test), axis=0)
Y_actual = np.argmax(y_test,axis=1)

CM = confusion_matrix(Y_actual, Y_pred)
Graph_Confusion_Matrix(CM, list(np.linspace(0,9,10)))
print("Accuary:", accuracy_score(Y_actual, Y_pred)*100)













