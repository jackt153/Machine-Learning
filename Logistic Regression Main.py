# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 14:36:40 2018

@author: Jack
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt    # more on this later

Path = """C:\\Users\\Jack\\OneDrive\\Uni Work\\Machine Learning\\Assignment1-2017\\part2\\logistic_regression\\ex4x.dat"""
Path_2 = """C:\\Users\\Jack\\OneDrive\\Uni Work\\Machine Learning\\Assignment1-2017\\part2\\logistic_regression\\ex4y.dat"""

def Sigmoid(X):
    ans = 1/(1+np.exp(-X))
    return(ans)
    
def CostFunction(theta, X, y):
    m = len(y)
    J = 0
    h = Sigmoid(np.dot(X, theta))
    #h = Sigmoid(X @ theta)undefined
    J = 1/m*((np.dot(-1*y.transpose(), np.log(h))) - (np.dot((1-y).transpose(), np.log(1-h))))
    return(J)

def GradientDecent(theta, X, y):
    h = Sigmoid(np.dot(X,theta))
    return((1/m)*(np.dot(X.transpose(), (h-y))))
    
def Normalisation(X):
    #X = X.values
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_norm = (X - X_mean)/X_std
    return(X_norm, X_mean, X_std)
    

def Finding_Theta_Test_Train(iterations, alpha, X, y, theta, X_test, y_test):
    J_history_train = []
    J_history_test = []
    m=len(y)
    
    for _ in range(iterations):
        #theta = theta * grad_reg_term - (alpha/m)* Gradient(X=X, y=y, theta=theta)
        theta = theta - (alpha/m)* GradientDecent(X=X, y=y, theta=theta)
        
        J_history_train.append(CostFunction(X=X, y=y, theta=theta))
        J_history_test.append(CostFunction(X=X_test, y=y_test, theta=theta))

    #converts array to dataframe so can be easily plotted.
    df_J_history_train = pd.DataFrame.from_records(J_history_train)
    df_J_history_test = pd.DataFrame.from_records(J_history_test)
    return(theta, df_J_history_train, df_J_history_test)
    
def Finding_Theta(iterations, alpha, X, y, theta):
    J_history = []
    m=len(y)
    
    for _ in range(iterations):
        #theta = theta * grad_reg_term - (alpha/m)* Gradient(X=X, y=y, theta=theta)
        theta = theta - (alpha/m)* GradientDecent(X=X, y=y, theta=theta)
        
        J_history.append(CostFunction(X=X, y=y, theta=theta))

    #converts array to dataframe so can be easily plotted.
    df_J_history = pd.DataFrame.from_records(J_history)
    return(theta, df_J_history)
    

X = pd.read_csv(Path, header=None)
y = pd.read_csv(Path_2, header=None)

y.describe()
X.describe()

############# SCATTER PLOT ################################
df = pd.concat([X,y], axis=1)
df.columns = ['P1', 'P2', 'T']
plt.figure(1,figsize=(8,8))
adm = plt.scatter(df["P1"].loc[df['T'] == 1], df["P2"].loc[df['T'] == 1])
not_adm =  plt.scatter(df["P1"].loc[df['T'] == 0], df["P2"].loc[df['T'] == 0])
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
plt.title("Exam Score Normalised")
plt.show()
############################################################
## Plot Sigmoid
Y_AXIS = []
X_AXIS = []
for x in range (-50,50,1):
    Y_AXIS.append(Sigmoid(x/10))
    X_AXIS.append(x/10)

plt.scatter(X_AXIS,Y_AXIS)
############################################################

df_2 = pd.DataFrame(X_matrix[:,1:3])
df = pd.concat([df_2,y], axis=1)
df.columns = ['P1', 'P2', 'T']
df
X_pred = np.array([1,10,20,1,20,30])
X_pred = X_pred.reshape(2,3)
theta = np.array([0.5,0.6,0.7])
theta = theta.reshape(3,1)
Sigmoid(np.dot(X_pred,theta))
############################################################

df.head(5)
df = df.sample(frac=1)
df[["T"]].values

df[["P1", "P2"]]

#turn dataframe into nparray
y_vector = df[["T"]].values
X_matrix = df[["P1", "P2"]].values

#number of training examples
m = len(y_vector)

#################################################################
#### Poly

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

poly = PolynomialFeatures(degree=2)
X_matrix = poly.fit_transform(X_matrix)
type(X_matrix)
#Normalisation
X_matrix, X_mean, X_std = Normalisation(X_matrix)
X_matrix[:,0] = 1

################################################################

#Add the intercept term to the X matrix
ones = np.ones((m,1))
X_matrix = np.hstack((ones,X_matrix))

theta = np.zeros([X_matrix.shape[1],1])

#to make sure the matrix are the right size
print(X_matrix.shape)
print(y_vector.shape)
print(theta.shape)

print(theta)
print(X_matrix)

#Split the data

X_train, X_test, y_train, y_test = train_test_split(
        X_matrix, y_vector, test_size = 0.2)

print(X_train.shape)

#Initial cost
J = CostFunction(theta, X_matrix, y_vector)
print(J)

#Train Error Only 
theta = np.zeros([X_matrix.shape[1],1])
theta, J_Hist_train = Finding_Theta(theta=theta, X=X_matrix, y=y_vector, alpha=0.5, 
                                    iterations=200)


# Test and Train Data
theta = np.zeros([X_train.shape[1],1])
theta, J_Hist_train, J_Hist_test = Finding_Theta_Test_Train(theta=theta, 
    X=X_train, y=y_train, alpha=0.5, iterations=500, X_test=X_test, y_test= y_test)

    
#Plot error
test = plt.plot(J_Hist_test, color = "Red", label="Test Cost")
train = plt.plot(J_Hist_train, color = "Blue", label="Train Cost")
plt.title("Train/Test Split 80/20")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend()
plt.show()


J = CostFunction(theta_optimized, X_matrix, y_vector)
print(J)
type(theta_optimized)
theta.shape

#how accurate is our model?
pred = [Sigmoid(np.dot(X_matrix,theta_optimized)) >= 0.5]
np.mean(pred == y_vector.flatten())*100


 ###########################################
 ## Decision Boundry
 ###########################################

theta_optimized.shape
 
w = theta_optimized
a = -w[1] / w[2]
xx = np.linspace(20, 60)
yy = a * xx - (w[0]) / w[2]

plt.figure(1,figsize=(8,8))
adm = plt.scatter(df["P1"].loc[df['T'] == 1], df["P2"].loc[df['T'] == 1])
not_adm =  plt.scatter(df["P1"].loc[df['T'] == 0], df["P2"].loc[df['T'] == 0])
plt.plot(xx, yy, 'k-')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
plt.title("Train/Test = 70/30")
plt.show()
