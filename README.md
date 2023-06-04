# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Predict the values of array.

5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

6.Obtain the graph.
```

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Karthikeyan R
RegisterNumber: 212222240045
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data2.txt",delimiter = ',')
x= data[:,[0,1]]
y= data[:,2]
print('Array Value of x:')
x[:5]

print('Array Value of y:')
y[:5]

print('Exam 1-Score graph')
plt.figure()
plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label=' Not Admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()


def sigmoid(z):
  return 1/(1+np.exp(-z))
  
print('Sigmoid function graph: ')
plt.plot()
x_plot = np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot))
plt.show()


def costFunction(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad = np.dot(x.T,h-y)/x.shape[0]
  return j,grad


print('X_train_grad_value: ')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
j,grad = costFunction(theta,x_train,y)
print(j)
print(grad)


print('y_train_grad_value: ')
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

def cost(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return j


def cost(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return j

print('res.x:')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)


def plotDecisionBoundary(theta,x,y):
  x_min,x_max = x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max = x[:,1].min()-1,x[:,1].max()+1
  xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot = np.c_[xx.ravel(),yy.ravel()]
  x_plot = np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot = np.dot(x_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
  plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label='Not Admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel('Exam  1 score')
  plt.ylabel('Exam 2 score')
  plt.legend()
  plt.show()

print('DecisionBoundary-graph for exam score: ')
plotDecisionBoundary(res.x,x,y)

print('Proability value: ')
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)


def predict(theta,x):
  x_train = np.hstack((np.ones((x.shape[0],1)),x))
  prob = sigmoid(np.dot(x_train,theta))
  return (prob >=0.5).astype(int)


print('Prediction value of mean:')
np.mean(predict(res.x,x)==y)
```
## Output:
![ml5](https://github.com/karthikeyan-R16/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119421232/6e8fa5b3-1f81-48d1-b3f4-cabcb38b5e49)

![ml5 1](https://github.com/karthikeyan-R16/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119421232/9eca192f-707b-4a48-ae07-ee40947bb9a7)

![ml5 2](https://github.com/karthikeyan-R16/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119421232/230bf0a6-dcd1-44d6-a24a-f4ab17d9b069)

![ml5 3](https://github.com/karthikeyan-R16/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119421232/22c580de-22c6-4b38-9b17-7049dcb1c47c)

![ml5 4](https://github.com/karthikeyan-R16/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119421232/ffcf63a9-1941-419a-9662-4ae7396a0386)

![ml5 5](https://github.com/karthikeyan-R16/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119421232/f2655fc4-ce99-438c-b419-b8d92e5084e0)

![ml5 6](https://github.com/karthikeyan-R16/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119421232/f3a19681-1476-490b-9c3e-b3bb751003dc)

![ml5 7](https://github.com/karthikeyan-R16/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119421232/33c6707c-f303-4860-9e4b-ce069743ec2e)

![ml5 8](https://github.com/karthikeyan-R16/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119421232/ef834a1d-d3a6-436f-be5a-e30ed061bb35)

![ml5 9](https://github.com/karthikeyan-R16/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119421232/e6f11621-5e07-4ce7-83bc-9071d98b8401)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

