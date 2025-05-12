"""
---Laboratory work #1---
"We decide to go on a hike or not"
1. Packed a bag
2. It's raining
3. It's cold
4. Gathered a group
5. Good health
With the introduction of the hyperbolic tangent function
Completed by Sorokoumov A.P.
"""
import numpy as np
import math

train_X = np.array([[1,1,1,1,1],
                    [1,0,0,1,1],
                    [0,1,0,0,0],
                    [0,1,0,1,1],
                    [0,1,1,1,0],
                    [1,1,1,0,1],
                    [1,0,1,0,0],
                    [1,0,0,1,1],
                    [1,0,1,1,1],
                    [0,0,1,0,0]])

train_Y = np.array([[1],
                    [1],
                    [-1],
                    [1],
                    [-1],
                    [1],
                    [-1],
                    [1],
                    [1],
                    [-1]])

def sigmoid(x):     
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))  

def predict(inputs, weights):     
    weighted_sum = np.dot(inputs, weights)     
    return sigmoid(weighted_sum)

weights = 2 * np.random.random((5, 1)) - 1
print('Случайные веса')
print(weights)

for i in range(100):         
    prediction = predict(train_X, weights)           
    error = train_Y - prediction       
    delta = error * (1 - pow(prediction, 2))
    weights += np.dot(train_X.T, delta)

print('Веса после обучения:') 
print(weights)   
print('Результат:') 
print(prediction)


