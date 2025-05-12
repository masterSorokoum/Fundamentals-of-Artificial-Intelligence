"""
---Laboratory work #1---
"We decide to go on a hike or not"
1. Packed a bag
2. It's raining
3. It's cold
4. Gathered a group
5. Good health
With the introduction of new data
Completed by Sorokoumov A.P.
"""

import numpy as np

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
                    [0],
                    [1],
                    [0],
                    [1],
                    [0],
                    [1],
                    [1],
                    [0]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def predict(inputs, weights):
    weighted_sum = np.dot(inputs, weights)
    return sigmoid(weighted_sum)

weights = 2 * np.random.random((5, 1)) - 1

for i in range(10):
    # вычисление предсказаний по входным данным и весам
    prediction = predict(train_X, weights)
    # ошибки - разности желаемых сигналов и действительных
    error = train_Y - prediction
    # коррекция ошибки
    # значение ошибки умножается на производную функции активации
    # производная: f’(x) = f(x) * (1 – f(x))
    delta = error * (prediction * (1 - prediction))
    # обновление весов
    weights += np.dot(train_X.T, delta)

print('Веса после обучения:')
print(weights)
print('Результат:')
print(prediction)


new_input = np.array([0,1,0,0,1]) 
prediction = predict(new_input, weights)  
print('Новый результат:') 
print(prediction)
