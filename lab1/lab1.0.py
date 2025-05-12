import numpy as np

# входные сигналы, разное сочетание условий
train_X = np.array([[1,1,1,0,0],  
                    [1,0,0,1,0], 
                    [0,1,1,1,0],
                    [0,1,0,0,0],
                    [1,1,1,1,1],
                    [0,1,0,0,1],
                    [0,0,1,0,0],
                    [1,0,0,1,1],
                    [1,0,1,1,1],
                    [0,0,1,0,0]])

# праввильные ответы на каждое из условий
train_Y = np.array([[1], 
                    [1],
                    [0],
                    [0],
                    [1],
                    [0],
                    [0],
                    [1],
                    [1],
                    [0]])

# функция вычисляет значение взвешенной суммы и передает для предсказания
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

# функция вычисляет значение лог функции
def predict(inputs, weights):   
    weighted_sum = np.dot(inputs, weights)
    return sigmoid(weighted_sum)

weights = 2 * np.random.random((5, 1)) - 1
print('Случайные веса')
print(weights)

# обучение нейрона
for i in range(500):
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


