from math import exp, fabs
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

"""
Краткое описание программы:
Для начала автоматически создаются исходные данные по произвольной формуле fx - функции от одной переменной
Затем по этим данным строятся 3 регрессионные модели: линейная, полиномиальная, экспотенциальная
Для разных функций они имею разную эффективность, поэтому нужно определить среднее отклонение наших приблежений
от оригинальной функции и выбрать модель с минимальным отклонением, которая и будет лучшим приблежением
Также красивые цветные визуализации прилагаются
"""

n = 10000 # Количество точек
x = []
y = []

for i in range(n):      # Заполняем х и у
    try:                # Проверка на деление на 0
        fx = i + 2      # Наша формула, для генерации у
        x.append(i)
        y.append(fx)
    except:
        pass

# Делаем три вида регрессии

X = np.array(x).reshape(-1,1)
Y = np.array(y)
model1 = LinearRegression().fit(X, Y)
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
model2 = LinearRegression().fit(x_poly,Y)
index_0 = []
i = 0
y_0 = y
x_0 = x
for y0 in y:    # Устраняем деление на 0 для экспотенциальной регресии
    if y0 == 0:
        x_0.pop(i)
        y_0.pop(i)
        i-=1
    i+=1
expo = np.polyfit(x_0, np.log(y_0), 1, w=np.sqrt(y_0))
y_pred3 = []    # Предсказываем у по экспотенциальной регрессии
for x1 in x:
    y_pred3.append(exp(expo[1]) * exp(expo[0] * x1))
y_pred1 = model1.predict(X).tolist()        # Предсказываем у по линейной регрессии
y_pred2 = model2.predict(x_poly).tolist()   # Предсказываем у по полиноальной регрессии

# Оцениевам эффективность приблежения по разнице между оригинальным у и предсказанным

dists1 = []
dists2 = []
dists3 = []
for j in range(len(x)):
    dists1.append(fabs(y_pred1[j] - y[j]))
    dists2.append(fabs(y_pred2[j] - y[j]))
    dists3.append(fabs(y_pred3[j] - y[j]))
dists = [np.std(dists1), np.std(dists2), np.std(dists3)]   # Находим среднее отклонение от оригинала
min_dist = float('inf')
i = 0
for dist in dists:
    if dist < min_dist:
        min_dist = dist
for dist in dists:
    if dist > min_dist - 0.1 and dist < min_dist + 0.1:   # Достаточно хорошим является приблежение с погрешностью 0.01
                                                            # относительно самого точного приблежения
        if i == 0:
            print('Линейное приближение достаточно хорошо')
        if i == 1:
            print('Полиномиальное приближение достаточно хорошо')
        if i == 2:
            print('Экспотенциальное приближение достаточно хорошо')
    i+=1

# Рисуем красивые графики всех этих приблежений и исходный график

size = 4
trans = 1

for i in range(len(x)):
    scatter1 = plt.scatter(x[i], y[i], c='blue', s = size, alpha = trans)
    scatter1 = plt.scatter(x[i], y_pred1[i], c = 'red', s = size, alpha = trans)
    scatter1 = plt.scatter(x[i], y_pred2[i], c = 'green', s = size, alpha = trans)
    scatter1 = plt.scatter(x[i], y_pred3[i], c = 'yellow', s = size, alpha = trans)
plt.show()
