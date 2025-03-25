from typing import Callable
import math

def diff(f, x, eps, index: int):
        x_i = x.copy()
        x_i[index] += eps
        return (f(x_i) - f(x)) / eps

def gradient(f, x, eps):
    grad = []
    for i in range(len(x)):
        grad.append(diff(f, x, eps, i))
    return grad

def mult(x_1, x_2):
    res = 0
    for i in range(len(x_1)):
        res += x_1[i] * x_2[i]
    return res

def addv(x_1, x_2):
    return [x_1[i] + x_2[i] for i in range(len(x_1))]

'''
    Тип для Learning rate scheduling
'''
LRS = Callable[[tuple, int, Callable[[tuple], float]], float]

'''
    Правило Армихо
    
    c1 - гиперпараметр (из конспекта, хз пока нужен ли)

    return - LRS (learning rate scheduling) по правилу Армихо с заданными гипер-параметрами 
'''
def armiho(c1: float, q: float) -> LRS:
    return lambda x, k, f: _arm(c1, q, x, k, f)

def _arm(c1, q, x, k, f):
    ff = gradient(f, x, 0.00001)
    fff = [-ff_i for ff_i in ff]
    l = lambda a: f(x) + c1 * a * mult(fff, ff)
    a = 500000
    while l(a) < f(addv([a * ff_i for ff_i in fff], x)):
        a = q * a
    return a

'''
    Правило Вольфе

    c1 - гиперпараметр (из конспекта, хз пока нужен ли)
    c2 - гиперпараметр (из конспекта, хз пока нужен ли)
    
    return - LRS (learning rate scheduling) по правилу Вольфе с заданными гипер-параметрами 
'''
def wolfe(c1: float, c2: float) -> LRS:
    ...

'''
    Правило Голдстейна
   
    c1 - гиперпараметр (из конспекта, хз пока нужен ли)
    c2 - гиперпараметр (из конспекта, хз пока нужен ли)

    return - LRS (learning rate scheduling) по правилу Голдстейна с заданными гипер-параметрами 
'''
def goldstein(c1: float, c2: float) -> LRS:
    ...

'''
    Постоянный метод планирования шага

    h0 - шаг

    return - Постоянный LRS (learning rate scheduling) с заданными гипер-параметрами 
'''
def constant(h0: float) -> LRS:
    return lambda x, k, f: h0

'''
    Функциональный метод планирования шага (Экспоненциальное затухание)

    h0 - начальный шаг
    l - степень затухания

    return - Функциональный LRS (learning rate scheduling) с заданными гипер-параметрами 
'''
def exponential_decay(h0: float, l: float) -> LRS:
    return lambda x, k, f: h0 * math.e**(-l*k)

'''
    Функциональный метод планирования шага (Полиномиальное затухание)
   
    a - гиперпараметр (из конспекта)
    b - гиперпараметр (из конспекта)

    return - Функциональный LRS (learning rate scheduling) с заданными гипер-параметрами 
'''
def polynomial_decay(a: float, b: float) -> LRS:
    return lambda x, k, f: (1.0 / math.sqrt(k + 1)) * (b * k + 1)**(-a)