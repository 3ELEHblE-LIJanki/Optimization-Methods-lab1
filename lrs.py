from typing import Callable, List
import math

import numpy as np

from graphics_plotter import GraphicsPlotter
from linear_desent import LinearDecent

EPS = 1e-5

def diff(f, x, eps, index: int):
    x_i = x.copy()
    x_i[index] += eps
    x_j = x.copy()
    x_j[index] -= eps
    return (f(x_i) - f(x_j)) / (2 * eps)

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
    ff = gradient(f, x, EPS)
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
def wolfe(c1: float = 1e-4, c2: float = 0.9) -> LRS:
    return lambda x, k, f: _wolfe(c1, c2, x, k, f)


def _wolfe(c1, c2, x, k, f):
    max_iterations = 100
    a = 1

    fx = f(x)
    grad = lambda x: np.array(gradient(f, x, EPS))
    grad_x_np = grad(x)

    # (1-Armijo) f(x + a * p) <= f(x) + c1 * a * graf(f)^T * p
    # (2-Curvature) grad(f(x + ap))^T * p >= c2 * grad(f(x))^T * p

    p = -grad_x_np
    right = grad_x_np.dot(p)  # grad(f(x))^T p - где p - прошлое направление - в моём случае -grad (??)

    for _ in range(max_iterations):
        x_new = np.array(x) + a * p
        fx_new = f(x_new.tolist())
        grad_new = grad(x_new)

        # Проверка 1
        if fx_new > fx + c1 * a * right:
            a *= 0.5
            continue

        # Проверка 2
        if grad_new.dot(p) < c2 * right:
            a *= 1.5
            continue
        return a
    return a


'''
    Правило Голдстейна
   
    c1 - гиперпараметр (из конспекта, хз пока нужен ли)
    c2 - гиперпараметр (из конспекта, хз пока нужен ли)

    return - LRS (learning rate scheduling) по правилу Голдстейна с заданными гипер-параметрами 
'''
def goldstein(c1: float = 0.1, c2: float = 0.9) -> LRS:
    return lambda x, k, f: _goldstein(c1, x, k, f)


def _goldstein(c1, x, k, f):
    max_iterations = 100
    a = 1

    fx = f(x)
    grad = lambda x: np.array(gradient(f, x, EPS))
    grad_x_np = grad(x)

    # (1) f(x + a * p) <= f(x) + c1 * a * graf(f)^T * p
    # (2) f(x + a * p) >= f(x) + (1 - c1) * a * graf(f)^T * p

    p = -grad_x_np
    right = grad_x_np.dot(p)  # grad(f(x))^T p - где p grad (??)

    for _ in range(max_iterations):
        x_new = np.array(x) + a * p
        fx_new = f(x_new.tolist())

        # Проверка 1
        if fx_new > fx + c1 * a * right:
            a *= 0.5
            continue

        grad_new = grad(x_new)
        # Проверка 2
        if fx_new < (1 - c1) * right:
            a *= 1.5
            continue
        return a
    return a


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

def linear_decent(bounds: List[List[float]], eps: float, max_steps_count: int, f_bounds: List[List[float]]) -> LRS:
    """
    bounds - границы поиска
    Функциональный метод планирования шага (Линейное золотое сечение)
    eps - точность линейного поиска
    max_steps_count - максимальное количество шагов линейного поиска
    f_bounds - границы функции
    """
    return lambda x, _, f: __linear_decent(x, f, bounds, eps, max_steps_count, f_bounds)

def __linear_decent(x, f, bounds, eps, max_steps_count, f_bounds):
    def ff(h):
        delta = x - h * np.array(gradient(f, x, EPS))
        for i in range(len(delta)):
            delta[i] = max(min(delta[i], f_bounds[i][1]), f_bounds[i][0])
        return f(delta)
    lin_dec = LinearDecent(ff, bounds, eps)
    lin_dec.find_min(bounds[0][0], max_steps_count)
    return lin_dec.get_path()[-1][0]
