from typing import Callable
import math

'''
    Тип для Learnig rate sheduling
'''
LRS = Callable[[tuple, int, Callable[[tuple], float]], float]

'''
    Правило Армихо
    
    c1 - гиперпараметр (из конспекта, хз пока нужен ли)

    return - LRS (learning rate sheduling) по правилу Армихо с заданными гипер-параметрами 
'''
def armiho(с1: float) -> LRS:
    ...

'''
    Правило Вольфе

    c1 - гиперпараметр (из конспекта, хз пока нужен ли)
    c2 - гиперпараметр (из конспекта, хз пока нужен ли)
    
    return - LRS (learning rate sheduling) по правилу Вольфе с заданными гипер-параметрами 
'''
def wolfe(c1: float, c2: float) -> LRS:
    ...

'''
    Правило Голдстейна
   
    c1 - гиперпараметр (из конспекта, хз пока нужен ли)
    c2 - гиперпараметр (из конспекта, хз пока нужен ли)

    return - LRS (learning rate sheduling) по правилу Голдстейна с заданными гипер-параметрами 
'''
def goldstein(c1: float, c2: float) -> LRS:
    ...

'''
    Постоянный метод планирования шага

    h0 - шаг

    return - Постоянный LRS (learning rate sheduling) с заданными гипер-параметрами 
'''
def constant(h0: float) -> LRS:
    return lambda x, k, f: h0;

'''
    Функциональный метод планирования шага (Экспоненциальное затухание)

    h0 - начальный шаг
    l - степень затухания

    return - Функциональный LRS (learning rate sheduling) с заданными гипер-параметрами 
'''
def exponentialDecay(h0: float, l: float) -> LRS:
    return lambda x, k, f: h0 * math.e**(-l*k)

'''
    Функциональный метод планирования шага (Полиномиальное затухание)
   
    a - гиперпараметр (из конспекта)
    b - гиперпараметр (из конспекта)

    return - Функциональный LRS (learning rate sheduling) с заданными гипер-параметрами 
'''
def polinomialDecay(a: float, b: float) -> LRS:
    return lambda x, k, f: (1.0 / math.sqrt(k + 1)) * (b * k + 1)**(-a)