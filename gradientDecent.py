from typing import Callable
from lrs import LRS

'''
    Класс реализующий поиск максимума и минимума на основе градиентного спуска и переданных параметров
'''
class GradientDecent:

    '''
        learningRateSheduling - выбранная модель поиска шага
        maxIteartions - максимальное число итераций (чтобы не зациклиться)
    '''
    def __init__(self, learningRateSheduling: LRS, maxIterations: int):
        self.learningRateSheduling = learningRateSheduling
        self.maxIterations = maxIterations

    '''
        f - исследуемая функция

        return - всё что необходимо, допишем когда графики будем рисовать
    '''
    def findMin(self, f: Callable[[tuple], float]):
        ...

    '''
        f - исследуемая функция

        return - всё что необходимо, допишем когда графики будем рисовать
    '''
    def findMax(self, f: Callable[[tuple], float]):
        ...

    pass