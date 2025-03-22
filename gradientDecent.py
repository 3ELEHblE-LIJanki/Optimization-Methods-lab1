from typing import Callable
from lrs import LRS

'''
    Класс реализующий поиск максимума и минимума на основе градиентного спуска и переданных параметров
'''
class GradientDecent:

    '''
        learning_rate_scheduling - выбранная модель поиска шага
        max_iterations - максимальное число итераций (чтобы не зациклиться)
    '''

    def __init__(self, learning_rate_scheduling: LRS, max_iterations: int):
        self.learning_rate_scheduling = learning_rate_scheduling
        self.max_iterations = max_iterations

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