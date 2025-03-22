from typing import Callable
from lrs import LRS


class GradientDecent:
    """
        Класс реализующий поиск максимума и минимума на основе градиентного спуска и переданных параметров
    """

    def __init__(self, learning_rate_scheduling: LRS, max_iterations: int):
        """
            learning_rate_scheduling - выбранная модель поиска шага
            max_iterations - максимальное число итераций (чтобы не зациклиться)
        """
        self.learning_rate_scheduling = learning_rate_scheduling
        self.max_iterations = max_iterations

    def find_min(self, f: Callable[[tuple], float]):
        """
            f - исследуемая функция
            return - всё что необходимо, допишем когда графики будем рисовать
        """
        ...

    def find_max(self, f: Callable[[tuple], float]):
        """
            f - исследуемая функция
            return - всё что необходимо, допишем когда графики будем рисовать
        """
        ...
