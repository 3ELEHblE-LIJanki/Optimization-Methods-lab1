from typing import Callable, List

import numpy as np

from lrs import LRS, gradient


class GradientDecent:
    """
        Класс реализующий поиск максимума и минимума на основе градиентного спуска и переданных параметров
    """

    ACCEPTABLE_ACCURACY: float = 0.00001


    def __init__(self, learning_rate_scheduling: LRS, f: Callable[[List[float]], float], bounds: List[List[float]],
                 eps: float):
        """
            learning_rate_scheduling - выбранная модель поиска шага
            max_iterations - максимальное число итераций (чтобы не зациклиться)
        """
        self.learning_rate_scheduling = learning_rate_scheduling
        self.f = f
        self.bounds = bounds
        self.eps = eps

    def __init(self, start: List[float]):
        self.x = start.copy()
        self.path = []

    def find_min(self, start: List[float], max_iterations: int) -> float:
        """
            start: List[float] - стартовая точка, в которой начнём поиск
            max_iterations: int - максимальное количество итераций спуска
            return - минимум полученный в ходе спуска
        """
        self.__init(start)
        for i in range(max_iterations):
            h = self.learning_rate_scheduling(self.x, i, self.f)
            self.path.append(self.x)
            grad = gradient(self.f, self.x, self.eps)
            xx = []
            for j in range(len(self.x)):
                coord = self.x[j] - h * grad[j]
                coord = max(coord, self.bounds[j][0])
                coord = min(coord, self.bounds[j][1])
                xx.append(coord)
            if np.linalg.norm(np.array(self.x) - np.array(xx)) < self.eps:
                break
            self.x = xx
            # print(self.current_point())
        return self.f(self.x)

    def find_max(self, start: List[float], max_iterations: int) -> float:
        """
            max_iterations: int - максимальное количество итераций спуска
            return - максимум полученный в ходе спуска
        """
        self.__init(start)
        for i in range(max_iterations):
            h = self.learning_rate_scheduling(self.x, i, self.f)
            self.path.append(self.x)
            grad = gradient(self.f, self.x, self.eps)
            xx = []
            for j in range(len(self.x)):
                coord = self.x[j] + h * grad[j]
                coord = max(coord, self.bounds[j][0])
                coord = min(coord, self.bounds[j][1])
                xx.append(coord)
            if np.linalg.norm(np.array(self.x) - np.array(xx)) < self.eps:
                break
            self.x = xx
        return self.f(self.x)

    def get_bounds(self):
        return self.bounds

    def get_f(self):
        return self.f

    def get_path(self):
        return self.path

    def current_point(self):
        return [self.x, self.f(self.x)]
