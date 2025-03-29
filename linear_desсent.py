from math import sqrt
from typing import Callable, List


# работает только для функций 1d
# Расчитываю что он будет рисовать границы

class LinearDecent:
    invphi : float = (-1 + sqrt(5)) / 2 # 1/phi

    def __init__(self, f: Callable[[List[float]], float], bounds: List[List[float]], eps):
        self.f = f
        self.eps = eps
        self.bounds = bounds
        self.path = []
        self.x = -100

    def __init(self,  start: float):
        if isinstance(start, list):
            if len(start) > 1:
                raise TypeError("Only for R -> R functions")
            self.path.append(start)
            start = start[0]
        else:
            self.path.append([start])
        self.x : float = start

    def find_min(self, start: float | list[float], max_steps_count: int):
        self.__init(start)
        l = self.bounds[0][0]
        r = self.bounds[0][1]
        count_iter = 2
        while abs(l - r) > self.eps and count_iter < max_steps_count:
            x1 = r -  (r - l) * self.invphi
            x2 = l + (r - l) * self.invphi
            # print("__", x1, x2, self.f([x1]), self.f([x2]))
            if self.f([x1]) < self.f([x2]):
                r = x2
                self.path.append([r])
            else:
                l = x1
                self.path.append([x1])
            count_iter += 1
            # print(l, r)
        self.x = min(r, l)
        self.path.append([self.x])
        return self.f([self.x])

    def find_max(self, start: float | list[float], max_steps_count: int):
        self.__init(start)
        l = self.bounds[0][0]
        r = self.bounds[1][0]
        count_iter = 0
        while abs(l - r) > self.eps and count_iter < max_steps_count:
            x1 = l + (r - l) * self.invphi
            x2 = r - (r - l) * self.invphi
            if self.f([x1]) > self.f([x2]):
                r = x2
                self.path.append([r])
            else:
                l = x1
                self.path.append([x1])
            count_iter += 1
        self.x = max(r , l)
        self.path.append([self.x])
        return self.f([self.x])


    def get_bounds(self):
        return self.bounds

    def get_f(self):
        return self.f

    def get_path(self):
        return self.path


