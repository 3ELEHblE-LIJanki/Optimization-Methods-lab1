from gradientDecent import GradientDecent

'''
    Класс рисующий необходимые для исследования графики
'''
class GraphicsPainter:

    '''
        descent - градиентный спуск, для которого будут производиться исследования
    '''
    def __init__(self, descent: GradientDecent):
        self.descent = descent

    '''
        Метод рисующий график уровней
    '''
    def paint_levels(self):
        print("heheheh")

    '''
        Метод рисующий ещё график траектории
    '''
    def paint_trajectory(self):
        print("ahahahah")
