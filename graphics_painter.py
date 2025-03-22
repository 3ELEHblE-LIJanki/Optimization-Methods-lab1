from gradient_decent import GradientDecent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
    Класс рисующий необходимые для исследования графики
'''
class GraphicsPainter:
    """
        descent - градиентный спуск, для которого будут производиться исследования
    """
    def __init__(self, descent: GradientDecent):
        self.descent = descent

    '''
        Метод рисующий график уровней
    '''
    def plot_levels(self):
        x_grid, y_grid = np.meshgrid(*self.descent.get_bounds())
        f_grid = self.descent.get_f()(x_grid, y_grid)
        palette = sns.color_palette("flare", as_cmap=True)
        contourf = plt.contourf(x_grid, y_grid, f_grid, levels=50, cmap=palette, alpha=0.9)
        contours = plt.contour(x_grid, y_grid, f_grid, levels=15,
                               colors=sns.color_palette("dark:white", n_colors=15),
                               linewidths=1, alpha=0.95)
        plt.clabel(contours, inline=True, fontsize=12, fmt='%.1f',
                   colors='black', inline_spacing=8)
        cbar = plt.colorbar(contourf, pad=0.05, fraction=0.046, aspect=25)
        cbar.set_label('Значение функции', fontsize=14, labelpad=15)
        cbar.set_ticks(np.linspace(f_grid.min(), f_grid.max(), 8))
        cbar.ax.tick_params(labelsize=12)

    '''
        Метод рисующий ещё график траектории
    '''
    def plot_trajectory(self):
        path = np.array(self.descent.get_path())
        plt.plot(path[:, 0], path[:, 1],
                 color=sns.color_palette("rocket")[2],
                 linewidth=3, alpha=0.9,
                 label='Путь градиентного спуска',
                 zorder=3)
        sns.scatterplot(x=path[0:1, 0], y=path[0:1, 1],
                        color='lime', s=250,
                        edgecolor='black', linewidth=1.5,
                        label='Старт', zorder=5)
        sns.scatterplot(x=path[-1:, 0], y=path[-1:, 1],
                        color='red', s=250,
                        edgecolor='black', linewidth=1.5,
                        label='Минимум', zorder=5)

    def plot(self):
        sns.set_theme(style="white", context="talk", palette="deep")
        sns.set_style("ticks", {"axes.grid": True, "grid.linestyle": "--", "grid.color": "0.85"})
        plt.rcParams['figure.facecolor'] = '#f5f5f5'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.figure(figsize=(12, 9), dpi=100)
        self.plot_levels()
        self.plot_trajectory()
        sns.despine(left=False, bottom=False)
        plt.xlabel('x', fontsize=16, labelpad=15)
        plt.ylabel('y', fontsize=16, labelpad=15)
        plt.title('Оптимизация функции методом градиентного спуска',
                  fontsize=20, pad=25, fontweight='bold')
        plt.legend(fontsize=14, loc='upper right',
                   frameon=True, edgecolor='black',
                   facecolor='white', framealpha=0.95,
                   bbox_to_anchor=(1.0, 1.0))

        plt.tight_layout()
        plt.show()