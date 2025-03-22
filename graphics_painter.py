from gradient_decent import GradientDecent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class GraphicsPainter:
    """
    Класс для отрисовки графиков градиентного спуска для одномерных и двумерных функций.

    Args:
        descent (GradientDecent): Экземпляр класса градиентного спуска.
    """

    def __init__(self, descent: GradientDecent):
        self.descent = descent
        bounds = self.descent.get_bounds()
        self.is_1d = len(bounds) == 1

    def plot_levels(self):
        """
        Рисует график уровней функции (контуры для 2D или линию для 1D).
        """
        bounds = [np.arange(start, end, 0.01) for start, end in self.descent.get_bounds()]

        if self.is_1d:
            x = bounds[0]
            f_values = self.descent.get_f()([x])
            _ = plt.plot(x, f_values, color=sns.color_palette("flare")[2],
                           linewidth=2, alpha=0.9, label='Функция')[0]
        else:
            grid = np.meshgrid(*bounds)
            f_grid = self.descent.get_f()(grid)
            palette = sns.color_palette("flare", as_cmap=True)
            contourf = plt.contourf(*grid, f_grid, levels=50, cmap=palette, alpha=0.9)
            contours = plt.contour(*grid, f_grid, levels=15,
                                   colors=sns.color_palette("dark:white", n_colors=15),
                                   linewidths=1, alpha=0.95)
            plt.clabel(contours, inline=True, fontsize=12, fmt='%.1f',
                       colors='black', inline_spacing=8)
            cbar = plt.colorbar(contourf, pad=0.05, fraction=0.046, aspect=25)
            cbar.set_label('Значение функции', fontsize=14, labelpad=15)
            cbar.set_ticks(np.linspace(f_grid.min(), f_grid.max(), 8))
            cbar.ax.tick_params(labelsize=12)

    def plot_trajectory(self):
        """
        Рисует траекторию градиентного спуска.
        """
        path = np.array(self.descent.get_path())
        if self.is_1d:
            x_path = path
            y_path = self.descent.get_f()([x_path])
            plt.plot(x_path, y_path,
                     color=sns.color_palette("rocket")[2],
                     linewidth=3, alpha=0.9,
                     label='Путь градиентного спуска',
                     zorder=3)
            sns.scatterplot(x=x_path[0:1], y=y_path[0:1],
                            color='lime', s=250,
                            edgecolor='black', linewidth=1.5,
                            label='Старт', zorder=5)
            sns.scatterplot(x=x_path[-1:], y=y_path[-1:],
                            color='red', s=250,
                            edgecolor='black', linewidth=1.5,
                            label='Минимум', zorder=5)
        else:
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
        """
        Основной метод для построения полного графика.
        """
        sns.set_theme(style="white", context="talk", palette="deep")
        sns.set_style("ticks", {"axes.grid": True, "grid.linestyle": "--", "grid.color": "0.85"})
        plt.rcParams['figure.facecolor'] = '#f5f5f5'
        plt.rcParams['axes.facecolor'] = 'white'

        plt.figure(figsize=(12, 9), dpi=100)

        self.plot_levels()
        self.plot_trajectory()

        sns.despine(left=False, bottom=False)
        plt.xlabel('x', fontsize=16, labelpad=15)
        plt.ylabel('f(x)' if self.is_1d else 'y', fontsize=16, labelpad=15)
        plt.title('Оптимизация функции методом градиентного спуска',
                  fontsize=20, pad=25, fontweight='bold')

        plt.legend(fontsize=14, loc='upper right',
                   frameon=True, edgecolor='black',
                   facecolor='white', framealpha=0.95,
                   bbox_to_anchor=(1.0, 1.0))

        plt.tight_layout()
        plt.show()
