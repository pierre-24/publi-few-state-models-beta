import matplotlib.axes
import numpy

from typing import Callable

from few_state import CT_dipole_2s, CT_dipole_3s, CT_dipole_4s, CT_dipole_5s
from few_state.system_maker import make_system_from_mtT
from few_state.system import System


class Walk:
    """A collection of paths, each of them being a collection of (coordinate, system)"""
    def __init__(self, paths: list[list[tuple[float, System]]]):
        self.paths = paths

    @classmethod
    def walk_2s_to_5s_mtT(cls, m_CT: float, t: float, T: float, N: int = 20, spl: float=.5):
        Ng = int(N * spl)
        Nt = 2 * N - Ng

        paths = []

        # 0 → C_oov
        current_path = []
        for i in range(0, Ng + 1):
            dips = CT_dipole_2s(1.)
            dips[-1] = i / Ng * dips[-1]
            current_path.append((i / N, make_system_from_mtT(m_CT, t, T, dips)))

        paths.append(current_path)

        # C_oov → D_ooh
        current_path = []
        for i in range(0, Ng):
            dips = CT_dipole_3s(1., numpy.deg2rad(90))
            dips[-1] = i / Ng * dips[-1]
            current_path.append((spl + i / N, make_system_from_mtT(m_CT, t, T, dips)))

        # D_ooh → C_2v
        for i in range(Nt + 1):
            dips = CT_dipole_3s(1., numpy.deg2rad(90 - i * 90 / Nt))
            current_path.append((spl + (Ng + i) / N, make_system_from_mtT(m_CT, t, T, dips)))

        paths.append(current_path)

        # C_2v → D_3h
        current_path = []
        for i in range(Ng):
            dips = CT_dipole_4s(1., numpy.deg2rad(90))
            dips[0] = i / Ng * dips[0]
            current_path.append((spl + 2 + i / N, make_system_from_mtT(m_CT, t, T, dips)))

        # D_ooh → C_3v
        for i in range(Nt + 1):
            dips = CT_dipole_4s(1., numpy.deg2rad(90 - i * 90 / Nt))
            current_path.append((spl + 2 + (Ng + i) / N, make_system_from_mtT(m_CT, t, T, dips)))

        paths.append(current_path)

        # C_3v → T_d
        current_path = []
        for i in range(Ng + 1):
            dips = CT_dipole_5s(1.)
            dips[-1] = i / Ng * dips[-1]
            current_path.append((spl + 4 + i / N, make_system_from_mtT(m_CT, t, T, dips)))

        paths.append(current_path)


        return cls(paths)

    def plot_property(self, ax: matplotlib.axes.Axes, tr: Callable[[System], float], color: str, label: str = None):
        for j, path in enumerate(self.paths):
            Y = numpy.array([tr(system) for x, system in path])
            ax.plot([x for x, system in path], Y, color=color, label=label if j == 0 else None)


def plot_walks(ax: matplotlib.axes.Axes, walks: list[Walk], tr: Callable[[System], float], labels: list[str], ylim: tuple[float, float], ylabel: str, color_shift: int = 0):
    ax.vlines([0, .5, 2.5, 4.5], ylim[0], ylim[1], color='grey')
    ax.vlines([1., 1 + 1.5 / 3, 3, 3 + 1.5 * 2 / 9, 5.], ylim[0], ylim[1], linestyle='--', color='grey')

    for i, (walk, label) in enumerate(zip(walks, labels if labels is not None else [None] * len(walks))):
        walk.plot_property(ax, tr, color='C{}'.format(color_shift + i), label=label)

    if labels is not None:
        ax.legend()

    ax.set_ylabel(ylabel)
    ax.set_xlim(-.1, 5.1)
    ax.set_ylim(*ylim)

    ax.set_xticks(
        [0, .5, 1, 2.5, 3, 4.5, 5.],
        ['0', '$C_{\\infty v}$|$C_{\\infty v}$', '$D_{\\infty h}$', '$C_{\\infty v}$|$C_{2v}$', '$D_{3h}$', '$C_{\\infty v}$|$C_{3v}$', '$T_d$'])

    secax = ax.secondary_xaxis('top', functions=(lambda x_: x_, lambda x_: x_))
    secax.set_xticks(
        [0, .5, 1, 1 + 1.5 / 3, 2.5, 3, 3 + 1.5 * 2 / 9, 4.5, 5.],
        ['2$s^\\star$', '2$s$|3$s^\\star_1$', '3$s_1$', '3$s_{2/3}$', '3$s_0$|4$s^\\star_1$', '4$s_1$', '4$s_{7/9}$', '4$s_0$|5$s^\\star$', '5$s$'])
