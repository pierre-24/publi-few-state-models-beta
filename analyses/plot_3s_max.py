from few_state import CT_dipole_3s
from few_state.properties import beta_HRS
from few_state.system_maker import make_system_from_mtT
from scipy.optimize import fmin
import matplotlib.pyplot as plt
import argparse
import numpy


def mkf(xi: float, mu_CT: float = 1):

    def f(m_CT, theta ) -> float:
        system = make_system_from_mtT(m_CT, 1, xi, CT_dipole_3s(mu_CT, numpy.deg2rad(theta)))
        tensor = system.response_tensor((1, 1), frequency=0)
        return beta_HRS(tensor)

    return f


parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='Data_exci.pdf')
parser.add_argument('-t', default=1, type=float)
parser.add_argument('-T', default=.5, type=float)
parser.add_argument('--mu_CT', default=1, type=float)
parser.add_argument('-N', default=200, type=int)

args = parser.parse_args()

X = 10**(numpy.linspace(0, -4, args.N))
Y = numpy.zeros((3, args.N))

for i, xi in enumerate(X):
    f = mkf(xi, args.mu_CT)
    mx = fmin(lambda x: -f(*x), (0, 54.7), xtol=1e-7, maxiter=100, full_output=True)
    Y[:, i] = *mx[0], -mx[1]
    print(xi, Y[:, i])


figure = plt.figure(figsize=(4, 8))
axes = figure.subplots(3)

axes[0].plot(Y[0], Y[2])

XM = numpy.linspace(-1, 1, args.N)
YM = numpy.zeros((5, args.N))

for i, xi in enumerate([1, 1e-1, 1e-2, 1e-3, 1e-4]):
    ix = numpy.argmin(numpy.abs(X - xi))
    theta = Y[1, ix]
    f = mkf(xi, args.mu_CT)
    for j, m_CT in enumerate(XM):
        YM[i, j] = f(m_CT, theta)

    axes[0].plot(XM, YM[i], '--', color='gray', label='{}'.format(xi), lw=0.7)

    if xi > 1e-4:
        axes[0].annotate(
            '$\\xi=10^{{{:.0f}}}$'.format(numpy.log10(xi)),
            (Y[0, ix], Y[2, ix]),
            xytext=(-15, 10),
            textcoords='offset points',
            va='center',
            ha='right',
            arrowprops=dict(arrowstyle="->")
        )

axes[0].set_xlim(-1, 1)
axes[0].set_ylim(0, .27)
axes[0].set_xlabel('$m_{CT}$')
axes[0].set_ylabel('$\\beta_{SHS}\\times\\mu_{CT}^{-3}\\times t^2$')

axes[1].semilogx(X, Y[2])

axes[2].plot(Y[0], Y[1])

for anot_x in [0, -1, -2, -3]:
    ix = numpy.argmin(numpy.abs(X - 10**anot_x))
    axes[2].annotate(
        '$\\xi=10^{{{}}}$'.format(anot_x),
        (Y[0, ix], Y[1, ix]),
        xytext=(-15, -10),
        textcoords='offset points',
        va='center',
        ha='right',
        arrowprops=dict(arrowstyle="->")
    )

axes[1].set_xlabel('$\\xi = T / t$')
axes[1].set_ylabel('$\\beta_{SHS}\\times\\mu_{CT}^{-3}\\times t^2$')

axes[2].set_xlabel('$m_{CT}$')
axes[2].set_ylabel('$\\theta$ (°)')
axes[2].set_ylim(54.6, 55.4)
axes[2].set_xlim(-.2, 1)

plt.tight_layout()
figure.savefig(args.output)
