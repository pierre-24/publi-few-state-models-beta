from few_state import CT_dipole_4s
from few_state.system_maker import make_system_from_mtT
from few_state.properties import beta_HRS
import matplotlib.pyplot as plt
import argparse
import numpy

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='Data_exci.pdf')
parser.add_argument('-t', default=1, type=float)
parser.add_argument('-T', default=.5, type=float)
parser.add_argument('--mu_CT', default=1, type=float)
parser.add_argument('--m_CT', default=1 - 1e-15, type=float)
parser.add_argument('-N', default=200, type=int)

args = parser.parse_args()

figure = plt.figure(figsize=(5, 4))
ax = figure.subplots()

X = numpy.linspace(0, 90, args.N)
Y = numpy.zeros((2, X.shape[0]))

for i, theta in enumerate(X):
    xi = args.T / args.t
    system_4s = make_system_from_mtT(args.m_CT, args.t, args.T, CT_dipole_4s(args.mu_CT, numpy.deg2rad(theta)))

    t_4s = system_4s.response_tensor((1, 1), frequency=0)
    Y[0, i] = beta_HRS(t_4s)
    Y[1, i] = args.mu_CT**3/args.t**2 * numpy.sqrt(42)/(63*xi**2)*(1-2/(3*args.T)*numpy.sqrt(1.5*(1-args.m_CT))) * numpy.sin(numpy.deg2rad(theta))**3


ax.plot(X, Y[0], color='C2', label='$C_{3v}$ (4-state)')
ax.plot(X, Y[1], '--', color='C2')

ax.legend()
ax.set_xlim(0, 90)

ax.set_ylim(0, numpy.sqrt(42)/(63*args.T**2))

ax.set_xlabel('$\\theta$ (°)')
ax.set_ylabel('$\\beta_{SHS}\\times \\mu_{CT}^{-3}\\times t^2$')

plt.tight_layout()
figure.savefig(args.output)
