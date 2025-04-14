from few_state import CT_dipole_3s, CT_dipole_4s, CT_dipole_5s
from few_state.system_maker import make_system_from_mtT
from few_state.properties import beta_HRS
import matplotlib.pyplot as plt
import argparse
import numpy

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='Data_exci.pdf')
parser.add_argument('-t', default=1)
parser.add_argument('-T', default=.5)
parser.add_argument('--mu_CT', default=1)
parser.add_argument('-N', default=200)

args = parser.parse_args()

figure = plt.figure(figsize=(5, 4))
ax = figure.subplots()

X = numpy.linspace(0.5, 1, args.N)
Y = numpy.zeros((6, X.shape[0]))

for i, m_CT in enumerate(X):
    xi = args.T / args.t
    system_3s = make_system_from_mtT(m_CT, args.t, args.T, CT_dipole_3s(args.mu_CT, numpy.deg2rad(90)))

    t_3s = system_3s.response_tensor((1, 1), frequency=0)
    Y[0, i] = beta_HRS(t_3s)

    system_4s = make_system_from_mtT(m_CT, args.t, args.T, CT_dipole_4s(args.mu_CT, numpy.deg2rad(90)))

    t_4s = system_4s.response_tensor((1, 1), frequency=0)
    Y[2, i] = beta_HRS(t_4s)
    Y[3, i] = args.mu_CT**3/args.t**2 * numpy.sqrt(42)/(63*xi**2)*(1-2/(3*args.T)*numpy.sqrt(1.5*(1-m_CT)))

    system_5s = make_system_from_mtT(m_CT, args.t, args.T, CT_dipole_5s(args.mu_CT))

    t_5s = system_5s.response_tensor((1, 1), frequency=0)
    Y[4, i] = beta_HRS(t_5s)
    Y[5, i] = args.mu_CT**3/args.t**2 * numpy.sqrt(21)/(84*xi**2)*(1-1/args.T*numpy.sqrt(0.5*(1-m_CT)))


ax.plot(X, Y[2], color='C2', label='$D_{3h}$ (4-state, $\\theta=90$°)')
ax.plot(X, Y[3], '--', color='C2')
ax.plot(X, Y[4], color='C3', label='$T_d$ (5-state)')
ax.plot(X, Y[5], '--', color='C3')

ax.legend()
ax.set_xlim(0.5, 1)

ax.set_ylim(0, numpy.sqrt(42)/(63*args.T**2))
ax.set_xlabel('$m_{CT}$')
ax.set_ylabel('$\\beta_{SHS}\\times \\mu_{CT}^{-3}\\times t^2$')

plt.tight_layout()
figure.savefig(args.output)
