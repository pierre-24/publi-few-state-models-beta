from few_state import CT_dipole_2s, CT_dipole_3s, CT_dipole_4s, CT_dipole_5s
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
parser.add_argument('--theta', default=60)
parser.add_argument('-N', default=200)

args = parser.parse_args()

figure = plt.figure(figsize=(5, 4))
ax = figure.subplots()

X = numpy.linspace(-1, 1, args.N)
Y = numpy.zeros((6, X.shape[0]))

def _mk(system, Yp: numpy.ndarray, a: int, i: int):
    tensor = system.response_tensor((1, 1), frequency=0)
    Yp[a, i] = beta_HRS(tensor)

for i, m_CT in enumerate(X):
    system_2s = make_system_from_mtT(m_CT, args.t, args.T, CT_dipole_2s(args.mu_CT))
    _mk(system_2s, Y, 0, i)

    system_3s = make_system_from_mtT(m_CT, args.t, args.T, CT_dipole_3s(args.mu_CT, numpy.deg2rad(args.theta)))
    _mk(system_3s, Y, 1, i)

    system_3s = make_system_from_mtT(m_CT, args.t, args.T, CT_dipole_3s(args.mu_CT, numpy.deg2rad(90)))
    _mk(system_3s, Y, 2, i)

    system_4s = make_system_from_mtT(m_CT, args.t, args.T, CT_dipole_4s(args.mu_CT, numpy.deg2rad(args.theta)))
    _mk(system_4s, Y, 3, i)

    system_4s = make_system_from_mtT(m_CT, args.t, args.T, CT_dipole_4s(args.mu_CT, numpy.deg2rad(90)))
    _mk(system_4s, Y, 4, i)

    system_5s = make_system_from_mtT(m_CT, args.t, args.T, CT_dipole_5s(args.mu_CT))
    _mk(system_5s, Y, 5, i)

ax.plot(X, Y[0], color='C0', label='$C_{\\infty v}$ (2-state)')
ax.plot(X, Y[1], color='C1', label='$C_{{2v}}$ (3-state, $\\theta={}$°)'.format(args.theta))
ax.plot(X, Y[2], '--', color='C1', label='$D_{\\infty h}$ (3-state, $\\theta=90$°)')
ax.plot(X, Y[3], color='C2', label='$C_{{3v}}$ (4-state, $\\theta={}$°)'.format(args.theta))
ax.plot(X, Y[4], '--', color='C2', label='$D_{3h}$ (4-state, $\\theta=90$°)')
ax.plot(X, Y[5], color='C3', label='$T_d$ (5-state)')

ax.legend()
ax.set_xlim(-1, 1)
ax.set_xlabel('$m_{CT}$')
ax.set_ylabel('$\\beta_{SHS}\\times \\mu_{CT}^{-3}\\times t^2$')

plt.tight_layout()
figure.savefig(args.output)
