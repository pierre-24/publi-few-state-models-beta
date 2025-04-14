from few_state import CT_dipole_2s, CT_dipole_3s, CT_dipole_4s, CT_dipole_5s
from few_state.system_maker import make_system_from_mtT
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

figure = plt.figure(figsize=(5,7))
ax1, ax2  = figure.subplots(2, sharex=True)

X = numpy.linspace(-1, 1, args.N)
Y = numpy.zeros((4, X.shape[0]))
Yp = numpy.zeros((4, X.shape[0]))

for i, m_CT in enumerate(X):
    system_2s = make_system_from_mtT(m_CT, args.t, args.T, CT_dipole_2s(args.mu_CT))
    Y[0, i] = system_2s.e_exci[-1]
    system_3s = make_system_from_mtT(m_CT, args.t, args.T, CT_dipole_3s(args.mu_CT, numpy.deg2rad(args.theta)))
    Y[1, i] = system_3s.e_exci[-1]
    Yp[1, i] = system_3s.e_exci[1]
    system_4s = make_system_from_mtT(m_CT, args.t, args.T, CT_dipole_4s(args.mu_CT, numpy.deg2rad(args.theta)))
    Y[2, i] = system_4s.e_exci[-1]
    Yp[2, i] = system_4s.e_exci[1]
    system_5s = make_system_from_mtT(m_CT, args.t, args.T, CT_dipole_5s(args.mu_CT))
    Y[3, i] = system_5s.e_exci[-1]
    Yp[3, i] = system_5s.e_exci[1]

ax1.plot(X, Y[0], color='C0', label='2-state')
ax1.plot(X, Y[1], color='C1', label='3-state ($\\theta={}$°)'.format(args.theta))
ax2.plot(X, Yp[1], '-', color='C1')
ax1.plot(X, Y[2], color='C2', label='4-state ($\\theta={}$°)'.format(args.theta))
ax2.plot(X, Yp[2], '-', color='C2')
ax1.plot(X, Y[3], color='C3', label='5-state')
ax2.plot(X, Yp[3], '-', color='C3')

ax1.legend()
ax1.set_ylim(0, 15)
ax1.set_ylabel('$\\Delta E_{0f}\\,/\\, t$')

ax2.set_ylim(0, 15)
ax2.set_xlim(-1, 1)
ax2.set_xlabel('$m_{CT}$')
ax2.set_ylabel('$\\Delta E_{0e}\\,/\\, t$')

plt.tight_layout()
figure.savefig(args.output)
