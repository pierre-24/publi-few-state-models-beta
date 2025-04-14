from few_state import CT_dipole_2s, CT_dipole_3s, CT_dipole_4s, CT_dipole_5s
from few_state.system_maker import make_system_from_mtT
import matplotlib.pyplot as plt
import argparse
import numpy

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='Data_exci.pdf')
parser.add_argument('-t', default=1, type=float)
parser.add_argument('-T', default=.5, type=float)
parser.add_argument('--mu_CT', default=1, type=float)
parser.add_argument('--theta', default=30, type=float)
parser.add_argument('-N', default=100, type=int)
parser.add_argument('-C', default='zzz')
parser.add_argument('-S', default=2, type=int)

args = parser.parse_args()

figure = plt.figure(figsize=(4, 3))
ax = figure.subplots()

tr_ = {'x': 0, 'y': 1, 'z': 2}
components = []

spx = args.C.split(',')
for i, sp in enumerate(spx):
    components.append(tuple(tr_[x] for x in sp))

X = numpy.linspace(-1, 1, args.N)
Y = numpy.zeros((len(spx), X.shape[0]))


for i, m_CT in enumerate(X):
    if args.S == 2:
        system = make_system_from_mtT(m_CT, args.t, args.T, CT_dipole_2s(args.mu_CT))
    elif args.S == 3:
        system = make_system_from_mtT(m_CT, args.t, args.T, CT_dipole_3s(args.mu_CT, numpy.deg2rad(args.theta)))
    elif args.S == 4:
        system = make_system_from_mtT(m_CT, args.t, args.T, CT_dipole_4s(args.mu_CT, numpy.deg2rad(args.theta)))
    elif args.S == 5:
        system = make_system_from_mtT(m_CT, args.t, args.T, CT_dipole_5s(args.mu_CT))
    else:
        raise Exception('Unknown system type')

    tensor = system.response_tensor((1, 1), frequency=0)

    for j, cpt in enumerate(components):
        Y[j, i] = tensor[*cpt]


for i, cpt in enumerate(spx):
    ax.plot(X, Y[i], label=cpt)


ax.legend()
ax.set_title('$\\theta={:.0f}$°'.format(args.theta))

ax.set_xlim(-1, 1)
ax.set_xlabel('$m_{CT}$')
ax.set_ylabel('$\\beta_{ijk}\\times\\mu_{CT}^{-3}\\times t^2$')

plt.tight_layout()
figure.savefig(args.output)
