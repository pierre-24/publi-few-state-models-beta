from few_state import CT_dipole_3s
from few_state.system_maker import make_system_from_mtT
from few_state.properties import beta_HRS
import matplotlib.pyplot as plt
import argparse
import numpy

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='Data_exci.pdf')
parser.add_argument('-t', default=1, type=float)
parser.add_argument('-T', default=.1, type=float)
parser.add_argument('--mu_CT', default=1, type=float)
parser.add_argument('-N', default=100, type=int)

args = parser.parse_args()

figure = plt.figure(figsize=(6, 5))
ax = figure.subplots()

X = numpy.linspace(-1, 1, args.N)
Y = numpy.linspace(0, 90, args.N)

im = numpy.zeros((args.N, args.N))

max_value = -numpy.inf
max_pos = (0, 0)

for i, m_CT in enumerate(X):
    for j, theta in enumerate(Y):
        system_3s = make_system_from_mtT(m_CT, args.t, args.T, CT_dipole_3s(args.mu_CT, numpy.deg2rad(theta)))
        tensor = system_3s.response_tensor((1, 1), frequency=0)
        im[j, i] = beta_HRS(tensor)

        if im[j, i] > max_value:
            max_value = im[j, i]
            max_pos = (j, i)

imx = ax.imshow(im)
ax.figure.colorbar(imx, ax=ax, label='$\\beta_{SHS}\\times \\mu_{CT}^{-3}\\times t^2$', fraction=0.046, pad=0.04)

ax.set_xlabel('$m_{CT}$')
ax.set_xticks([0, args.N / 4, 2 * args.N / 4, 3 * args.N / 4, args.N], [-1, -.5, 0, .5, 1])
ax.set_ylabel('$\\theta$ (°)')
ax.set_yticks([0, args.N / 3, 2 * args.N/3, args.N], [0, 30, 60, 90])

ax.hlines(max_pos[0], 0, args.N, ls='--', color='white', lw=0.7)
ax.vlines(max_pos[1], 0, args.N, ls='--', color='white', lw=0.7)
ax.text(max_pos[1], 5, '{:.3f}'.format((max_pos[1] / args.N) * 2 - 1), color='white')
ax.text(10, max_pos[0], '{:.1f}°'.format((max_pos[0] / args.N) * 90), c='white')

ax.set_xlim(0, args.N)
ax.set_ylim(0, args.N)
ax.set_title('T = {}t'.format(args.T))

plt.tight_layout()
figure.savefig(args.output)