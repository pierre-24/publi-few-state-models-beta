import matplotlib.pyplot as plt
import argparse

from few_state.walks import Walk, plot_walks
from few_state.properties import beta_HRS, beta_DR

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--output', default='Data_exci.pdf')
parser.add_argument('-t', default=1)
parser.add_argument('-T', default=.5)
parser.add_argument('-N', default=30)
parser.add_argument('--ymax', default=0.2, type=float)
parser.add_argument('--m-CTs', nargs='*', type=float, default=[-.75, -.5, 0, .5, .75])

args = parser.parse_args()

figure = plt.figure(figsize=(6, 6))
ax1, ax2 = figure.subplots(2, sharex=True)

walks = [Walk.walk_2s_to_5s_mtT(m_CT, args.t, args.T, args.N) for m_CT in args.m_CTs]

plot_walks(
    ax1,
    walks,
    lambda s: beta_HRS(s.response_tensor((1, 1), frequency=0)),
    ['$m_{{CT}}={:.2f}$'.format(m_CT) for m_CT in args.m_CTs],
    (0, args.ymax),
    '$\\beta_{SHS}\\times \\mu_{CT}^{-3}\\times t^2$',
)

plot_walks(
    ax2,
    walks,
    lambda s: beta_DR(s.response_tensor((1, 1), frequency=0)),
    None,
    (1.5, 9),
    '$DR_{SHS}$',
)

plt.tight_layout()
figure.savefig(args.output)
