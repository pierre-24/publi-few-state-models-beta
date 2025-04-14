"""
A package to implement the formulas of the paper and check them.

Note: this code is partially copied from https://github.com/pierre-24/check-sos/.
"""

import numpy

__version__ = '0.1'

HC_IN_EV = 45.56335
AU_TO_EV = 27.211386245981


# Define systems
CT_dipole_2s = lambda mu_CT: [numpy.array([.0, .0, mu_CT])]

CT_dipole_3s = lambda mu_CT, theta: [
    mu_CT * numpy.array([numpy.sin(theta), .0, numpy.cos(theta)]),
    mu_CT * numpy.array([-numpy.sin(theta), .0, numpy.cos(theta)]),
]

CT_dipole_4s = lambda mu_CT, theta: [
    mu_CT * numpy.array([.0, numpy.sin(theta), numpy.cos(theta)]),
    mu_CT / 2 * numpy.array([
        numpy.sqrt(3) * numpy.sin(theta),
        -numpy.sin(theta),
        2 * numpy.cos(theta)
    ]),
    mu_CT / 2 * numpy.array([
        -numpy.sqrt(3) * numpy.sin(theta),
        -numpy.sin(theta),
        2 * numpy.cos(theta)
    ]),
]

CT_dipole_5s = lambda mu_CT: mu_CT * numpy.sqrt(3) / 3 * numpy.array([
    [1, 1, 1],
    [1, -1, -1],
    [-1, -1, 1],
    [-1, 1, -1]
])
