import numpy
import itertools

def beta_J1sq(tensor: numpy.ndarray) -> float:
    """Compute $|\beta_{J=1}|^2$
    """

    value = .0
    for i, j, k in itertools.product(range(3), repeat=3):
        value += 3/5 * tensor[i, j, j] * tensor[i, k, k]

    return value

def beta_J3sq(tensor: numpy.ndarray) -> float:
    """Compute $|\beta_{J=3}|^2$
    """

    value = .0
    for i, j, k in itertools.product(range(3), repeat=3):
        value += tensor[i, j, k] ** 2 - 3/5 * tensor[i, j, j] * tensor[i, k, k]

    return value

def beta_HRS(tensor: numpy.ndarray) -> float:
    J1sq = beta_J1sq(tensor)
    J3sq = beta_J3sq(tensor)

    b2ZZZ = 1/5 * J1sq + 2/35 * J3sq
    b2ZXX = 1/45 * J1sq + 4/105 * J3sq

    return numpy.sqrt(b2ZZZ + b2ZXX)

def beta_DR(tensor: numpy.ndarray) -> float:
    J1sq = beta_J1sq(tensor)
    J3sq = beta_J3sq(tensor)

    rho2 = J3sq / J1sq

    return (18 * rho2 + 63) / (12 * rho2 + 7)