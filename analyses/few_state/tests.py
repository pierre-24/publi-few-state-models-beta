import itertools

import numpy
import pytest
from few_state import CT_dipole_2s, CT_dipole_3s, CT_dipole_4s, CT_dipole_5s
from few_state.system_maker import make_system_from_E2tT, make_system_from_mtT
from few_state.properties import beta_J1sq, beta_J3sq, beta_HRS, beta_DR


def _get_energies(E_VB, E_CT, t, T, n):

    V = E_CT - E_VB

    E_0 = .5 * (E_VB + E_CT - (n - 1) * T - numpy.sqrt((V - (n - 1) * T) ** 2 + 4 * n * t ** 2))
    E_e1 = E_CT + T
    E_f = .5 * (E_VB + E_CT - (n - 1) * T + numpy.sqrt((V - (n - 1) * T) ** 2 + 4 * n * t ** 2))

    return E_0, E_e1, E_f


def test_make_system_E2tT():
    E_VB, E_CT = -1, 1
    t = .5
    T = .1
    theta = numpy.deg2rad(45)
    mu_CT = 1

    # 2-state system
    n = 1
    system_2s = make_system_from_E2tT(E_VB, E_CT, t, T, CT_dipole_2s(theta))
    E_0, E_e1, E_f = _get_energies(E_VB, E_CT, t, T, n)

    assert system_2s.e_exci[0] == .0
    assert system_2s.e_exci[1] == pytest.approx(E_f - E_0)

    # 3-state system
    n = 2
    system_3s = make_system_from_E2tT(E_VB, E_CT, t, T, CT_dipole_3s(mu_CT, theta))
    E_0, E_e1, E_f = _get_energies(E_VB, E_CT, t, T, n)

    assert system_3s.e_exci[0] == .0
    assert system_3s.e_exci[1] == pytest.approx(E_e1 - E_0)
    assert system_3s.e_exci[2] == pytest.approx(E_f - E_0)

    # 4-state system
    n = 3
    system_4s = make_system_from_E2tT(E_VB, E_CT, t, T, CT_dipole_4s(mu_CT, theta))
    E_0, E_e1, E_f = _get_energies(E_VB, E_CT, t, T, n)

    assert system_4s.e_exci[0] == .0
    assert system_4s.e_exci[1] == pytest.approx(E_e1 - E_0)
    assert system_4s.e_exci[2] == pytest.approx(E_e1 - E_0)
    assert system_4s.e_exci[3] == pytest.approx(E_f - E_0)

    # 5-state system
    n = 4
    system_5s = make_system_from_E2tT(E_VB, E_CT, t, T, CT_dipole_5s(mu_CT))
    E_0, E_e1, E_f = _get_energies(E_VB, E_CT, t, T, n)

    assert system_5s.e_exci[0] == .0
    assert system_5s.e_exci[1] == pytest.approx(E_e1 - E_0)
    assert system_5s.e_exci[2] == pytest.approx(E_e1 - E_0)
    assert system_5s.e_exci[3] == pytest.approx(E_e1 - E_0)
    assert system_5s.e_exci[4] == pytest.approx(E_f - E_0)


def _check_expr(N: int, product: tuple[str, str], non_null: set[str], tr: dict[str, str] = {}):
    terms = {}

    for ngram in itertools.product(('x', 'y', 'z'), repeat=N):
        p0, p1 = product[0], product[1]
        for i, c in enumerate(ngram):
            p0 = p0.replace(str(i), c)
            p1 = p1.replace(str(i), c)

        rp0 = ''.join(sorted(p0))
        rp1 = ''.join(sorted(p1))

        if rp0 in tr:
            rp0 = tr[rp0]
        if rp1 in tr:
            rp1 = tr[rp1]

        if rp0 in non_null and rp1 in non_null:
            term = tuple(sorted([rp0, rp1]))
            if term not in terms:
                terms[term] = 0
            terms[term] += 1

    final_expr = ''
    for term, v in terms.items():
        if term[0] != term[1]:
            final_expr += ' + {} {}*{}'.format(v, term[0], term[1])
        else:
            final_expr += ' + {} {}^2'.format(v, term[0])

    print(final_expr)

def test_invariants():
    """Just check the "simplified" invariant formula for different symmetries
    """
    m_CT = -.5
    t = .05
    T = .01
    theta = numpy.deg2rad(45)
    mu_CT = 1.25

    # 2-state system
    system_2s = make_system_from_mtT(m_CT, t, T, CT_dipole_2s(mu_CT))

    t_2s = system_2s.response_tensor((1, 1), frequency=0)
    assert beta_J1sq(t_2s) == pytest.approx(3/5 * t_2s[2, 2, 2] ** 2)
    assert beta_J3sq(t_2s) == pytest.approx(2/5 * t_2s[2, 2, 2] ** 2)
    assert beta_HRS(t_2s) == pytest.approx(numpy.sqrt(6/35) * numpy.abs(t_2s[2, 2, 2]))
    assert beta_DR(t_2s) == pytest.approx(5.0)

    # 3-state system
    system_3s = make_system_from_mtT(m_CT, t, T, CT_dipole_3s(mu_CT, theta))

    t_3s = system_3s.response_tensor((1, 1), frequency=0)
    # _check_expr(3, ('011', '022'), {'zzz', 'xxz'})
    # _check_expr(3, ('012', '012'), {'zzz', 'xxz'})
    coefs = 1/5 * numpy.array([
        [3, 3, 6],
        [2, 12, -6],
    ])
    cpts = numpy.array([
        t_3s[2, 2, 2] ** 2,
        t_3s[2, 0, 0] ** 2,
        t_3s[2, 0, 0] * t_3s[2, 2, 2]
    ])
    assert numpy.allclose(
        [beta_J1sq(t_3s), beta_J3sq(t_3s)],
        coefs @ cpts
    )
    assert beta_HRS(t_3s) == pytest.approx(numpy.sqrt(numpy.sum(numpy.array([10/45, 10/105]).dot(coefs @ cpts))))

    # 4-state system
    system_4s = make_system_from_mtT(m_CT, t, T, CT_dipole_4s(mu_CT, theta))

    t_4s = system_4s.response_tensor((1, 1), frequency=0)
    #_check_expr(3, ('011', '022'), {'zzz', 'yyy', 'xxy', 'xxz', 'yyz'}, {'xxz': 'yyz'})
    #_check_expr(3, ('012', '012'), {'zzz', 'yyy', 'xxy', 'xxz', 'yyz'}, {'xxz': 'yyz'})
    cpts = numpy.array([
        t_4s[1, 1, 2] ** 2,
        t_4s[1, 1, 1] ** 2,
        t_4s[1, 1, 2] * t_4s[2, 2, 2],
        t_4s[2, 2, 2] ** 2,
    ])
    coefs = 1/5 * numpy.array([
            [12, 0, 12, 3],
            [18, 20, -12, 2],
    ])
    assert numpy.allclose(
        [beta_J1sq(t_4s), beta_J3sq(t_4s)],
        coefs @ cpts
    )
    assert beta_HRS(t_4s) == pytest.approx(numpy.sqrt(numpy.sum(numpy.array([10/45, 10/105]).dot(coefs @ cpts))))

    # 5-state system
    system_5s = make_system_from_mtT(m_CT, t, T, CT_dipole_5s(mu_CT))

    t_5s = system_5s.response_tensor((1, 1), frequency=0)
    assert beta_J1sq(t_5s) == pytest.approx(0)
    assert beta_J3sq(t_5s) == pytest.approx(6 * t_5s[0, 1, 2] ** 2)
    assert beta_HRS(t_5s) == pytest.approx(numpy.sqrt(4/7) * numpy.abs(t_5s[0, 1, 2]))
    assert beta_DR(t_5s) == pytest.approx(1.5)


def test_invariants_high_symmetry():
    """Just check the "simplified" invariant formula for "high" symmetries
    """
    m_CT = 1 - 1e-15
    t = .05
    T = .01
    theta = numpy.deg2rad(90)
    mu_CT = 1.25

    # 4-state system (D_3h)
    system_4s = make_system_from_mtT(m_CT, t, T, CT_dipole_4s(mu_CT, theta))

    t_4s = system_4s.response_tensor((1, 1), frequency=0)
    assert beta_J1sq(t_4s) == pytest.approx(.0)
    assert beta_J3sq(t_4s) == pytest.approx(4 * t_4s[1, 1, 1] ** 2)
    assert beta_HRS(t_4s) == pytest.approx(numpy.sqrt(8/21) * numpy.abs(t_4s[1, 1, 1]))
    assert beta_DR(t_4s) == pytest.approx(1.5)


def test_make_system_mtT_dipoles():
    """Check dipoles. See Section 1.1 of SI.
    """

    m_CT = .1
    t = .05
    T = .01
    theta = numpy.deg2rad(60)
    mu_CT = 1.25

    # 2-state system (Section 1.1.1 of SI)
    system_2s = make_system_from_mtT(m_CT, t, T, CT_dipole_2s(mu_CT))

    assert system_2s.t_dips[0, 0][2] == pytest.approx(mu_CT * (1 + m_CT) / 2)
    assert system_2s.t_dips[0, 1][2] == pytest.approx(-mu_CT / 2 * numpy.sqrt(1 - m_CT ** 2))
    assert system_2s.t_dips[1, 1][2] == pytest.approx(mu_CT * (1 - m_CT) / 2)

    # 3-state system (Section 1.1.2 of SI)
    system_3s = make_system_from_mtT(m_CT, t, T, CT_dipole_3s(mu_CT, theta))

    assert numpy.allclose(system_3s.t_dips[0, 0], [.0, .0, mu_CT * (1 + m_CT) / 2 * numpy.cos(theta)])
    assert numpy.allclose(system_3s.t_dips[0, 1], [-mu_CT * numpy.sqrt((1 + m_CT) / 2) * numpy.sin(theta), .0, .0])
    assert numpy.allclose(system_3s.t_dips[0, 2], [.0, .0, -mu_CT / 2 * numpy.sqrt(1 - m_CT ** 2) * numpy.cos(theta)])
    assert numpy.allclose(system_3s.t_dips[1, 1], [.0, .0, mu_CT * numpy.cos(theta)])
    assert numpy.allclose(system_3s.t_dips[1, 2], [mu_CT * numpy.sqrt((1 - m_CT) / 2) * numpy.sin(theta), .0, .0])
    assert numpy.allclose(system_3s.t_dips[2, 2], [.0, .0, mu_CT * (1 - m_CT) / 2 * numpy.cos(theta)])

    # 4-state system (Section 1.1.3)
    system_4s = make_system_from_mtT(m_CT, t, T, CT_dipole_4s(mu_CT, theta))

    assert numpy.allclose(system_4s.t_dips[0, 0], [.0, .0, mu_CT * (1 + m_CT) / 2 * numpy.cos(theta)])

    assert numpy.allclose(
        system_4s.t_dips[0, 1],
        mu_CT / 4 * numpy.sqrt((1 + m_CT) / 2) * numpy.array(
            [numpy.sqrt(2) * numpy.sin(theta), -numpy.sqrt(6) * numpy.sin(theta), 0]
        )
    )

    assert numpy.allclose(
        system_4s.t_dips[0, 2],
        mu_CT / 4 * numpy.sqrt(2) * numpy.sqrt((1 + m_CT) / 2) * numpy.array(
            [-numpy.sqrt(3) * numpy.sin(theta), -numpy.sin(theta), 0]
        )
    )

    assert numpy.allclose(
        system_4s.t_dips[0, 3],
        mu_CT / 2 * numpy.sqrt(1 - m_CT**2) * numpy.array([0, 0, -numpy.cos(theta)])
    )

    assert numpy.allclose(
        system_4s.t_dips[1, 1],
        mu_CT / 4 * numpy.array([numpy.sqrt(3) * numpy.sin(theta), numpy.sin(theta), 4 * numpy.cos(theta)])
    )

    assert numpy.allclose(
        system_4s.t_dips[1, 2],
        mu_CT / 4 * numpy.array([-numpy.sin(theta), numpy.sqrt(3) * numpy.sin(theta), 0])
    )

    assert numpy.allclose(
        system_4s.t_dips[1, 3],
        mu_CT / 4 * numpy.sqrt(2) * numpy.sqrt((1 - m_CT) / 2) * numpy.array(
            [-numpy.sin(theta), numpy.sqrt(3) * numpy.sin(theta), 0]
        )
    )

    assert numpy.allclose(
        system_4s.t_dips[2, 2],
        mu_CT / 4 * numpy.array([-numpy.sqrt(3) * numpy.sin(theta), -numpy.sin(theta), 4 * numpy.cos(theta)])
    )

    assert numpy.allclose(
        system_4s.t_dips[2, 3],
        mu_CT / 4 * numpy.sqrt(2) * numpy.sqrt((1 - m_CT) / 2) * numpy.array(
            [numpy.sqrt(3) * numpy.sin(theta), numpy.sin(theta), 0]
        )
    )

    assert numpy.allclose(system_4s.t_dips[3, 3], [.0, .0, mu_CT * (1 - m_CT) / 2 * numpy.cos(theta)])

    # 5-state system (Section 1.1.4)
    system_5s = make_system_from_mtT(m_CT, t, T, CT_dipole_5s(mu_CT))

    assert numpy.allclose(system_5s.t_dips[0, 0], [.0, .0, .0])

    assert numpy.allclose(
        system_5s.t_dips[0, 1],
        mu_CT * numpy.sqrt(6) / 6 * numpy.sqrt((1 + m_CT) / 2) * numpy.array(
            [.0, -1., -1.]
        )
    )

    assert numpy.allclose(
        system_5s.t_dips[0, 2],
        mu_CT * numpy.sqrt(18) / 18 * numpy.sqrt((1 + m_CT) / 2) * numpy.array(
            [-2., -1., 1.]
        )
    )

    assert numpy.allclose(
        system_5s.t_dips[0, 3],
        mu_CT * 1 / 3 * numpy.sqrt((1 + m_CT) / 2) * numpy.array(
            [-1., 1., -1.]
        )
    )

    assert numpy.allclose(system_5s.t_dips[0, 4], [.0, .0, .0])

    assert numpy.allclose(system_5s.t_dips[1, 1], mu_CT * numpy.sqrt(3) / 3  * numpy.array([1., .0, .0]))

    assert numpy.allclose(system_5s.t_dips[1, 2], mu_CT / 3  * numpy.array([0., 1., 1.]))
    assert numpy.allclose(system_5s.t_dips[1, 3], mu_CT * numpy.sqrt(2) / 6  * numpy.array([0., 1., 1.]))

    assert numpy.allclose(system_5s.t_dips[2, 2], mu_CT * numpy.sqrt(3) / 9  * numpy.array([-1., -2., 2.]))
    assert numpy.allclose(system_5s.t_dips[2, 3], mu_CT * numpy.sqrt(6) / 18  * numpy.array([2., 1., -1.]))

    assert numpy.allclose(system_5s.t_dips[3, 3], mu_CT * 2 * numpy.sqrt(3) / 9  * numpy.array([-1., 1., -1.]))

    assert numpy.allclose(system_5s.t_dips[4, 4], [.0, .0, .0])


def test_make_system_mtT_beta():
    m_CT = -.2
    t = .05
    T = .01
    theta = numpy.deg2rad(60)
    mu_CT = 1.

    # 2-state system
    system_2s = make_system_from_mtT(m_CT, t, T, CT_dipole_2s(mu_CT))

    t_2s = system_2s.response_tensor((1, 1), frequency=0)

    assert t_2s[2, 2, 2] == pytest.approx(-3. / 8 * m_CT * (1 - m_CT**2)**2 * mu_CT**3 / t**2)
    assert all(x == .0 for x in t_2s.flatten()[:-1])

    # 3-state system
    system_3s = make_system_from_mtT(m_CT, t, T, CT_dipole_3s(mu_CT, theta))

    t_3s = system_3s.response_tensor((1, 1), frequency=0)

    assert t_3s[2, 2, 2] == pytest.approx(-3. / 16 * m_CT * (1 - m_CT**2)**2 * mu_CT**3 / t**2 * numpy.cos(theta) ** 3)

    assert t_3s[2, 0, 0] == pytest.approx((1 - m_CT**2) / 2 * mu_CT**3 * numpy.sin(theta) ** 2 * numpy.cos(theta) * (
        (2 * T + t * numpy.sqrt(2 * (1 - m_CT) / (1 + m_CT)))**-2 + 2 * (
            2 * t * numpy.sqrt(2 / (1 - m_CT**2)) * (2 * T + t * numpy.sqrt(2 * (1 - m_CT) / (1 + m_CT)))
        )**-1
    ))

    mask = numpy.zeros((3, 3, 3), dtype=bool)
    mask[2, 2, 2] = True
    for i in itertools.permutations((2, 0, 0)):
        mask[i] = True

    assert numpy.allclose(t_3s[~mask], 0)

    # 4-state system
    system_4s = make_system_from_mtT(m_CT, t, T, CT_dipole_4s(mu_CT, theta))

    t_4s = system_4s.response_tensor((1, 1), frequency=0)

    assert t_4s[2, 2, 2] == pytest.approx(-1. / 8 * m_CT * (1 - m_CT**2)**2 * mu_CT**3 / t**2 * numpy.cos(theta) ** 3)

    assert t_4s[2, 1, 1] == pytest.approx((1 - m_CT**2) / 4 * mu_CT**3 * numpy.sin(theta) ** 2 * numpy.cos(theta) * (
        (3 * T + t * numpy.sqrt(3 * (1 - m_CT) / (1 + m_CT)))**-2 + 2 * (
            2 * t * numpy.sqrt(3 / (1 - m_CT**2)) * (3 * T + t * numpy.sqrt(3 * (1 - m_CT) / (1 + m_CT)))
        )**-1
    ))

    assert t_4s[2, 0, 0] == pytest.approx(t_4s[2, 1, 1])

    assert t_4s[1, 1, 1] == pytest.approx(3 * (1 + m_CT) / 4 * mu_CT**3 * numpy.sin(theta) ** 3 * (
        (3 * T + t * numpy.sqrt(3 * (1 - m_CT) / (1 + m_CT)))**-2
    ))

    assert t_4s[1, 1, 1] == pytest.approx(-t_4s[1, 0, 0])

    mask = numpy.zeros((3, 3, 3), dtype=bool)
    mask[2, 2, 2] = True
    mask[1, 1, 1] = True
    for i in itertools.permutations((2, 0, 0)):
        mask[i] = True
    for i in itertools.permutations((2, 1, 1)):
        mask[i] = True
    for i in itertools.permutations((1, 0, 0)):
        mask[i] = True

    assert numpy.allclose(t_4s[~mask], 0)

    # 5-state system
    system_5s = make_system_from_mtT(m_CT, t, T, CT_dipole_5s(mu_CT))

    t_5s = system_5s.response_tensor((1, 1), frequency=0)

    assert t_5s[0, 1, 2] == pytest.approx(
       numpy.sqrt(3) / 3 * (1 + m_CT) * mu_CT**3 / (4 * T + t * numpy.sqrt(4*(1-m_CT)/(1+m_CT)))**2)

    mask = numpy.zeros((3, 3, 3), dtype=bool)
    for i in itertools.permutations((0, 1, 2)):
        mask[i] = True

    assert numpy.allclose(t_5s[~mask], 0)

def test_high_symmetry():
    m_CT = 1 - 1e-15  # using 1 leads to `DivisionByZero` ;)
    t = .05
    T = .01
    theta = numpy.deg2rad(90)
    mu_CT = 1.25

    # 2-state system
    system_2s = make_system_from_mtT(m_CT, t, T, CT_dipole_2s(mu_CT))

    t_2s = system_2s.response_tensor((1,), frequency=0)

    assert numpy.allclose(t_2s, .0)

    # 3-state system (D_ooh)
    system_3s = make_system_from_mtT(m_CT, t, T, CT_dipole_3s(mu_CT, theta))

    t_3s = system_3s.response_tensor((1, 1), frequency=0)
    assert numpy.allclose(t_3s, .0)

    # 4-state system (D_3h)
    system_4s = make_system_from_mtT(m_CT, t, T, CT_dipole_4s(mu_CT, theta))

    t_4s = system_4s.response_tensor((1, 1), frequency=0)
    assert t_4s[1, 1, 1] == pytest.approx(mu_CT ** 3 / 6 * T ** -2)
    assert beta_J1sq(t_4s) == pytest.approx(.0)
    assert beta_J3sq(t_4s) == pytest.approx(mu_CT ** 6 / 9 * T **-4)
    assert beta_HRS(t_4s) == pytest.approx(numpy.sqrt(2) / (3 * numpy.sqrt(21)) * mu_CT ** 3 * T ** -2)

    # 5-state system (T_d)
    system_5s = make_system_from_mtT(m_CT, t, T, CT_dipole_5s(mu_CT))

    t_5s = system_5s.response_tensor((1, 1), frequency=0)
    assert t_5s[0, 1, 2] == pytest.approx(numpy.sqrt(3) / 24 * mu_CT**3 * T**-2)
    assert beta_J1sq(t_5s) == pytest.approx(.0)
    assert beta_J3sq(t_5s) == pytest.approx(mu_CT ** 6 / 32 * T **-4)
    assert beta_HRS(t_5s) == pytest.approx(1 / (4 * numpy.sqrt(21)) * mu_CT**3 * T**-2)
