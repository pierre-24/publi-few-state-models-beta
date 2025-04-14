import collections
import itertools
import math
import more_itertools
import numpy

from typing import List, Iterable, TextIO
from numpy.typing import NDArray


class ComponentsIterator:
    """Iterate over (unique) components of a NLO tensor
    """

    def __init__(self, input_fields: Iterable[int]):
        self.fields = [-sum(input_fields)] + list(input_fields)

        self.each = collections.Counter(self.fields)
        self.last = {}

        # prepare a ideal→actual map
        N = 0
        self.ideal_to_actual = {}

        for c, n in self.each.items():
            self.last[c] = N
            N += n

        for i, field in enumerate(self.fields):
            self.ideal_to_actual[i] = self.last[field]
            self.last[field] += 1

        self.actual_to_ideal = dict((b, a) for a, b in self.ideal_to_actual.items())

    def __len__(self) -> int:
        """Return the number of components that the iterator will yield, which is a combination of
        combination with replacements for each type of field
        (see https://www.calculatorsoup.com/calculators/discretemathematics/combinationsreplacement.php)
        """
        return int(numpy.prod([math.factorial(3 + n - 1) / math.factorial(n) / 2 for n in self.each.values()]))

    def __iter__(self) -> Iterable[tuple]:
        yield from self.iter()

    def _iter_reordered(self, perms) -> Iterable[tuple]:
        """Perform the cartesian product of all permutation, then reorder so that it matches `self.fields`
        """

        for ideal_component in itertools.product(*perms.values()):
            ideal_component = list(more_itertools.collapse(ideal_component))
            yield tuple(ideal_component[self.ideal_to_actual[i]] for i in range(len(self.fields)))

    def iter(self) -> Iterable[tuple]:
        """Iterate over unique components.
        It is a cartesian product between combination for each type of field, then reordered.
        """

        # perform combinations for each "type" of field
        perms = {}
        for c, n in self.each.items():
            perms[c] = list(itertools.combinations_with_replacement(range(3), n))

        # cartesian product
        yield from self._iter_reordered(perms)

    def reverse(self, component: tuple) -> Iterable[tuple]:
        """Yield all possible (unique) combination of a given component"""

        assert len(component) == len(self.fields)

        # fetch components for each type of fields
        c_with_field = dict((f, []) for f in self.each.keys())
        for i, field in enumerate(self.fields):
            c_with_field[field].append(component[i])

        # permute
        perms = {}
        for field, components in c_with_field.items():
            perms[field] = list(more_itertools.unique_everseen(itertools.permutations(components)))

        # cartesian product
        yield from self._iter_reordered(perms)


class System:
    """A system, represented by a list of excitation energies and corresponding transition dipoles
    """

    def __init__(self, e_exci: List[float], t_dips: NDArray):

        assert len(e_exci) == t_dips.shape[0] - 1

        e_exci.insert(0, 0)

        self.e_exci = numpy.array(e_exci)
        self.t_dips = t_dips

    def __len__(self):
        return len(self.e_exci)

    def to_file(self, f: TextIO):
        f.write('{}\n'.format(len(self) - 1))
        for i, e in enumerate(self.e_exci[1:]):
            f.write('{} {:.7f}\n'.format(i + 1, e))

        for i in range(len(self)):
            for j in range(0, i + 1):
                f.write('{} {} {:.7f} {:.7f} {:.7f}\n'.format(i, j, *self.t_dips[i, j]))

    def response_tensor(
        self,
        input_fields: tuple = (1, 1),
        frequency: float = 0,
        skip_divergent: bool = False,
        verbose: bool = False,
    ) -> NDArray:
        """Get a response tensor, using SOS
        """

        if len(input_fields) == 0:
            raise Exception('input fields is empty?!?')

        it = ComponentsIterator(input_fields)
        t = numpy.zeros(numpy.repeat(3, len(it.fields)))
        e_fields = list(i * frequency for i in it.fields)

        for c in it:
            component = self.response_tensor_element_f(c, e_fields, skip_divergent, verbose)

            for ce in it.reverse(c):
                t[ce] = component

        return t

    def response_tensor_element_f(self, component: tuple, e_fields: List[float], skip_divergent: bool = False, verbose: bool = False) -> float:
        """
        Compute the value of a component of a response tensor, using fluctuation dipoles.

        Note: it breaks for n > 5, since `(ab)_a1(cd)_a3(efg)_a5a6` appears and that I did not yet implement a
        general scheme.
        """

        assert len(component) == len(e_fields)
        assert len(e_fields) < 6

        value = .0

        to_permute = list(zip(component, e_fields))
        num_perm = numpy.prod([math.factorial(i) for i in collections.Counter(to_permute).values()])

        if verbose:
            print('---', num_perm, component)

        for p in more_itertools.unique_everseen(itertools.permutations(to_permute)):
            for states in itertools.product(range(1, len(self)), repeat=len(component) - 1):
                stx = list(states)
                stx.append(0)
                stx.insert(0, 0)

                dips = [
                    self.t_dips[stx[i], stx[i + 1], p[i][0]] - (
                        0 if stx[i] != stx[i + 1]
                        else self.t_dips[0, 0, p[i][0]]
                    ) for i in range(len(component))
                ]

                ens = [
                    self.e_exci[e] + sum(p[j][1] for j in range(i + 1)) for i, e in enumerate(states)
                ]

                if abs(numpy.prod(dips)) > 1e-3 and verbose:
                    print([_p[0] for _p in p], stx, numpy.prod(dips) / numpy.prod(ens))

                value += numpy.prod(dips) / numpy.prod(ens)

        if len(component) > 3 and not skip_divergent:
            for set_g in range(1, len(component) - 2):
                value += self._secular_term_non_divergent(component, e_fields, set_g, verbose)
        
        return value * num_perm

    def _secular_term_non_divergent(self, component: tuple, e_fields: List[float], set_ground: int, verbose: bool = False) -> float:
        """provide a non-divergent secular term.
        """

        value = .0

        to_permute = list(zip(component, e_fields))

        for p in more_itertools.unique_everseen(itertools.permutations(to_permute)):
            x = -sum(p[j][1] for j in range(set_ground + 1))

            for states in itertools.product(range(1, len(self)), repeat=len(component) - 2):
                states = list(states)

                states.insert(set_ground, 0)

                stx = list(states)
                stx.append(0)
                stx.insert(0, 0)

                # numerator
                dips = [
                    self.t_dips[stx[i], stx[i + 1], p[i][0]] - (
                        0 if stx[i] != stx[i + 1]
                        else self.t_dips[0, 0, p[i][0]]
                    ) for i in range(len(component))
                ]

                # denominator
                # TODO: there is probably a way to write a nicer code here, without all those `continue`.
                for i in range(len(component) - 1):
                    if i == set_ground:
                        continue

                    ens = []

                    for l_ in range(i + 1):
                        if l_ == set_ground:
                            continue

                        ens.append(self.e_exci[states[l_]] + sum(p[j][1] for j in range(l_ + 1)))

                    for l_ in range(i, len(component) - 1):
                        if l_ == set_ground:
                            continue

                        ens.append(self.e_exci[states[l_]] + sum(p[j][1] for j in range(l_ + 1)) + x)

                    if abs(numpy.prod(dips)) > 1e-3 and verbose:
                        print([_p[0] for _p in p], stx, numpy.prod(dips) / numpy.prod(ens))

                    value += numpy.prod(dips) / numpy.prod(ens)
        
        return -.5 * value
