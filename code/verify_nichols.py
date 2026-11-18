#!/usr/bin/env python3
"""Witness Q, Q², Q³ and Q⁴ in the even part of SU(2) at level 13.

This computes ONE diagonal entry of the Woronowicz symmetrizer in each
of degrees 2, 4, 6 and 8. It proves nonvanishing, not the full ranks.

Conventions: spin v=5; q=exp(i*pi/15); left-associated fusion paths.
The source path (5,0,5,0,...) is the product of adjacent invariant cups.
Every pair returns to the unit through its unique fusion channel. The
path thus represents the tensor power of the degree-two invariant up
to nonzero scalar. Its image in the Nichols algebra is nonzero if its
quantum symmetrization is nonzero.

In the real orthogonal q-6j gauge, for the vertex (a,v,v;b), write
F_xy = sign * A_x * B_y * T_xy, with T the rational Racah sum,
 A_x² = [2x+1] Delta(a,v,x)² Delta(x,v,b)²,
 B_y² = [2y+1] Delta(v,v,y)² Delta(a,y,b)².
The sign is the same for all x,y in the block. Rescale each path by
the product of sqrt([2j+1]) over internal edges and Delta over fusion
vertices. The ratio of these factors when x is replaced by x' is
A_x'/A_x. Thus G = F diag(R) F^t, in the rescaled path basis, has
the rational entries A_x² sum_y B_y² T_x'y T_xy R(v,v;y).
Diagonal entries of the final symmetrizer are unchanged by rescaling.

The prime 31 admits q -> 3 of order 30, and [1] through [14] are units.
The denominator of a Delta² uses an admissible vertex sum plus one,
hence at most 14. In the Racah sum z-low[i] <= high[0]-low[i] and
high[j]-z <= high[j]-low[0]. Each right-hand side is the sum of
two spins minus one, hence at most 12 since the spins are at most 6.
The script also checks denominators before inverting. A NUMERATOR
factorial with index at least 15 vanishes because [15]=0 both in
characteristic zero and in this residue field. Consequently a
nonzero residue proves nonvanishing for the complex root of unity.

Run with Python 3.8+; only the standard library is required.
"""
from functools import lru_cache
from math import prod


LEVEL = 13
SPIN = 5
PRIME = 31
ROOT = 3
EXPECTED = {2: (1, 2), 4: (4, 30), 6: (34, 23), 8: (364, 26)}


def quantum_integer(n):
    numerator = pow(ROOT, n, PRIME) - pow(ROOT, -n, PRIME)
    denominator = ROOT - pow(ROOT, -1, PRIME)
    return numerator * pow(denominator, -1, PRIME) % PRIME


@lru_cache(maxsize=None)
def factorial(n):
    if n < 0:
        raise ArithmeticError("Negative q-factorial index")
    return prod(quantum_integer(j) for j in range(1, n + 1)) % PRIME


def inverse_factorials(indices):
    if not all(0 <= i <= LEVEL + 1 for i in indices):
        raise ArithmeticError(f"Index outside the permitted denominators: {indices}")
    return pow(prod(factorial(i) for i in indices), -1, PRIME)


@lru_cache(maxsize=None)
def triangle_squared(a, b, c):
    numerator = factorial(a+b-c) * factorial(a-b+c) * factorial(-a+b+c)
    return numerator * inverse_factorials((a+b+c+1,)) % PRIME


@lru_cache(maxsize=None)
def racah_sum(a, b, x, c, d, y):
    """Bare factorial sum; triangle factors are inserted in braid_entry."""
    low = (a+b+x, x+c+d, b+c+y, a+y+d)
    high = (a+b+c+d, a+c+x+y, b+d+x+y)
    total = 0
    for z in range(max(low), min(high) + 1):
        indices = tuple(z-l for l in low) + tuple(h-z for h in high)
        total += (-1)**z * factorial(z+1) * inverse_factorials(indices)
    return total % PRIME


def fusion(a, b):
    return range(abs(a-b), min(a+b, LEVEL-a-b) + 1)


def braiding_eigenvalue(y):
    exponent = y*(y+1) - 2*SPIN*(SPIN+1)
    return (-1)**(2*SPIN-y) * pow(ROOT, exponent, PRIME) % PRIME


@lru_cache(maxsize=None)
def braid_entry(a, b, x_prime, x):
    v = SPIN
    a_squared = quantum_integer(2*x+1) * triangle_squared(a,v,x) * triangle_squared(x,v,b)
    total = 0
    for y in fusion(v,v):
        if b in fusion(a,y):
            b_squared = quantum_integer(2*y+1) * triangle_squared(v,v,y) * triangle_squared(a,y,b)
            total += b_squared * racah_sum(a,v,x_prime,v,b,y) * racah_sum(a,v,x,v,b,y) * braiding_eigenvalue(y)
    return a_squared * total % PRIME


def build_braids(n):
    paths = [(SPIN,)]
    for _ in range(1, n):
        paths = [p+(j,) for p in paths for j in fusion(p[-1], SPIN)]
    paths = [p for p in paths if p[-1] == 0]
    index = {p: k for k, p in enumerate(paths)}
    generators = []
    for i in range(1, n):
        columns = []
        for k, p in enumerate(paths):
            if i == 1:
                columns.append({k: braiding_eigenvalue(p[1])})
                continue
            a, x, b = p[i-2:i+1]
            column = {}
            for x_prime in fusion(a, SPIN):
                if b in fusion(x_prime, SPIN):
                    row = index[p[:i-1] + (x_prime,) + p[i:]]
                    value = braid_entry(a, b, x_prime, x)
                    if value:
                        column[row] = value
            columns.append(column)
        generators.append(columns)
    return paths, index, generators


def apply(matrix, vector):
    accumulated = {}
    for j, v in vector.items():
        for i, value in matrix[j].items():
            accumulated[i] = accumulated.get(i, 0) + value*v
    return {i: a % PRIME for i, a in accumulated.items() if a % PRIME}


def check_braid_relations(generators, dim):
    """Check on every basis vector of the terminal-unit multiplicity space."""
    for i, g in enumerate(generators):
        for j in range(i+1, len(generators)):
            h = generators[j]
            for k in range(dim):
                gv, hv = g[k], h[k]
                if j == i+1:
                    left, right = apply(g, apply(h, gv)), apply(h, apply(g, hv))
                else:
                    left, right = apply(g, hv), apply(h, gv)
                if left != right:
                    raise ArithmeticError(f"Braid relation failed at {i}, {j}, column {k}")


def witness(n):
    paths, index, generators = build_braids(n)
    check_braid_relations(generators, len(paths))
    cup = index[(SPIN, 0) * (n//2)]
    vector = {cup: 1}
    # S_n = T_2 ... T_n, T_k = 1+g_(k-1)+...+g_(k-1)...g_1.
    # Act first by T_n. Horner's rule constructs each shuffle sum.
    for k in range(n, 1, -1):
        result = vector
        for generator in generators[:k-1]:
            result = apply(generator, result)
            for i, value in vector.items():
                result[i] = (result.get(i, 0) + value) % PRIME
        vector = result
    return len(paths), vector.get(cup, 0)


if __name__ == "__main__":
    if pow(ROOT, 30, PRIME) != 1 or any(pow(ROOT, d, PRIME) == 1 for d in (6, 10, 15)):
        raise ArithmeticError("The chosen residue lacks exact order 30")
    if any(quantum_integer(i) == 0 for i in range(1, 15)) or quantum_integer(15) != 0:
        raise ArithmeticError("Unexpected q-integer specialization")
    print("degree  unit-channel dimension  symmetrizer diagonal mod 31")
    for degree, expected in EXPECTED.items():
        result = witness(degree)
        print(f"{degree:>6}  {result[0]:>22}  {result[1]:>27}")
        if result != expected:
            raise ArithmeticError(f"Expected {expected} in degree {degree}")
    print("All four diagonal entries are nonzero.")
