#!/usr/bin/env python3
"""Independently verify the degree-84 obstruction for the doubled exterior square.

Run: python3 verify_exterior.py
Optional comparison: python3 verify_exterior.py --compare PATH/TO/lambda2-172.json
Requires Python 3.8+ and SymPy. The computation uses integer arithmetic only.

The Weyl character formula expresses the multiplicity of the dominant weight
lambda in the nth tensor power of the SO(6) vector module as

    sum_{w in W(D3)} det(w) U_n(lambda + rho - w*rho),

where rho=(2,1,0), W(D3) consists of the even signed permutations (an even
number of sign changes), and U_n(v) is the number of unrestricted n-step
walks from 0 to v with steps ±e_i. Equivalently this is the chamber reflection
principle. See Grabiner and Magyar, J. Algebraic Combin. 2 (1993), 239–260.
The invariant dimension c_n sums these multiplicities over lambda=(j,j,j),
j>=0. The optional data file supplies no inputs to this computation.
"""

import argparse
import json
from collections import Counter
from itertools import permutations, product
from math import comb
from pathlib import Path

from sympy import ZZ
from sympy.polys.matrices import DomainMatrix


MAX_DEGREE = 172
MAX_SIZE = 43


def signed_rho_orbit():
    """Return the distinct w*rho and their Weyl signs."""
    rho = (2, 1, 0)
    orbit = []
    for perm in permutations(range(3)):
        inversions = sum(perm[i] > perm[j] for i in range(3) for j in range(i + 1, 3))
        sign = (-1) ** inversions
        for signs in product((-1, 1), repeat=3):
            if signs[0] * signs[1] * signs[2] == 1:
                image = tuple(signs[i] * rho[perm[i]] for i in range(3))
                orbit.append((image, sign))
    return orbit


def unrestricted_walks(n, endpoint, binomial):
    """Count walks to a nonnegative endpoint by conditioning on the third axis."""
    x, y, z = endpoint
    if x + y + z > n or (n - x - y - z) % 2:
        return 0
    result = 0
    for k in range(z, n - x - y + 1, 2):
        # Choose the k third-axis steps and their signs. In the remaining plane
        # the change (x,y) -> (x+y,x-y) gives two independent ±1 sequences.
        planar = n - k
        result += (
            binomial[n][k]
            * binomial[k][(k + z) // 2]
            * binomial[planar][(planar + x + y) // 2]
            * binomial[planar][(planar + x - y) // 2]
        )
    return result


def invariant_dimensions(max_degree):
    rho = (2, 1, 0)
    orbit = signed_rho_orbit()
    binomial = [[comb(n, k) for k in range(n + 1)] for n in range(max_degree + 1)]
    dimensions = []
    for n in range(max_degree + 1):
        weights = Counter()
        # Reaching (j,j,j) needs at least 3j steps with n and 3j of equal parity.
        for j in range(n % 2, n // 3 + 1, 2):
            for image, sign in orbit:
                # Unrestricted walk counts are invariant under all changes of
                # signs and permutations, so combine equivalent endpoints first.
                endpoint = tuple(sorted(abs(j + a - b) for a, b in zip(rho, image)))
                weights[endpoint] += sign
        dimensions.append(sum(
            weight * unrestricted_walks(n, endpoint, binomial)
            for endpoint, weight in weights.items() if weight
        ))
    return dimensions


def verify_hankel(name, moments, expected_digits=None):
    """Evaluate each determinant by a separate exact integer-matrix calculation."""
    determinants = []
    for size in range(1, MAX_SIZE + 1):
        matrix = [[moments[i + j] for j in range(size)] for i in range(size)]
        determinant = int(DomainMatrix.from_list(matrix, ZZ).det())
        if (size < MAX_SIZE and determinant <= 0) or (size == MAX_SIZE and determinant >= 0):
            raise ValueError(f"{name} has unexpected determinant sign at size {size}")
        determinants.append(determinant)
    last_digits = tuple(len(str(abs(value))) for value in determinants[-2:])
    if expected_digits is not None and last_digits != expected_digits:
        raise ValueError(f"{name} has digit counts {last_digits}, expected {expected_digits}")
    print(f"{name}: positive at sizes 1–42, negative at size 43.")
    if expected_digits is not None:
        print(f"  Sizes 42 and 43 have {last_digits[0]} and {last_digits[1]} digits.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compare", type=Path, help="optionally compare all counts with an archived JSON array")
    args = parser.parse_args()
    dimensions = invariant_dimensions(MAX_DEGREE)
    print(f"Computed c_0 through c_{MAX_DEGREE}. First 13 values: {dimensions[:13]}")
    if args.compare:
        reference = [int(value) for value in json.loads(args.compare.read_text())]
        if dimensions != reference:
            raise ValueError(f"The computed sequence differs from {args.compare}")
        print(f"All {len(dimensions)} dimensions agree with {args.compare.name}.")

    doubled = dimensions[::2]
    shifted = doubled[1:]
    upper_endpoint = [36 * value - next_value for value, next_value in zip(doubled, shifted)]
    verify_hankel("H = (d_(i+j))", doubled, (1526, 1603))
    verify_hankel("H' = (d_(i+j+1))", shifted, (1564, 1641))
    verify_hankel("36H - H'", upper_endpoint)


if __name__ == "__main__":
    main()
