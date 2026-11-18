#!/usr/bin/env python3
"""Independent exact Laurent-character checks for the Levi counterexamples.

The main verification scripts use Pieri decomposition. This script instead
expands tensor characters in the weight lattice and multiplies by the Weyl
denominator. The dominant terms give highest-weight multiplicities. It checks
the exterior square of SL_4 through degree 12 and M tensor M* tensor V for
the SL_3 vector stabilizer through degree 4. All arithmetic is integral.

Run python3 independent_levi_check.py. The two assertions and their exact
sequences are recorded in independent-levi-output.txt.
"""

from collections import Counter
import itertools


def multiply(first, second):
    result = Counter()
    for weight, multiplicity in first.items():
        for other, coefficient in second.items():
            combined = tuple(a + b for a, b in zip(weight, other))
            result[combined] += multiplicity * coefficient
    return Counter({weight: value for weight, value in result.items() if value})


def standard_weights(rank):
    return ([tuple(int(i == j) for j in range(rank - 1)) for i in range(rank - 1)]
            + [(-1,) * (rank - 1)])


def weyl_denominator(rank):
    weights = standard_weights(rank)
    zero = (0,) * (rank - 1)
    result = Counter({zero: 1})
    for i in range(rank):
        for j in range(i + 1, rank):
            negative_root = tuple(b - a for a, b in zip(weights[i], weights[j]))
            result = multiply(result, Counter({zero: 1, negative_root: -1}))
    return result


def exterior_square_moments(degree):
    weights = standard_weights(4)
    character = Counter(tuple(a + b for a, b in zip(weights[i], weights[j]))
                        for i, j in itertools.combinations(range(4), 2))
    power = Counter({(0, 0, 0): 1})
    denominator = weyl_denominator(4)
    moments = []
    for _ in range(degree + 1):
        highest_weights = multiply(power, denominator)
        moments.append(sum(multiplicity for (a, b, c), multiplicity in highest_weights.items()
                           if a >= 0 and b == c == 0))
        power = multiply(power, character)
    return moments


def sl3_stabilizer_moments(degree):
    weights = standard_weights(3)
    character = Counter(tuple(a - b for a, b in zip(first, second))
                        for first in weights for second in weights)
    power = Counter({(0, 0): 1})
    denominator = weyl_denominator(3)
    sl2_multiplicities = Counter({0: 1})
    moments = []
    for _ in range(degree + 1):
        highest_weights = multiply(power, denominator)
        moments.append(sum(multiplicity * sl2_multiplicities.get(b, 0)
                           for (a, b), multiplicity in highest_weights.items()
                           if a >= b >= 0))
        power = multiply(power, character)
        next_sl2 = Counter()
        for weight, multiplicity in sl2_multiplicities.items():
            next_sl2[weight + 1] += multiplicity
            if weight:
                next_sl2[weight - 1] += multiplicity
        sl2_multiplicities = next_sl2
    return moments


def main():
    exterior = exterior_square_moments(12)
    stabilizer = sl3_stabilizer_moments(4)
    assert exterior == [1, 0, 1, 1, 3, 10, 21, 105, 252, 1302, 3816, 19008, 65043]
    assert stabilizer == [1, 1, 4, 46, 356]
    lines = ["PASS SL_4 exterior square, c_0..c_12: " + str(exterior),
             "PASS SL_3 stabilizer M tensor M* tensor V, c_0..c_4: " + str(stabilizer),
             "ALL 2 INDEPENDENT CHARACTER CHECKS PASSED"]
    with open("independent-levi-output.txt", "w") as stream:
        for line in lines:
            print(line)
            stream.write(line + "\n")


if __name__ == "__main__":
    main()
