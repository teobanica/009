#!/usr/bin/env python3
"""Check the integral bipartite fusion-graph obstruction.

Usage: python3 check_bipartite_census.py path/to/census-tables.json.gz
Requires SymPy. All data come directly from the supplied table file.
"""
import argparse
import gzip
import json
from collections import Counter, deque

from sympy import Matrix, Poly, Symbol


# The first count at each rank is the number of commutative tables.
# The second counts those with no nonunit integral bipartite basic element.
EXPECTED = [(1, 1), (2, 1), (4, 4), (10, 5), (16, 14),
            (37, 22), (40, 36), (77, 32), (116, 96)]


def bipartite(matrix):
    adjacency = [set() for row in matrix]
    for j, row in enumerate(matrix):
        for k, multiplicity in enumerate(row):
            if multiplicity:
                adjacency[j].add(k)
                adjacency[k].add(j)
    colors = {}
    for j in range(len(matrix)):
        if j in colors:
            continue
        colors[j] = 0
        queue = deque([j])
        while queue:
            source = queue.popleft()
            for target in adjacency[source]:
                if target in colors:
                    if colors[target] == colors[source]:
                        return False
                else:
                    colors[target] = 1 - colors[source]
                    queue.append(target)
    return True


def integral_perron_root(matrix):
    """Use the fact that a nonnegative matrix has a real Perron root.

    A candidate integer d is the largest real root of its characteristic
    polynomial p exactly when p(d)=0 and p(t+d) has nonnegative coefficients.
    Necessity follows by grouping the shifted complex roots in conjugate pairs
    with nonpositive real parts. Sufficiency follows by evaluating at t>0.
    """
    variable = Symbol("t")
    polynomial = Poly(Matrix(matrix).charpoly(variable).as_expr(), variable)
    for candidate in range(1, 1 + max(map(sum, matrix))):
        if polynomial.eval(candidate) != 0:
            continue
        if all(c >= 0 for c in polynomial.shift(candidate).all_coeffs()):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables")
    args = parser.parse_args()
    with gzip.open(args.tables, "rt") as source:
        records = json.load(source)
    total, excluded = Counter(), Counter()
    for record in records:
        table = record["table"]
        rank = len(table)
        if any(table[i][j] != table[j][i]
               for i in range(rank) for j in range(i)):
            continue
        total[rank] += 1
        if not any(bipartite(matrix) and integral_perron_root(matrix)
                   for matrix in table[1:]):
            excluded[rank] += 1
    print("rank  commutative  excluded")
    for rank in sorted(total):
        print(rank, total[rank], excluded[rank], sep="\t")
    print("All", sum(total.values()), sum(excluded.values()), sep="\t")
    actual = [(total[rank], excluded[rank]) for rank in range(1, 10)]
    if actual != EXPECTED or set(total) != set(range(1, 10)):
        raise ValueError("The rank totals differ from the published census table")


if __name__ == "__main__":
    main()
