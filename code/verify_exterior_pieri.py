#!/usr/bin/env python3
"""Check the restricted exterior-square dimensions by the GL(4) Pieri rule.

Run: python3 verify_exterior_pieri.py data/lambda2-172.json

If the rows of a Young diagram are r1>=r2>=r3>=r4, its SL(4) Dynkin
labels are (r1-r2, r2-r3, r3-r4). Tensoring by the exterior square
adds a box to each of two different rows. A negative difference is
the only reason for this addition to be inadmissible. The endpoint
(a,0,0) restricts to Sym^a of the defining module. Summing over these
endpoints counts the invariants under the stabilizer of the first vector in SL(4).
"""
import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path


def pieri_dimensions(max_degree):
    increments = []
    for i, j in combinations(range(4), 2):
        boxes = [0] * 4
        boxes[i] = boxes[j] = 1
        increments.append(tuple(boxes[k] - boxes[k + 1] for k in range(3)))

    layer = {(0, 0, 0): 1}
    yield 1
    for _ in range(max_degree):
        following = defaultdict(int)
        for (a, b, c), multiplicity in layer.items():
            for da, db, dc in increments:
                x, y, z = a + da, b + db, c + dc
                if x >= 0 and y >= 0 and z >= 0:
                    following[x, y, z] += multiplicity
        layer = following
        yield sum(m for (_, b, c), m in layer.items() if b == c == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path, help="the archived 173-entry JSON sequence")
    args = parser.parse_args()
    reference = [int(value) for value in json.loads(args.reference.read_text())]
    if len(reference) != 173:
        raise ValueError("Expected the 173 archived dimensions through degree 172")
    for degree, (actual, expected) in enumerate(zip(pieri_dimensions(172), reference, strict=True)):
        if actual != expected:
            raise ValueError(f"Degree {degree}: got {actual}, expected {expected}")
    print("PASS GL(4) vertical Pieri counts match the archive in every degree 0 through 172.")


if __name__ == "__main__":
    main()
