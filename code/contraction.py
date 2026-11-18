#!/usr/bin/env python3
"""
Check of the theorem on the Euclidean group of 3-space: for the double cover
E_3 = C^3 x| SL_2 inside Sp_4 = Spin_5, a module Y|_{E_3} (x) W, with Y a Spin_5-module and
W an SL_2-module, has the same invariant dimensions as the module Y|_K (x) (W [x] C) of
K = Spin_4 = SL_2 x SL_2.  Hence every sum of such modules has a law, the distribution
of a character on SU(2) x SU(2) for the Haar measure.

For each test module the program computes c_0, ..., c_8 twice:
  - on E_3, with highest weight theory (weights.py, the highest-weight stabilizer formula),
  - on SU(2) x SU(2), as Haar integrals of powers of the transported character, where
    a weight (a, b) of Spin_5 (usual coordinates) becomes z^(a+b) w^(a-b), and the
    weight k of the Levi factor (doubled coordinates) becomes z^k.
It also checks c_{2n}(S) = C_n C_{n+1} for the 4-dimensional module S through degree 24.

Run:   python3 contraction.py        (a few minutes)
Output: contraction-output.txt.  Needs weights.py.
"""
import sys, os, itertools
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weights import *

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = open(os.path.join(HERE, "contraction-output.txt"), "w")


def say(s=""):
    print(s, flush=True)
    OUT.write(s + "\n")
    OUT.flush()


def mul(p, q):
    r = defaultdict(int)
    for (i, j), x in p.items():
        for (k, l), y in q.items():
            r[(i + k, j + l)] += x * y
    return {k: v for k, v in r.items() if v}


def haar(p):
    """Integral over SU(2) x SU(2) of a Weyl-invariant Laurent polynomial in z, w."""
    return p.get((0, 0), 0) - p.get((2, 0), 0) - p.get((0, 2), 0) + p.get((2, 2), 0)


def haar_moments(ch, n):
    out, p = [], {(0, 0): 1}
    for _ in range(n + 1):
        out.append(haar(p))
        p = mul(p, ch)
    return out


def transported(hmod, lmod):
    """Character on SU(2) x SU(2) of Y|_K (x) (W [x] C); weights of Spin_5 and of the Levi
    factor in the doubled coordinates of weights.py."""
    ch = defaultdict(int)
    for (X, Y), m in hmod.items():
        for (k,), n in lmod.items():
            ch[((X + Y) // 2 + k, (X - Y) // 2)] += m * n
    return dict(ch)


def catalan(n):
    from math import comb
    return comb(2 * n, n) // (n + 1)


if __name__ == "__main__":
    st = spin_vector(5)                      # C^3 x| Spin_3 inside Spin_5
    h, l = st.h, st.lfactors[0]
    hmods = []
    # irreducible Spin_5-modules: highest weights (X, Y) in doubled coordinates, X >= Y >= 0,
    # X = Y modulo 2
    for X in range(0, 6):
        for Y in range(0, X + 1):
            if (X - Y) % 2 == 0 and (X, Y) != (0, 0):
                M = irrep(h, (X, Y))
                if dimension(M) <= 35:
                    half = lambda x: str(x // 2) if x % 2 == 0 else "%d/2" % x
                    hmods.append(("V(%s,%s)" % (half(X), half(Y)), M))
    lmods = [("V_%d" % k, irrep(l, (k,))) for k in range(0, 5)]
    say("Euclidean group C^3 x| SL_2 inside Spin_5 versus SU(2) x SU(2)")
    say("Spin_5-modules: %s" % ", ".join("%s (dim %d)" % (nm, dimension(M)) for nm, M in hmods))
    tests = []
    for (hn, hm), (ln, lm) in itertools.product(hmods, lmods):
        tests.append(([(hm, lm)], "%s (x) %s" % (hn, ln)))
    for (hn, hm), (ln, lm) in itertools.product(hmods[:4], lmods[1:]):
        tests.append(([(hm, lm), (h.trivial(), lm)], "%s (x) %s + 1 (x) %s" % (hn, ln, ln)))
    for (h1, m1), (h2, m2) in itertools.combinations(hmods[:5], 2):
        tests.append(([(m1, lmods[0][1]), (m2, lmods[1][1])], "%s + %s (x) V_1" % (h1, h2)))
    count = 0
    for parts, name in tests:
        X = outer_sum(*[st.module(a, b) for a, b in parts])
        if dimension(X) > 60:
            continue
        ch = {}
        for a, b in parts:
            for k, v in transported(a, b).items():
                ch[k] = ch.get(k, 0) + v
        c1 = st.moments(X, 8)
        c2 = haar_moments(ch, 8)
        assert c1 == c2, (name, c1, c2)
        count += 1
        say("  ok  %-40s dim %2d  c = %s" % (name, dimension(X), c1))
    say("%d modules: invariant dimensions on E_3 = Haar moments on SU(2) x SU(2)" % count)
    S = h.spin()
    c = st.moments(st.module(S), 24)
    assert all(c[2 * n] == catalan(n) * catalan(n + 1) and c[2 * n + 1] == 0
               for n in range(12)) and c[24] == catalan(12) * catalan(13)
    say("the 4-dimensional module: c_2n = C_n C_(n+1) through degree 24   ok")
    say("all assertions passed")
    OUT.close()
