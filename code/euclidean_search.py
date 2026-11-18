#!/usr/bin/env python3
"""
The Euclidean group E_3 = C^3 x| SO_3, embedded in C^3 x| SL_3 = the stabilizer of e_1
in SL_4.  For an SL_4-module V_mu, the invariants of E_3 in V_mu (x) V_k, with V_k an
irreducible SO_3-module inflated to E_3, are counted by highest weight theory:

    dim (V_mu (x) V_k)^{E_3} = multiplicity of V_k in the SL_3-module of highest weight
                               mu-bar, restricted to SO_3,

where mu-bar is the restriction of mu to SL_3 (the C^3 x| SL_3 lemma of the paper).  The
restriction from SL_3 to SO_3 is computed with Gelfand-Tsetlin patterns.  For k = 0 this
is the Cartan-Helgason rule: an invariant line exactly when mu-bar is even.

In this embedding E_3 has self-dual modules without a law.  The 14-dimensional example
M + M* + Lambda^2(M) fails at Hankel size 5.  The archived search also finds the
9-dimensional Lambda^2(M) + W, with W the inflated vector module of SO_3, failing at
size 7; it is the smallest dimension found among the tested candidates.  The script
tests modules V (x) V_k, with V = V_mu (+ its dual) of dimension at most 64
and k = 0, 1, 2, 3 (dimension 2k+1), and the sums of 2 of these, or of one of these and
some V_k alone, up to dimension 160.  At the end it lists the modules without a law,
the smallest Hankel sizes first.

Run:   /usr/bin/python3 euclidean_search.py          (budget 5 hours)
       /usr/bin/python3 euclidean_search.py 2.5      (budget 2.5 hours)
Results are written to euclidean-search-output.txt as they come (safe to interrupt).
Needs weights.py in the same folder.  Standard library only.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weights import *

_branch = {}


def so3_branching(nu):
    """{k: multiplicity of the SO_3-module of dimension 2k+1} in the SL_3-module of highest
    weight nu = (p1, p2, 0), for SO_3 = SO(q) with q = x1 x3 + x2^2 / 2."""
    if nu in _branch:
        return _branch[nu]
    p1, p2 = nu[0], nu[1]
    w = Counter()
    for x1 in range(p2, p1 + 1):
        for x2 in range(0, p2 + 1):
            z = p1 + p2 - x1 - x2
            for y in range(x2, x1 + 1):
                w[y - z] += 1                 # SO_3 weight of the vector, e1 -> 1, e2 -> 0, e3 -> -1
    res = {k: w[k] - w[k + 1] for k in range(0, max(w) + 1) if w[k] - w[k + 1]}
    _branch[nu] = res
    return res


class EuclideanInSL4(Stabilizer):
    def __init__(self):
        h, l = Factor('A', 4), Factor('B', 1)
        Stabilizer.__init__(self, "C^3 x| SO_3 in C^3 x| SL_3 in SL_4", h, h.unit(0), [l], None)

    def describe(self):
        return "%s: H = SL_4, Levi factor SO_3 (Spin_3 weights, doubled)" % self.name

    def invariants(self, state):
        total = 0
        for key, m in state.items():
            mu, k = key[0], key[1][0]
            if k % 2:
                continue
            nu = (mu[1] - mu[3], mu[2] - mu[3], 0)
            total += m * so3_branching(nu).get(k // 2, 0)
        return total


def sl4_dim(lam):
    l = list(lam) + [0] * (4 - len(lam))
    num = den = 1
    for i in range(4):
        for j in range(i + 1, 4):
            num *= l[i] - l[j] + j - i
            den *= j - i
    return num // den


def groups(maxh=64, maxdim=160):
    st = EuclideanInSL4()
    h, l = st.h, st.lfactors[0]
    seen, hmods = set(), []
    for a in range(0, 7):
        for b in range(0, 7):
            for c in range(0, 7):
                lam = (a + b + c, b + c, c, 0)
                if lam == (0, 0, 0, 0) or sl4_dim(lam) > maxh:
                    continue
                dl = (a + b + c, a + b, a, 0)          # highest weight of the dual
                if dl in seen:
                    continue
                seen.add(lam)
                M = irrep(h, lam)
                if dl == lam:
                    hmods.append(("V%s" % (lam[:3],), M))
                elif sl4_dim(lam) * 2 <= maxdim:
                    hmods.append(("V%s+dual" % (lam[:3],), plus(M, irrep(h, dl))))
    hmods.sort(key=lambda x: dimension(x[1]))
    lmods = [("V_%d'" % (2 * k + 1), (irrep(l, (2 * k,)),)) for k in range(1, 4)]
    cands = [c for c in candidates(st, hmods, lmods) if dimension(c[1]) <= maxdim]
    return [(st, cands)]


def selfcheck(say):
    st = EuclideanInSL4()
    h = st.h
    M = h.vector()
    for nu in [(2, 0, 0), (1, 1, 0), (2, 2, 0), (4, 2, 0), (3, 1, 0)]:
        even = nu[0] % 2 == 0 and nu[1] % 2 == 0
        assert (so3_branching(nu).get(0, 0) == 1) == even, nu
    X = st.module(plus(M, dual(h, M), ext(h, M, 2)))
    c = st.moments(X, 8)
    assert c == [1, 1, 8, 49, 380, 3233, 29520, 284623, 2859780], c
    say("self-check passed: M + M* + Lambda^2(M) has c = %s" % c)


if __name__ == "__main__":
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    here = os.path.dirname(os.path.abspath(__file__))
    selfcheck(print)
    search(groups(), os.path.join(here, "euclidean-search-output.txt"), hours, maxdim=160)
