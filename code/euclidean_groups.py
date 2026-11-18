#!/usr/bin/env python3
"""
The complex Euclidean groups E_m = C^m x| SO_m.

1. For 2 <= m <= 10, with E the vector module of SO_{m+2} restricted to E_m and W the
   vector module of SO_m inflated to E_m: the mixed invariant dimensions
   d(a,b) = dim (E^a (x) W^b)^{E_m} for a + b <= 4, the moments of X = E + W, and the
   exact Hankel test.  For m >= 5 this gives (1,1,3,10,33) and the witness 2t^2 - 7t + 1.
2. For m = 3, with E_3 = C^3 x| SO_3 inside C^3 x| SL_3 inside SL_4: the 14-dimensional
   module M + M* + Lambda^2(M), its moments through degree 14, its Hankel determinants,
   and the witness p = t^4 - 20t^3 + 112t^2 - 107t - 178 with L(p^2) = -5682.

Run:   python3 euclidean_groups.py        (about 1 minute)
Output: euclidean-groups-output.txt.  Needs weights.py and euclidean_search.py.
"""
import sys, os
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weights import *
from euclidean_search import EuclideanInSL4

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = open(os.path.join(HERE, "euclidean-groups-output.txt"), "w")


def say(s=""):
    print(s, flush=True)
    OUT.write(s + "\n")
    OUT.flush()


def hankel_dets(c):
    out = []
    for k in range(1, (len(c) + 1) // 2 + 1):
        M = [[Fraction(c[i + j]) for j in range(k)] for i in range(k)]
        d = Fraction(1)
        for i in range(k):
            p = next((r for r in range(i, k) if M[r][i] != 0), None)
            if p is None:
                d = Fraction(0)
                break
            if p != i:
                M[i], M[p] = M[p], M[i]
                d = -d
            d *= M[i][i]
            for r in range(i + 1, k):
                f = M[r][i] / M[i][i]
                for s in range(i, k):
                    M[r][s] -= f * M[i][s]
        out.append(int(d))
    return out


def L_square(c, p):
    """L(p^2) for p = p[0] + p[1] t + ..."""
    return sum(p[i] * p[j] * c[i + j] for i in range(len(p)) for j in range(len(p)))


def part1():
    say("1. Euclidean groups C^m x| SO_m inside SO_{m+2}: X = E + W")
    stable = {(1, 0): 1, (0, 1): 0, (2, 0): 2, (1, 1): 0, (0, 2): 1, (3, 0): 4, (2, 1): 1,
              (1, 2): 1, (0, 3): 0, (4, 0): 10, (3, 1): 2, (2, 2): 2, (1, 3): 0, (0, 4): 3}
    expected = {2: [1, 1, 4, 17, 60], 3: [1, 1, 3, 11, 48, 232, 1207],
                4: [1, 1, 3, 10, 34, 197, 825]}
    for m in range(2, 11):
        st = spin_vector(m + 2)                # C^m x| Spin_m inside Spin_{m+2}
        h, l = st.h, st.lfactors[0]
        E, W = h.vector(), l.vector()
        d = {}
        for a in range(5):
            for b in range(5 - a):
                if a + b == 0:
                    continue
                Ea = h.trivial()
                for _ in range(a):
                    Ea = tensor(h, Ea, E)
                Wb = l.trivial()
                for _ in range(b):
                    Wb = tensor(l, Wb, W)
                d[(a, b)] = st.moments(st.module(Ea, Wb), 1)[1]
        X = outer_sum(st.module(E), st.module(h.trivial(), W))
        c = st.moments(X, 8 if m <= 4 else 6)
        rep = hankel_report(c)
        diff = sorted(k for k in stable if d[k] != stable[k])
        say("  m = %2d  d(a,b) = %s" % (m, " ".join("d%d%d=%d" % (a, b, d[(a, b)])
                                                   for (a, b) in sorted(d))))
        say("          c = %s,  Hankel determinants %s" % (c, hankel_dets(c)))
        say("          %s;  differences from the stable values: %s" % (
            ("NO LAW at Hankel size %d" % rep['first_bad']) if rep['first_bad'] else
            "positive through size %d" % rep['tested'], diff if diff else "none"))
        if m >= 5:
            assert not diff and c[:5] == [1, 1, 3, 10, 33], m
            assert L_square(c, [1, -7, 2]) == -2
        else:
            assert c[:len(expected[m])] == expected[m], (m, c)
    say("  m >= 5: (c_0,...,c_4) = (1,1,3,10,33) and L((2t^2 - 7t + 1)^2) = -2   ok")
    say()


def part2():
    say("2. C^3 x| SO_3 inside C^3 x| SL_3 inside SL_4: X = M + M* + Lambda^2(M)")
    st = EuclideanInSL4()
    h = st.h
    M = h.vector()
    X = st.module(plus(M, dual(h, M), ext(h, M, 2)))
    c = st.moments(X, 14)
    say("  c = %s" % c)
    D = hankel_dets(c)
    say("  Hankel determinants: %s" % D)
    assert c[:9] == [1, 1, 8, 49, 380, 3233, 29520, 284623, 2859780]
    assert D[:5] == [1, 7, 531, 390368, -2855167756]
    p = [-178, -107, 112, -20, 1]
    v = L_square(c, p)
    say("  p = t^4 - 20t^3 + 112t^2 - 107t - 178:  L(p^2) = %d" % v)
    assert v == -5682
    for nm, Y in [("M + M*", plus(M, dual(h, M))), ("Lambda^2(M)", ext(h, M, 2))]:
        cc = st.moments(st.module(Y), 14)
        r = hankel_report(cc)
        say("  for comparison %-12s c = %s ... %s" % (nm, cc[:9], "positive through size %d"
                                                       % r['tested'] if not r['first_bad'] else "NO LAW"))
    say()


if __name__ == "__main__":
    part1()
    part2()
    say("all assertions passed")
    OUT.close()
