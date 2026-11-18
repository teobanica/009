#!/usr/bin/env python3
"""
The uniform witness for the vector stabilizers.  For G_N = C^N x| SL_N inside
SL_(N+1), and for the affine group C^N x| GL_N inside GL_(N+1), let
    X = M + M* + U + U*,
with M the natural module of SL_(N+1) or GL_(N+1), restricted, and U the natural module of
the Levi factor, inflated.  The program computes c_0, ..., c_10 by highest weight theory
(weights.py), for 2 <= N <= 12 and 1 <= N <= 12, and the exact Hankel determinants.  It
checks the 2 stability statements used in the proof of the theorem:
  - for the affine groups, c_n does not depend on N for N >= n,
  - for N > n, c_n is the same for C^N x| SL_N and for C^N x| GL_N,
and the signs: the Hankel determinant of size 5 is negative for N = 3 and N >= 5 (and
for the affine groups with N >= 2), and the one of size 6 for N = 2, 4.

Run:   python3 vector_stabilizers.py        (about 2 minutes)
Output: vector-stabilizers-output.txt.  Needs weights.py.  Standard library only.
"""
import sys, os
from fractions import Fraction
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weights import *

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = open(os.path.join(HERE, "vector-stabilizers-output.txt"), "w")
NMAX, DEG = 12, 10


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


def moments(st):
    h, l = st.h, st.lfactors[0]
    M, U = h.vector(), l.vector()
    X = outer_sum(st.module(plus(M, dual(h, M))), st.module(h.trivial(), plus(U, dual(l, U))))
    return st.moments(X, DEG)


if __name__ == "__main__":
    say("X = M + M* + U + U* on the vector stabilizers, moments c_0..c_%d" % DEG)
    c = {}
    for fam, Ns in (("SL", range(2, NMAX + 1)), ("GL", range(1, NMAX + 1))):
        for N in Ns:
            c[fam, N] = moments(sl_vector(N) if fam == "SL" else gl_vector(N))
            D = hankel_dets(c[fam, N])
            first = next((k + 1 for k, x in enumerate(D) if x <= 0), None)
            say("  C^%d x| %s_%d   c = %s" % (N, fam, N, c[fam, N]))
            say("      Hankel determinants %s;  first one <= 0: size %s" % (D, first))
            if fam == "SL":
                assert first == (6 if N in (2, 4) else 5)
            elif N >= 2:
                assert first == 5
    for n in range(DEG + 1):
        for N in range(max(n, 1), NMAX + 1):
            assert c["GL", N][n] == c["GL", NMAX][n], (n, N)
        for N in range(n + 1, NMAX + 1):
            if N >= 2:
                assert c["SL", N][n] == c["GL", N][n], (n, N)
    say("stability: c_n(GL, N) is constant for N >= n, and c_n(SL, N) = c_n(GL, N) for N > n")
    say("stable values: %s, Hankel determinants %s" % (c["GL", NMAX], hankel_dets(c["GL", NMAX])))
    say("all assertions passed")
    OUT.close()
