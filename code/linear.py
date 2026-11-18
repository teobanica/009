#!/usr/bin/env python3
"""
Laws of invariant dimensions for the groups C^N x| SL_N (stabilizer of a vector in
SL_{N+1}), the affine groups C^N x| GL_N (stabilizer of a vector in GL_{N+1}),
Sym^2(C^n) x| SL_n (from the Siegel parabolic of Sp_2n), and the Grassmannian groups
M_{k,n-k} x| (SL_k x SL_{n-k}) (from SL_n).

For each group, the script tests a list of self-dual modules: products Y (x) W of
self-dual modules of the big group and of the Levi factor, some sums Y (x) W + Y* (x) W*,
and sums of 2 of these.  A module has a law exactly when all its Hankel matrices are
positive semidefinite (Hamburger), and the script reports the first Hankel matrix which
is not, with a witness polynomial.

Run:   /usr/bin/python3 linear.py          (budget 5 hours)
       /usr/bin/python3 linear.py 2.5      (budget 2.5 hours)
Results are written to linear-output.txt as they come (safe to interrupt).
Needs weights.py in the same folder.  Standard library only.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weights import *


def sd(f, M):
    """The self-dual module M + M*."""
    return plus(M, dual(f, M))


def sl_modules(f, prime=""):
    """Self-dual modules of SL_n or GL_n."""
    n, mods = f.n, []
    if n == 0:
        return mods
    U = f.vector()
    if f.kind == 'A' and n == 2:
        mods.append(("U" + prime, U))
    else:
        mods.append(("U%s + U%s*" % (prime, prime), sd(f, U)))
    if n >= 2:
        adj = tensor(f, U, dual(f, U))
        if f.kind == 'A':
            adj = minus_trivial(f, adj)
        mods.append(("adj" + prime if f.kind == 'A' else "U%s (x) U%s*" % (prime, prime), adj))
    for k in range(2, n // 2 + 1):
        E = ext(f, U, k)
        if f.kind == 'A' and 2 * k == n:
            mods.append(("Lambda^%d%s" % (k, prime), E))
        else:
            mods.append(("Lambda^%d%s + dual" % (k, prime), sd(f, E)))
    if f.kind == 'GL':
        for a in (1, 2):
            mods.append(("det^%d%s + det^-%d%s" % (a, prime, a, prime), plus(f.det(a), f.det(-a))))
    return mods


def cross(st, terms):
    """G-modules Y (x) W + Y* (x) W*, for (name, Y, W) with W one module per factor."""
    out = []
    for nm, Y, Ws in terms:
        D = tuple(dual(f, W) for f, W in zip(st.lfactors, Ws))
        out.append((nm + " + dual", outer_sum(st.module(Y, *Ws), st.module(dual(st.h, Y), *D))))
    return out


def groups():
    out = []
    # C^N x| SL_N
    for N in range(2, 8):
        st = sl_vector(N)
        h, l = st.h, st.lfactors[0]
        M, U = h.vector(), l.vector()
        hm = sl_modules(h)
        lm = [(nm, (W,)) for nm, W in sl_modules(l, "'")]
        ex = cross(st, [("M (x) U'*", M, (dual(l, U),)), ("M (x) U'", M, (U,)),
                        ("Lambda^2 M (x) U'*", ext(h, M, 2), (dual(l, U),))])
        out.append((st, candidates(st, hm, lm, ex)))
    # the affine groups C^N x| GL_N
    for N in range(1, 5):
        st = gl_vector(N)
        h, l = st.h, st.lfactors[0]
        M, U = h.vector(), l.vector()
        hm = sl_modules(h)
        lm = [(nm, (W,)) for nm, W in sl_modules(l, "'")]
        terms = [("M (x) U'*", M, (dual(l, U),))]
        for k in (1, 2, 3, 4):
            for b in (0, 1):
                Y = tensor(h, sym(h, M, k), h.det(-b)) if b else sym(h, M, k)
                for a in range(-2, 3):
                    terms.append(("Sym^%d M (x) det^-%d (x) det'^%d" % (k, b, a), Y, (l.det(a),)))
        out.append((st, candidates(st, hm, lm, cross(st, terms), pair_extra=False)))
    # Sym^2(C^n) x| SL_n
    for n in (2, 3, 4):
        st = sp_siegel(n)
        h, l = st.h, st.lfactors[0]
        V = h.vector()
        hm = [("V", V), ("Lambda^2 V", ext(h, V, 2)), ("Sym^2 V", sym(h, V, 2))]
        if n >= 3:
            hm.append(("Lambda^3 V", ext(h, V, 3)))
        lm = [(nm, (W,)) for nm, W in sl_modules(l, "'")]
        out.append((st, candidates(st, hm, lm)))
    # the Grassmannian groups
    for k, n in ((2, 5), (3, 6), (2, 6)):
        st = sl_grassmannian(k, n)
        h, l1, l2 = st.h, st.lfactors[0], st.lfactors[1]
        M = h.vector()
        hm = sl_modules(h)
        lm = [(nm, (W, l2.trivial())) for nm, W in sl_modules(l1, "'")]
        lm += [(nm, (l1.trivial(), W)) for nm, W in sl_modules(l2, "''")]
        U1, U2 = l1.vector(), l2.vector()
        ex = cross(st, [("1 (x) U' (x) U''*", h.trivial(), (U1, dual(l2, U2))),
                        ("M (x) U'* (x) 1", M, (dual(l1, U1), l2.trivial()))])
        out.append((st, candidates(st, hm, lm, ex)))
    return out


if __name__ == "__main__":
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    here = os.path.dirname(os.path.abspath(__file__))
    search(groups(), os.path.join(here, "linear-output.txt"), hours)
