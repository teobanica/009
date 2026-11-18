#!/usr/bin/env python3
"""
Laws of invariant dimensions for the complex Euclidean-type groups C^n x| Spin_n
(stabilizers of an isotropic vector in Spin_{n+2}), and for Lambda^2(C^n) x| SL_n
(stabilizers of a highest weight vector of a half-spin module of Spin_2n).
For n=1 the script uses the identity component G_a of the stabilizer in Spin_3;
the full inverse image is G_a x mu_2, and has different spinor invariants.

For each group, the script tests a list of self-dual modules: products Y (x) W of
self-dual modules of Spin_{n+2} and Spin_n, and sums of 2 of these.  A module has a law
exactly when all its Hankel matrices are positive semidefinite (Hamburger), and the
script reports the first Hankel matrix which is not, with a witness polynomial.

Run:   /usr/bin/python3 orthogonal.py          (budget 5 hours)
       /usr/bin/python3 orthogonal.py 2.5      (budget 2.5 hours)
Results are written to orthogonal-output.txt as they come (safe to interrupt).
Needs weights.py in the same folder.  Standard library only.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weights import *


def spin_modules(f, prime=""):
    """Self-dual modules of a factor of type B or D."""
    mods = []
    if f.n == 0:
        return mods
    V = f.vector()
    mods.append(("V" + prime, V))
    if f.kind == 'B':
        mods.append(("S" + prime, f.spin()))
    elif f.n == 1:      # Spin_2, a torus: characters
        for j in (1, 3, 4):
            mods.append(("chi%d%s" % (j, prime), Counter({(j,): 1, (-j,): 1})))
    else:
        if f.n % 2 == 0:
            mods.append(("S+" + prime, f.spin(0)))
        mods.append(("S+" + prime + " + S-" + prime, plus(f.spin(0), f.spin(1))))
    if f.n >= 2:
        mods.append(("Lambda^2 V" + prime, ext(f, V, 2)))
    return mods


def groups(mmax):
    out = []
    for m in range(3, mmax + 1):
        st = spin_vector(m)
        h, l = st.h, st.lfactors[0]
        hmods = spin_modules(h)
        lmods = [(nm, (M,)) for nm, M in spin_modules(l, "'")]
        out.append((st, candidates(st, hmods, lmods)))
    for n in (5, 6):
        st = spinor(n)
        h, l = st.h, st.lfactors[0]
        hmods = spin_modules(h)
        U = l.vector()
        lmods = [("U + U*", (plus(U, dual(l, U)),)), ("adj", (minus_trivial(l, tensor(l, U, dual(l, U))),))]
        out.append((st, candidates(st, hmods, lmods)))
    return out


if __name__ == "__main__":
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    here = os.path.dirname(os.path.abspath(__file__))
    search(groups(10), os.path.join(here, "orthogonal-output.txt"), hours)
