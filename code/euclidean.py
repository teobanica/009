#!/usr/bin/env python3
"""
The group C^3 x| SL_2, with SL_2 acting on C^3 = Sym^2(C^2) by the adjoint action: the
double cover of the complex Euclidean group of R^3.  It is the stabilizer
of a highest weight vector in Sp_4 (Siegel parabolic), and also of an isotropic vector in
Spin_5.  Every module in the transported family tested here has a law by the paper's
compact-group formula.  This does not apply to all self-dual modules of the cover:
the counterexample for C^3 x| SO_3 inflates to this group with the same invariants.

The script tests the self-dual modules V_mu (x) V_k, with V_mu an irreducible module of
Sp_4 of dimension at most 40 and V_k the irreducible module of SL_2 of dimension k+1 <= 7,
and the sums of 2 of these (or of one of these and some 1 (x) V_k), up to dimension 160.

Run:   /usr/bin/python3 euclidean.py          (budget 5 hours)
       /usr/bin/python3 euclidean.py 2.5      (budget 2.5 hours)
Results are written to euclidean-output.txt as they come (safe to interrupt).
Needs weights.py in the same folder.  Standard library only.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weights import *


def groups():
    st = sp_siegel(2)
    h, l = st.h, st.lfactors[0]
    hmods = []
    for a in range(0, 5):
        for b in range(0, 5):
            if a == b == 0:
                continue
            mu = (a + b, b)                       # a omega_1 + b omega_2, in epsilon coordinates
            M = irrep(h, mu)
            if dimension(M) <= 40:
                hmods.append(("V%s" % (mu,), M))
    hmods.sort(key=lambda x: dimension(x[1]))
    lmods = [("V_%d'" % k, (irrep(l, (k, 0)),)) for k in range(1, 7)]
    cands = [c for c in candidates(st, hmods, lmods) if dimension(c[1]) <= 160]
    return [(st, cands)]


if __name__ == "__main__":
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    here = os.path.dirname(os.path.abspath(__file__))
    search(groups(), os.path.join(here, "euclidean-output.txt"), hours, maxdim=160)
