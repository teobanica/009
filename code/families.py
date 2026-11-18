#!/usr/bin/env python3
"""
All the families at once: the groups of linear.py and orthogonal.py, and the Siegel
parabolic Sym^2(C^2) x| GL_2 of Sp_4.  For each group the output gives the smallest
self-dual modules found without a law, or the finite depth reached with no obstruction.
The n=1 orthogonal entry denotes the connected stabilizer G_a.

Run:   /usr/bin/python3 families.py          (budget 5 hours)
       /usr/bin/python3 families.py 2.5      (budget 2.5 hours)
Results are written to families-output.txt as they come (safe to interrupt).
Needs weights.py, linear.py and orthogonal.py in the same folder.  Standard library only.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weights import *
import linear, orthogonal


def parabolic():
    st = sp_parabolic(2)
    h, l = st.h, st.lfactors[0]
    V = h.vector()
    hm = [("V", V), ("Lambda^2 V", ext(h, V, 2)), ("Sym^2 V", sym(h, V, 2))]
    lm = [(nm, (W,)) for nm, W in linear.sl_modules(l, "'")]
    terms = []
    for nm, Y in hm:
        for a in (1, 2, 3):
            terms.append(("%s (x) det'^%d" % (nm, a), Y, (l.det(a),)))
    return (st, candidates(st, hm, lm, linear.cross(st, terms), pair_extra=False))


if __name__ == "__main__":
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    here = os.path.dirname(os.path.abspath(__file__))
    gs = orthogonal.groups(10) + linear.groups() + [parabolic()]
    search(gs, os.path.join(here, "families-output.txt"), hours)
