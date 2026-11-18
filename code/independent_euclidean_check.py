#!/usr/bin/env python3
"""
Independent check of the Euclidean group computations, from the Lie algebra
alone: no highest weight theory, no tensor product rule.

E_3 = C^3 x| SO_3 is the stabilizer of the isotropic vector e in SO_5, acting on
C^5 = C e + W + C f.  Its Lie algebra is spanned by so(W) and the translations, and the
invariants of a module Z are the vectors of weight 0 (for the torus of SO_3) killed by
the raising operator e+ of so(W) and by the translation t_0: indeed the map t -> t.v
from the translations C^3 to Z is SO_3-equivariant for an SO_3-invariant v.  The
modules are built from
    A  = e^perp           (the restriction of the natural module M of SL_4, on which
                           E_3 acts through C^3 x| SL_3)
    A* = C^5 / C e        (the affine module of E_3)
    Lambda^2(A)           (the adjoint module of E_3)
    V  = C^5,  W = e^perp / C e.
The tensor powers are expanded into words in these modules, and the kernels are
computed blockwise (weight 0, and degree for the grading e: 1, W: 0, f: -1, which the
operators respect) by exact sparse elimination modulo the prime 2^31 - 1. Each resulting nullity gives an upper bound on
the characteristic-zero nullity. The character computations in euclidean_groups.py
give the characteristic-zero values.

Checks:
  E + W (m = 3, a law)       c_0..c_7 = 1, 1, 3, 11, 48, 232, 1207, 6631
  A + A*                     c_0..c_7 = 1, 1, 4, 11, 64, 246, 1600, 7309
  A + A* + Lambda^2(A)       c_0..c_8 = 1, 1, 8, 49, 380, 3233, 29520, 284623, 2859780
the last one being the module without a law of the theorem on the Euclidean groups.
Degree 8 of the last module is the long part (1 to 2 hours, below 1 GB of memory);
degree 7 takes about 1 minute.

Run:   python3 independent_euclidean_check.py        (all, through degree 8)
       python3 independent_euclidean_check.py 7      (through degree 7, about 2 minutes)
Output: independent-euclidean-output.txt.  Standard library only.
"""
import sys, os, time
from math import factorial

P = 2147483647
# basis of C^5: 0 = e, 1 = w1, 2 = w0, 3 = w-1, 4 = f, with <e,f> = <w1,w-1> = <w0,w0> = 1
EPLUS = {3: [(2, 1)], 2: [(1, -1)]}       # e+ : w-1 -> w0, w0 -> -w1
TZERO = {4: [(2, 1)], 2: [(0, -1)]}       # t0 : f -> w0, w0 -> -e
WT = [0, 1, 0, -1, 0]
DG = [1, 0, 0, 0, -1]
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = open(os.path.join(HERE, "independent-euclidean-output.txt"), "w")


def say(s=""):
    print(s, flush=True)
    OUT.write(s + "\n")
    OUT.flush()


def basic(name):
    """Sub- and quotient modules of C^5, as dictionaries of sparse matrices."""
    idx = {"V": [0, 1, 2, 3, 4], "A": [0, 1, 2, 3], "A*": [1, 2, 3, 4], "W": [1, 2, 3]}[name]
    pos = dict((b, i) for i, b in enumerate(idx))

    def op(M):
        out = {}
        for b in idx:
            col = [(pos[r], v) for r, v in M.get(b, []) if r in pos]
            if col:
                out[pos[b]] = col
        return out
    dg = [DG[b] for b in idx] if name != "W" else [0, 0, 0]
    return dict(d=len(idx), E=op(EPLUS), T=op(TZERO), wt=[WT[b] for b in idx], dg=dg)


def exterior_square(B):
    d = B["d"]
    pairs = [(i, j) for i in range(d) for j in range(i + 1, d)]
    pos = dict((p, k) for k, p in enumerate(pairs))

    def op(Mx):
        out = {}
        for k, (i, j) in enumerate(pairs):
            acc = {}
            for r, v in Mx.get(i, ()):
                if r != j:
                    a, b, s = (r, j, 1) if r < j else (j, r, -1)
                    acc[pos[(a, b)]] = acc.get(pos[(a, b)], 0) + s * v
            for r, v in Mx.get(j, ()):
                if r != i:
                    a, b, s = (i, r, 1) if i < r else (r, i, -1)
                    acc[pos[(a, b)]] = acc.get(pos[(a, b)], 0) + s * v
            col = [(r, v) for r, v in acc.items() if v]
            if col:
                out[k] = col
        return out
    return dict(d=len(pairs), E=op(B["E"]), T=op(B["T"]),
                wt=[B["wt"][i] + B["wt"][j] for i, j in pairs],
                dg=[B["dg"][i] + B["dg"][j] for i, j in pairs])


def invariants(factors):
    """Dimension of the invariants of the tensor product of the given modules."""
    n = len(factors)
    if n == 0:
        return 1
    blocks = {}
    reach = [sum(max(abs(w) for w in f["wt"]) for f in factors[i:]) for i in range(n)] + [0]
    stack = [(0, (), 0, 0)]
    while stack:
        i, t, w, g = stack.pop()
        if i == n:
            if w == 0:
                blocks.setdefault(g, []).append(t)
            continue
        f = factors[i]
        for b in range(f["d"]):
            w2 = w + f["wt"][b]
            if abs(w2) <= reach[i + 1]:
                stack.append((i + 1, t + (b,), w2, g + f["dg"][b]))
    total = 0
    for g, cols in blocks.items():
        cols.sort()                      # this order keeps the fill-in small
        pivots = {}
        rank = 0
        for t in cols:
            v = {}
            for i in range(n):
                f = factors[i]
                b = t[i]
                for which, M in ((0, f["E"]), (1, f["T"])):
                    for r, val in M.get(b, ()):
                        key = (which,) + t[:i] + (r,) + t[i + 1:]
                        v[key] = (v.get(key, 0) + val) % P
            v = dict((k, x) for k, x in v.items() if x)
            while v:
                r = max(v)
                if r in pivots:
                    c = v[r]
                    for k, x in pivots[r].items():
                        y = (v.get(k, 0) - c * x) % P
                        if y:
                            v[k] = y
                        else:
                            v.pop(k, None)
                else:
                    inv = pow(v[r], P - 2, P)
                    pivots[r] = dict((k, x * inv % P) for k, x in v.items())
                    rank += 1
                    break
        total += len(cols) - rank
    return total


def moments(types, nmax, expected):
    """types: list of modules X_1, ..., X_r; the moments of X_1 + ... + X_r."""
    r = len(types)
    c = [1]
    for n in range(1, nmax + 1):
        t0 = time.time()
        total = 0

        def comps(k, left):
            if k == r - 1:
                yield (left,)
                return
            for a in range(left + 1):
                for rest in comps(k + 1, left - a):
                    yield (a,) + rest
        for e in comps(0, n):
            coeff = factorial(n)
            word = []
            for k, a in enumerate(e):
                coeff //= factorial(a)
                word += [types[k]] * a
            total += coeff * invariants(word)
        c.append(total)
        ok = "ok" if n < len(expected) and expected[n] == total else "MISMATCH"
        say("    degree %d: c = %d   %s   (%.1f s)" % (n, total, ok, time.time() - t0))
        assert ok == "ok", (n, total)
    return c


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    A, Ad, V, W = basic("A"), basic("A*"), basic("V"), basic("W")
    L = exterior_square(A)
    say("Lie algebra kernels for the Euclidean group C^3 x| SO_3 (exact, modulo 2^31 - 1)")
    say("Modular nullities bound characteristic-zero dimensions from above.")
    say("  E + W, which has a law by the theorem on the Euclidean group of 3-space:")
    moments([V, W], min(nmax, 7), [1, 1, 3, 11, 48, 232, 1207, 6631])
    say("  A + A*, the restriction of M + M* from SL_4:")
    moments([A, Ad], min(nmax, 7), [1, 1, 4, 11, 64, 246, 1600, 7309])
    say("  A + A* + Lambda^2(A), the restriction of M + M* + Lambda^2(M) (no law):")
    moments([A, Ad, L], nmax, [1, 1, 8, 49, 380, 3233, 29520, 284623, 2859780])
    say("all values agree with euclidean_groups.py and contraction.py")
    OUT.close()
