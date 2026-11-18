#!/usr/bin/env python3
"""
Independent check of the theorem on the Euclidean groups C^m x| SO_m, from the Lie algebra alone: no highest weight theory, no tensor product rule.

E_m is the stabilizer of the isotropic vector e in SO(E), E = C e + W + C f, with W of
dimension m.  With a Witt basis u_1, ..., u_r, (u_0), u_-1, ..., u_-r of W, the Lie
algebra of SO(E) is spanned by the operators X_{a,b}(x) = <x,b> a - <x,a> b, and that of
E_m by the X_{a,b} with a, b in W (the Lie algebra of SO(W)) and the translations
X_{e,w}, w in W.  The invariants of E_m in a module are the vectors of weight 0 for the
maximal torus of SO(W) which are killed by the simple root vectors of so(W) (this gives
the SO(W)-invariants, SO(W) being connected) and by all the translations.  The tensor
powers of X = E + W are expanded into words E^a W^b, and the kernels are computed
blockwise (weight 0, and degree for the grading e: 1, W: 0, f: -1) by exact sparse
elimination modulo the prime 2^31 - 1. Each resulting nullity gives an upper bound on
the characteristic-zero nullity. The character computations in euclidean_groups.py
give the characteristic-zero values.

Expected values (from weights.py, by highest weight theory):
  m = 2   1, 1, 4, 17, 60, 322, 1340, 7233, 33796, 182330, 909288   (no law, size 3)
  m = 3   1, 1, 3, 11, 48, 232, 1207, 6631, 37998     (a law, by the contraction theorem)
  m = 4   1, 1, 3, 10, 34, 197, 825, 5532, 29260      (no law at Hankel size 4)
  m = 5   1, 1, 3, 10, 33, 167, 733, 4272, 24068      (no law at Hankel size 3)
  m = 6   1, 1, 3, 10, 33, 166, 671, 4041, 20384
  m = 7   1, 1, 3, 10, 33, 166, 670, 3915, 19820

Run:   python3 independent_euclidean_groups_check.py          (budget 2 hours)
       python3 independent_euclidean_groups_check.py 0.5      (budget 0.5 hours)
The degrees are done in increasing order of cost, and each value is written to
independent-euclidean-groups-output.txt as soon as it is known; a degree which would not
finish within the budget is skipped.  Standard library only.
"""
import sys, os, time
from math import comb

P = 2147483647
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = open(os.path.join(HERE, "independent-euclidean-groups-output.txt"), "w")
EXPECTED = {
    2: [1, 1, 4, 17, 60, 322, 1340, 7233, 33796, 182330, 909288],
    3: [1, 1, 3, 11, 48, 232, 1207, 6631, 37998],
    4: [1, 1, 3, 10, 34, 197, 825, 5532, 29260],
    5: [1, 1, 3, 10, 33, 167, 733, 4272, 24068],
    6: [1, 1, 3, 10, 33, 166, 671, 4041, 20384],
    7: [1, 1, 3, 10, 33, 166, 670, 3915, 19820],
}


def say(s=""):
    print(s, flush=True)
    OUT.write(s + "\n")
    OUT.flush()


def euclidean(m):
    """The modules E and W of E_m, as dictionaries of sparse operators."""
    r = m // 2
    labels = ["e"] + ["u%d" % i for i in range(1, r + 1)] + (["u0"] if m % 2 else []) \
        + ["u-%d" % i for i in range(1, r + 1)] + ["f"]
    idx = dict((l, k) for k, l in enumerate(labels))

    def dual(l):
        if l == "e":
            return "f"
        if l == "f":
            return "e"
        if l == "u0":
            return "u0"
        return "u-" + l[1:] if not l.startswith("u-") else "u" + l[2:]

    def weight(l):
        w = [0] * r
        if l.startswith("u-"):
            w[int(l[2:]) - 1] = -1
        elif l.startswith("u") and l != "u0":
            w[int(l[1:]) - 1] = 1
        return tuple(w)

    def X(a, b):
        """X_{a,b}: dual(b) -> a, dual(a) -> -b, as {source: [(target, value)]}."""
        op = {}
        op.setdefault(idx[dual(b)], []).append((idx[a], 1))
        op.setdefault(idx[dual(a)], []).append((idx[b], -1))
        return op

    roots = [X("u%d" % i, "u-%d" % (i + 1)) for i in range(1, r)]
    if m % 2 == 0 and r >= 2:
        roots.append(X("u%d" % (r - 1), "u%d" % r))
    if m % 2 == 1:
        roots.append(X("u%d" % r, "u0"))
    wlabels = [l for l in labels if l not in ("e", "f")]
    trans = [X("e", w) for w in wlabels]
    ops = roots + trans
    wt = [weight(l) for l in labels]
    dg = [1 if l == "e" else -1 if l == "f" else 0 for l in labels]
    E = dict(d=len(labels), ops=ops, wt=wt, dg=dg)
    keep = [idx[l] for l in wlabels]
    pos = dict((b, i) for i, b in enumerate(keep))
    Wops = []
    for op in ops:
        new = {}
        for s, col in op.items():
            if s in pos:
                c = [(pos[t], v) for t, v in col if t in pos]
                if c:
                    new[pos[s]] = c
        Wops.append(new)
    W = dict(d=len(keep), ops=Wops, wt=[wt[b] for b in keep], dg=[0] * len(keep))
    return E, W, r


def invariants(factors, r):
    n = len(factors)
    if n == 0:
        return 1
    zero = (0,) * r
    blocks = {}
    stack = [(0, (), zero, 0)]
    # bound on the remaining weight, coordinatewise
    reach = [sum(1 for f in factors[i:]) for i in range(n)] + [0]
    while stack:
        i, t, w, g = stack.pop()
        if i == n:
            if w == zero:
                blocks.setdefault(g, []).append(t)
            continue
        f = factors[i]
        for b in range(f["d"]):
            w2 = tuple(x + y for x, y in zip(w, f["wt"][b]))
            if all(abs(x) <= reach[i + 1] for x in w2):
                stack.append((i + 1, t + (b,), w2, g + f["dg"][b]))
    nops = len(factors[0]["ops"])
    total = 0
    for g, cols in blocks.items():
        cols.sort()
        pivots = {}
        rank = 0
        for t in cols:
            v = {}
            for i in range(n):
                f = factors[i]
                b = t[i]
                for k in range(nops):
                    for target, val in f["ops"][k].get(b, ()):
                        key = (k,) + t[:i] + (target,) + t[i + 1:]
                        v[key] = (v.get(key, 0) + val) % P
            v = dict((k, x) for k, x in v.items() if x)
            while v:
                p = max(v)
                if p in pivots:
                    c = v[p]
                    for k, x in pivots[p].items():
                        y = (v.get(k, 0) - c * x) % P
                        if y:
                            v[k] = y
                        else:
                            v.pop(k, None)
                else:
                    inv = pow(v[p], P - 2, P)
                    pivots[p] = dict((k, x * inv % P) for k, x in v.items())
                    rank += 1
                    break
        total += len(cols) - rank
    return total


if __name__ == "__main__":
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
    deadline = time.time() + 3600 * hours
    say("Lie algebra kernels for the Euclidean groups C^m x| SO_m, module X = E + W")
    say("(exact, modulo 2^31 - 1; budget %.2f hours)" % hours)
    say("Modular nullities bound characteristic-zero dimensions from above.")
    groups = dict((m, euclidean(m)) for m in EXPECTED)
    last = {}
    done = dict((m, [1]) for m in EXPECTED)
    # all (m, n) pairs, cheapest first: the cost grows like (m+2)^n
    tasks = sorted(((m + 2) ** n, m, n) for m in EXPECTED for n in range(1, len(EXPECTED[m])))
    skipped = []
    for cost, m, n in tasks:
        if len(done[m]) != n:
            continue                       # a lower degree of this m was skipped
        if m in last and time.time() + 40 * last[m] > deadline:
            skipped.append((m, n))
            continue
        E, W, r = groups[m]
        t0 = time.time()
        c = sum(comb(n, a) * invariants([E] * a + [W] * (n - a), r) for a in range(n + 1))
        last[m] = time.time() - t0
        done[m].append(c)
        ok = "ok" if c == EXPECTED[m][n] else "MISMATCH"
        say("  m = %d, degree %d: c = %d   %s   (%.1f s)" % (m, n, c, ok, last[m]))
        assert ok == "ok", (m, n, c)
    say()
    for m in EXPECTED:
        say("  m = %d: c = %s" % (m, done[m]))
    if skipped:
        say("  not reached within the budget: %s" % ", ".join("m = %d degree %d" % s for s in skipped))
    say("all values computed agree with euclidean_groups.py")
    OUT.close()
