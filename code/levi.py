#!/usr/bin/env python3
"""
Search for failures of Hamburger positivity among self-dual modules of
G = C^2 x| SL_2, realized as the stabilizer of e_1 in SL_3.

Modules: sums of words in  M = C^3 restricted to G,  D = its dual,  and 1,2,3 = the
modules V_1, V_2, V_3 of the Levi factor SL_2, inflated to G.  By the highest-weight vector stabilizer formula,
    dim (Y|_G (x) W)^G = sum_{a,b} [Y : V(a,b)] [W : V_b] .
For each module the script computes c_0, c_1, ... and the exact inertia of the
largest available Hankel matrix. A negative eigenvalue rules out a Hamburger
law. Positive finite truncations are finite certificates only.
Run:  python3 levi.py      Results are printed and written to levi-output.txt.
"""
import itertools
from collections import Counter
from hankel import first_non_psd_size, hankel_inertia, hankel_summary

def times_M(dec, dual):
    out = Counter()
    for (a, b, c), m in dec.items():
        for (x, y) in ([(a, b + 1), (a + 1, b - 1), (a - 1, b)] if dual else [(a + 1, b), (a - 1, b + 1), (a, b - 1)]):
            if x >= 0 and y >= 0: out[(x, y, c)] += m
    return out
def times_V(dec, d):
    out = Counter()
    for (a, b, c), m in dec.items():
        for e in range(abs(c - d), c + d + 1, 2): out[(a, b, e)] += m
    return out
def moments(words, N):
    dec = Counter({(0, 0, 0): 1}); c = [1]
    for _ in range(N):
        new = Counter()
        for w in words:
            cur = dec
            for ch in w: cur = times_M(cur, ch == 'D') if ch in 'MD' else times_V(cur, int(ch))
            new.update(cur)
        dec = new; c.append(sum(m for (a, b, cc), m in dec.items() if b == cc))
    return c
units = {'V1': ['1'], 'V2': ['2'], 'V3': ['3'], 'V1.V1': ['11'], 'M.M*': ['MD'], 'M+M*': ['M', 'D'],
         '(M+M*).V1': ['M1', 'D1'], '(M+M*).V2': ['M2', 'D2'], 'M.M+M*.M*': ['MM', 'DD'], 'M.M*.V1': ['MD1']}
out = open("levi-output.txt", "w")
def log(s):
    print(s); out.write(s + "\n"); out.flush()
found = []
for r in (1, 2):
    for combo in itertools.combinations(units, r):
        X = sum((units[u] for u in combo), [])
        N = {1: 12, 2: 10, 3: 8}[max(len(w) for w in X)]
        c = moments(X, N)
        log(f"{' + '.join(combo):28s} c = {c[:7]}  {hankel_summary(c)}")
        if hankel_inertia(c)[1]:
            bad = first_non_psd_size(c)
            found.append((bad, ' + '.join(combo), c[:2 * bad - 1]))
log("-" * 70)
log("modules without a Hamburger law (size of the first bad Hankel matrix, module, moments):")
for f in sorted(found): log(str(f))
out.close()
