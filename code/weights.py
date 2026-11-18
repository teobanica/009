"""
Weights, tensor products and invariant dimensions for the stabilizers of highest weight
vectors.  Standard library only (works with the system Python 3.9 of macOS).

Setting.  H is a simply connected classical group, v a highest weight vector of weight
lam in an irreducible H-module, P the parabolic subgroup fixing the line of v, U_P its
unipotent radical, L its Levi factor and L' the semisimple part of L.  We study the
connected group

        G = U_P x| L'      (the identity component of the stabilizer of v).

For an H-module Y and an L'-module W, the U_P-invariants of the irreducible V_mu are the
irreducible L-module of highest weight mu, so that

        dim (Y|_G (x) W)^G = sum over mu, nu of [Y : V_mu] [W : V'_nu] [nu = (mu|L')^*].

We therefore decompose tensor powers of H x L'-modules into irreducibles, by the
Brauer-Klimyk rule, and count the pairs (mu, nu) with nu dual to the restriction of mu.

Coordinates.  Weights are integer tuples in the usual epsilon coordinates: SL_n (type A)
with n coordinates modulo (1,...,1), normalized with last coordinate 0; Sp_2n (type C)
with n coordinates; Spin_2n+1 and Spin_2n (types B and D) with n coordinates, doubled so
that spin weights are integers.
"""
import itertools, time
from collections import Counter
from fractions import Fraction


# ---------------------------------------------------------------------------------------
#  Simple factors
# ---------------------------------------------------------------------------------------
class Factor:
    """kind in 'A' (SL_n, n coordinates), 'GL' (GL_n), 'B' (Spin_2n+1), 'C' (Sp_2n),
    'D' (Spin_2n), n >= 0."""

    def __init__(self, kind, n):
        self.kind, self.n = kind, n
        if kind in ('A', 'GL'):
            self.rho = tuple(range(n - 1, -1, -1))
        elif kind == 'B':
            self.rho = tuple(2 * (n - i) - 1 for i in range(n))
        elif kind == 'C':
            self.rho = tuple(n - i for i in range(n))
        elif kind == 'D':
            self.rho = tuple(2 * (n - 1 - i) for i in range(n))
        else:
            raise ValueError(kind)
        self.zero = tuple([0] * n)

    def name(self):
        if self.n == 0:
            return "trivial"
        return {'A': "SL_%d" % self.n, 'GL': "GL_%d" % self.n, 'B': "Spin_%d" % (2 * self.n + 1),
                'C': "Sp_%d" % (2 * self.n), 'D': "Spin_%d" % (2 * self.n)}[self.kind]

    def normalize(self, w):
        if self.kind == 'A' and self.n:
            last = w[-1]
            return tuple(x - last for x in w)
        return tuple(w)

    def dot(self, lam):
        """Bring lam + rho into the dominant chamber: returns (sign, mu) with
        w(lam + rho) - rho = mu dominant, or (0, None) if lam + rho is singular."""
        n = self.n
        if n == 0:
            return 1, ()
        v = [a + b for a, b in zip(lam, self.rho)]
        k = self.kind
        if k in ('A', 'GL'):
            if len(set(v)) < n:
                return 0, None
            a, sign = v, 1
        elif k in 'BC':
            if 0 in v:
                return 0, None
            a = [abs(x) for x in v]
            if len(set(a)) < n:
                return 0, None
            sign = -1 if sum(1 for x in v if x < 0) % 2 else 1
        else:  # D
            a = [abs(x) for x in v]
            if len(set(a)) < n:
                return 0, None
            sign = 1
        inv = 0
        for i in range(n):
            ai = a[i]
            for j in range(i + 1, n):
                if ai < a[j]:
                    inv += 1
        if inv % 2:
            sign = -sign
        w = sorted(a, reverse=True)
        if k == 'D' and sum(1 for x in v if x < 0) % 2 and 0 not in v:
            w[-1] = -w[-1]
        mu = tuple(x - r for x, r in zip(w, self.rho))
        return sign, self.normalize(mu)

    def dual(self, mu):
        """Highest weight of the dual of the irreducible module of highest weight mu."""
        if self.n == 0:
            return ()
        if self.kind in ('A', 'GL'):
            return self.normalize(tuple(-x for x in reversed(mu)))
        if self.kind == 'D' and self.n % 2 == 1:
            return tuple(mu[:-1]) + (-mu[-1],)
        return tuple(mu)

    # ----- weights of basic modules (Counters of weight tuples) -----
    def unit(self, i, c=1):
        w = [0] * self.n
        w[i] = c
        return tuple(w)

    def trivial(self):
        return Counter({self.zero: 1})

    def vector(self):
        n, k = self.n, self.kind
        c = Counter()
        if k in ('A', 'GL'):
            for i in range(n):
                c[self.normalize(self.unit(i))] += 1
        elif k == 'C':
            for i in range(n):
                c[self.unit(i)] += 1
                c[self.unit(i, -1)] += 1
        else:  # B, D doubled
            for i in range(n):
                c[self.unit(i, 2)] += 1
                c[self.unit(i, -2)] += 1
            if k == 'B':
                c[self.zero] += 1
        return c

    def det(self, a):
        """The character det^a of GL_n."""
        return Counter({tuple([a] * self.n): 1})

    def spin(self, parity=None):
        """Spin module of type B, or half-spin module of type D (parity 0: even number of
        minus signs, parity 1: odd), in doubled coordinates."""
        c = Counter()
        for signs in itertools.product((1, -1), repeat=self.n):
            neg = sum(1 for s in signs if s < 0)
            if parity is None or neg % 2 == parity:
                c[tuple(signs)] += 1
        return c


# ---------------------------------------------------------------------------------------
#  Operations on weight multisets (Counters)
# ---------------------------------------------------------------------------------------
def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def expand(M):
    out = []
    for w, m in M.items():
        out.extend([w] * m)
    return out


def tensor(F, M1, M2):
    c = Counter()
    for a, m in M1.items():
        for b, k in M2.items():
            c[F.normalize(_add(a, b))] += m * k
    return c


def plus(*Ms):
    c = Counter()
    for M in Ms:
        c.update(M)
    return c


def dual(F, M):
    return Counter({F.normalize(tuple(-x for x in w)): m for w, m in M.items()})


def ext(F, M, k):
    c = Counter()
    for comb in itertools.combinations(expand(M), k):
        s = F.zero
        for w in comb:
            s = _add(s, w)
        c[F.normalize(s)] += 1
    return c


def sym(F, M, k):
    c = Counter()
    for comb in itertools.combinations_with_replacement(expand(M), k):
        s = F.zero
        for w in comb:
            s = _add(s, w)
        c[F.normalize(s)] += 1
    return c


def minus_trivial(F, M, k=1):
    c = Counter(M)
    c[F.zero] -= k
    assert c[F.zero] >= 0
    return +c


def dimension(M):
    return sum(M.values())


# ---------------------------------------------------------------------------------------
#  Products of factors: the group H x L'
# ---------------------------------------------------------------------------------------
class Group:
    """A product of simple factors; weights are tuples of factor weights."""

    def __init__(self, factors):
        self.factors = list(factors)
        self.zero = tuple(f.zero for f in self.factors)

    def dot(self, lam):
        sign, out = 1, []
        for f, l in zip(self.factors, lam):
            s, mu = f.dot(l)
            if s == 0:
                return 0, None
            sign *= s
            out.append(mu)
        return sign, tuple(out)

    def dual(self, mu):
        return tuple(f.dual(m) for f, m in zip(self.factors, mu))


def outer(*Ms):
    """External tensor product of weight multisets of the factors, in order."""
    c = Counter({(): 1})
    for M in Ms:
        new = Counter()
        for a, m in c.items():
            for b, k in M.items():
                new[a + (b,)] += m * k
        c = new
    return c


def outer_sum(*terms):
    c = Counter()
    for t in terms:
        c.update(t)
    return c


def decompose(group, M):
    """Irreducible decomposition of a module given by its weights (Weyl character formula,
    via the dot action of the Weyl group on all weights)."""
    # Brauer-Klimyk with the trivial module: V_0 (x) M
    return tensor_irreducible(group, Counter({group.zero: 1}), list(M.items()))


def tensor_irreducible(group, state, items):
    """state: Counter {highest weight: multiplicity}; items: weights of a module.
    Returns the decomposition of (sum state) (x) module, by the Brauer-Klimyk rule."""
    new = Counter()
    nf = len(group.factors)
    for mu, m in state.items():
        for a, k in items:
            lam = tuple(_add(mu[i], a[i]) for i in range(nf))
            s, nu = group.dot(lam)
            if s:
                new[nu] += s * m * k
    return +Counter({k: v for k, v in new.items() if v})


# ---------------------------------------------------------------------------------------
#  The stabilizer G = U_P x| L' and its invariant dimensions
# ---------------------------------------------------------------------------------------
class Stabilizer:
    """H = Group([h]) one simple factor, Lp = list of factors of L', restrict(mu) the highest
    weight of the L'-module V^L_mu, as a tuple of factor weights."""

    def __init__(self, name, h, lam, lfactors, restrict):
        self.name, self.h, self.lam = name, h, lam
        self.lfactors = list(lfactors)
        self.group = Group([h] + self.lfactors)
        self.lgroup = Group(self.lfactors)
        self.restrict = restrict

    def describe(self):
        lp = " x ".join(f.name() for f in self.lfactors) if self.lfactors else "trivial"
        return "%s: H = %s, lambda = %s, L' = %s" % (self.name, self.h.name(), self.lam, lp)

    def invariants(self, state):
        total = 0
        for key, m in state.items():
            mu, nus = key[0], key[1:]
            if self.lgroup.dual(self.restrict(mu)) == nus:
                total += m
        return total

    def module(self, hmod, *lmods):
        """The G-module Y|_G (x) W for an H-module Y and L'-modules W (one per factor)."""
        if not self.lfactors:
            return outer(hmod)
        if not lmods:
            lmods = [f.trivial() for f in self.lfactors]
        return outer(hmod, *lmods)

    def moments(self, X, nmax, deadline=None, cap=None, log=None, status=None):
        """c_0, ..., c_nmax for the G-module X (weights on H x L').  Stops before a degree
        whose predicted running time would overshoot the deadline (a time.time() value),
        or whose predicted number of irreducibles would exceed cap.  If supplied, status
        records the target, attained degree and stopping reason without changing counts."""
        items = list(X.items())
        state = Counter({self.group.zero: 1})
        c = [1]
        times, sizes = [], [1]
        reason = 'target_reached'
        for n in range(1, nmax + 1):
            if len(times) >= 2 and deadline is not None:
                growth = max(1.0, times[-1] / max(times[-2], 1e-6))
                if time.time() + times[-1] * growth > deadline:
                    reason = 'predicted_time_limit'
                    break
            if cap and len(sizes) >= 2:
                growth = max(1.0, sizes[-1] / max(sizes[-2], 1))
                if sizes[-1] * growth > cap:
                    reason = 'predicted_size_limit'
                    break
            t0 = time.time()
            state = tensor_irreducible(self.group, state, items)
            times.append(time.time() - t0)
            sizes.append(len(state))
            c.append(self.invariants(state))
            if log:
                log(n, c[-1], len(state))
            if deadline is not None and time.time() > deadline:
                reason = 'time_limit' if n < nmax else 'target_reached'
                break
        if status is not None:
            status.update(target_degree=nmax, attained_degree=len(c) - 1,
                          stop_reason=reason)
        return c


# ---------------------------------------------------------------------------------------
#  Exact Hankel tests
# ---------------------------------------------------------------------------------------
def inertia(Mat):
    """Exact inertia (positive, negative, zero) of a rational symmetric matrix, by
    congruence (2 x 2 pivots when the diagonal vanishes)."""
    A = [[Fraction(x) for x in row] for row in Mat]
    pos = neg = 0
    while A:
        n = len(A)
        p = next((i for i in range(n) if A[i][i] != 0), None)
        if p is not None:
            d = A[p][p]
            if d > 0:
                pos += 1
            else:
                neg += 1
            rest = [i for i in range(n) if i != p]
            A = [[A[i][j] - A[i][p] * A[p][j] / d for j in rest] for i in rest]
            continue
        q = next(((i, j) for i in range(n) for j in range(i + 1, n) if A[i][j] != 0), None)
        if q is None:
            return pos, neg, n
        i, j = q
        pos += 1
        neg += 1
        # eliminate the hyperbolic plane spanned by e_i, e_j
        a = A[i][j]
        rest = [k for k in range(n) if k not in (i, j)]
        B = []
        for k in rest:
            row = []
            for l in rest:
                # Schur complement of [[0,a],[a,0]]
                row.append(A[k][l] - (A[k][i] * A[j][l] + A[k][j] * A[i][l]) / a)
            B.append(row)
        A = B
    return pos, neg, 0


def hankel(c, k):
    return [[c[i + j] for j in range(k)] for i in range(k)]


def hankel_report(c):
    """Test the leading Hankel matrices of sizes 1, ..., (len(c)+1)//2.  Returns a dict
    with the largest size tested, the first size which is not positive semidefinite (or
    None), its inertia, and when possible a witness: the monic orthogonal polynomial p of
    the previous size, with L(p^2) < 0."""
    kmax = (len(c) + 1) // 2
    out = {'tested': kmax, 'first_bad': None, 'inertia': None, 'witness': None,
           'definite_up_to': 0, 'singular_at': None}
    for k in range(1, kmax + 1):
        p, n, z = inertia(hankel(c, k))
        if n > 0:
            out['first_bad'] = k
            out['inertia'] = (p, n, z)
            if out['definite_up_to'] == k - 1:
                out['witness'] = witness(c, k)
            return out
        if z == 0 and out['definite_up_to'] == k - 1:
            out['definite_up_to'] = k
        elif z > 0 and out['singular_at'] is None:
            out['singular_at'] = k
    return out


def witness(c, k):
    """For H_{k-1} positive definite and H_k not: the monic polynomial p of degree k-1
    orthogonal to lower degrees, and L(p^2) < 0."""
    m = k - 1
    A = [[Fraction(c[i + j]) for j in range(m)] for i in range(m)]
    b = [Fraction(c[i + m]) for i in range(m)]
    # solve A x = -b
    M = [A[i][:] + [-b[i]] for i in range(m)]
    for i in range(m):
        piv = next(r for r in range(i, m) if M[r][i] != 0)
        M[i], M[piv] = M[piv], M[i]
        for r in range(m):
            if r != i and M[r][i] != 0:
                f = M[r][i] / M[i][i]
                M[r] = [M[r][j] - f * M[i][j] for j in range(m + 1)]
    x = [M[i][m] / M[i][i] for i in range(m)]
    p = x + [Fraction(1)]
    val = sum(p[i] * p[j] * c[i + j] for i in range(k) for j in range(k))
    return p, val


def poly_string(p):
    terms = []
    for i in range(len(p) - 1, -1, -1):
        a = p[i]
        if a == 0:
            continue
        s = str(a)
        if i == 0:
            terms.append(s)
        elif a == 1:
            terms.append("t^%d" % i if i > 1 else "t")
        elif a == -1:
            terms.append("-t^%d" % i if i > 1 else "-t")
        else:
            terms.append("%s t^%d" % (s, i) if i > 1 else "%s t" % s)
    return " + ".join(terms).replace("+ -", "- ")


# ---------------------------------------------------------------------------------------
#  Running a list of cases
# ---------------------------------------------------------------------------------------
def run(cases, outfile, hours, nmax, cap):
    """cases: list of (Stabilizer, [(module name, module X), ...]).  The time budget is
    shared among the modules, unused time passing on to the next ones."""
    out = open(outfile, "w")

    def say(s=""):
        print(s, flush=True)
        out.write(s + "\n")
        out.flush()

    start = time.time()
    end = start + 3600 * hours
    todo = [(st, nm, X) for st, mods in cases for nm, X in mods]
    say("Invariant dimensions for stabilizers of highest weight vectors")
    say("started %s, budget %.1f hours, max degree %d, cap %d irreducibles"
        % (time.strftime("%Y-%m-%d %H:%M"), hours, nmax, cap))
    say("=" * 78)
    current = None
    for idx, (st, nm, X) in enumerate(todo):
        if st is not current:
            current = st
            say()
            say(st.describe())
            say("-" * 78)
        remaining = len(todo) - idx
        deadline = time.time() + max(0.0, end - time.time()) / remaining
        t0 = time.time()
        dimX = dimension(X)
        c = st.moments(X, nmax, deadline=deadline, cap=cap)
        rep = hankel_report(c)
        say("%s   (dimension %d)" % (nm, dimX))
        say("  c_0..c_%d = %s" % (len(c) - 1, c))
        if rep['first_bad']:
            say("  NO LAW: Hankel matrix of size %d is not positive semidefinite, inertia %s"
                % (rep['first_bad'], rep['inertia']))
            if rep['witness']:
                p, val = rep['witness']
                say("  witness p = %s, L(p^2) = %s" % (poly_string(p), val))
        else:
            extra = ""
            if rep['singular_at']:
                extra = ", singular from size %d on (finitely supported law?)" % rep['singular_at']
            say("  positive semidefinite through size %d (positive definite through size %d)%s"
                % (rep['tested'], rep['definite_up_to'], extra))
        say("  [%.1f s]" % (time.time() - t0))
        if time.time() > end:
            say("time budget exhausted")
            break
    say()
    say("finished %s, total %.2f hours" % (time.strftime("%Y-%m-%d %H:%M"), (time.time() - start) / 3600))
    out.close()


# ---------------------------------------------------------------------------------------
#  The families of stabilizers
# ---------------------------------------------------------------------------------------
def sl_vector(N):
    """C^N x| SL_N, the stabilizer of e_1 in SL_{N+1} (lambda = omega_1)."""
    h, l = Factor('A', N + 1), Factor('A', N)
    return Stabilizer("C^%d x| SL_%d" % (N, N), h, h.normalize(h.unit(0)), [l] if N > 0 else [],
                      (lambda mu: (l.normalize(mu[1:]),)) if N > 0 else (lambda mu: ()))


def gl_vector(N):
    """The affine group C^N x| GL_N, the stabilizer of e_1 in GL_{N+1}."""
    h, l = Factor('GL', N + 1), Factor('GL', N)
    return Stabilizer("C^%d x| GL_%d" % (N, N), h, h.unit(0), [l], lambda mu: (tuple(mu[1:]),))


def spinor(n):
    """Lambda^2(C^n) x| SL_n, from Spin_2n and lambda = the highest weight of S+."""
    h, l = Factor('D', n), Factor('A', n)
    return Stabilizer("Lambda^2(C^%d) x| SL_%d" % (n, n), h, tuple([1] * n), [l],
                      lambda mu: (tuple((x - mu[-1]) // 2 for x in mu),))


def sl_grassmannian(k, n):
    """M_{k,n-k} x| (SL_k x SL_{n-k}), from SL_n and lambda = omega_k."""
    h, l1, l2 = Factor('A', n), Factor('A', k), Factor('A', n - k)
    lam = h.normalize(tuple([1] * k + [0] * (n - k)))
    return Stabilizer("M_%d,%d x| (SL_%d x SL_%d)" % (k, n - k, k, n - k), h, lam, [l1, l2],
                      lambda mu: (l1.normalize(mu[:k]), l2.normalize(mu[k:])))


def sp_siegel(n):
    """Sym^2(C^n) x| SL_n, from Sp_2n and lambda = omega_n (Siegel parabolic)."""
    h, l = Factor('C', n), Factor('A', n)
    return Stabilizer("Sym^2(C^%d) x| SL_%d" % (n, n), h, tuple([1] * n), [l],
                      lambda mu: (l.normalize(mu),))


def spin_vector(m):
    """Isotropic-vector stabilizer in Spin_m for m >= 4; for m=3 its component G_a.

    In Spin_3 the full inverse image is G_a x mu_2.  The rank-zero Levi factor used
    here is trivial, so the engine computes G_a invariants, including one spinor
    invariant; it does not impose invariance under the covering involution.
    """
    n = m // 2
    kind = 'B' if m % 2 else 'D'
    h, l = Factor(kind, n), Factor(kind, n - 1)
    name = "G_a (identity component)" if m == 3 else "C^%d x| Spin_%d" % (m - 2, m - 2)
    return Stabilizer(name, h, h.unit(0, 2), [l],
                      lambda mu: (tuple(mu[1:]),))


# ---------------------------------------------------------------------------------------
#  Self-tests, run at the start of each script
# ---------------------------------------------------------------------------------------
def selftest(say):
    def dec(F, M):
        d = decompose(Group([F]), Counter({(w,): m for w, m in M.items()}))
        return sorted((k[0], v) for k, v in d.items())

    def check(label, got, expected):
        if got != expected:
            raise AssertionError("%s: got %s, expected %s" % (label, got, expected))
        say("  ok  %s" % label)

    say("Self-tests")
    A3, C2, B2, D3, B3 = Factor('A', 3), Factor('C', 2), Factor('B', 2), Factor('D', 3), Factor('B', 3)
    V = A3.vector()
    check("SL_3: 3 x 3 = 6 + 3*", dec(A3, tensor(A3, V, V)), [((1, 1, 0), 1), ((2, 0, 0), 1)])
    check("Sp_4: 4 x 4 = 1 + 5 + 10", dec(C2, tensor(C2, C2.vector(), C2.vector())),
          [((0, 0), 1), ((1, 1), 1), ((2, 0), 1)])
    check("Spin_5: 5 x 5 = 1 + 10 + 14", dec(B2, tensor(B2, B2.vector(), B2.vector())),
          [((0, 0), 1), ((2, 2), 1), ((4, 0), 1)])
    check("Spin_6: S+ x S- = 1 + 15", dec(D3, tensor(D3, D3.spin(0), D3.spin(1))),
          [((0, 0, 0), 1), ((2, 2, 0), 1)])
    check("Spin_7: 8 x 8 = 1 + 7 + 21 + 35", dec(B3, tensor(B3, B3.spin(), B3.spin())),
          [((0, 0, 0), 1), ((2, 0, 0), 1), ((2, 2, 0), 1), ((2, 2, 2), 1)])
    # the additive group, from SL_2: J_2 has moments binomial(n, [n/2])
    st = sl_vector(1)
    check("G_a, J_2: c_0..c_8", st.moments(st.module(st.h.vector()), 8), [1, 1, 2, 3, 6, 10, 20, 35, 70])
    # the 2 examples of the paper
    st = sl_vector(2)
    M = st.h.vector()
    MM = tensor(st.h, M, dual(st.h, M))
    check("C^2 x| SL_2, M + M*: c_0..c_6", st.moments(st.module(plus(M, dual(st.h, M))), 6),
          [1, 1, 4, 12, 48, 192, 832])
    check("C^2 x| SL_2, M x M* x V: c_0..c_4", st.moments(st.module(MM, st.lfactors[0].vector()), 4),
          [1, 1, 4, 46, 356])
    st = sl_vector(3)
    check("C^3 x| SL_3, Lambda^2(C^4): c_0..c_6", st.moments(st.module(ext(st.h, st.h.vector(), 2)), 6),
          [1, 0, 1, 1, 3, 10, 21])
    # the same group, twice: Sym^2(C^2) x| SL_2 = C^3 x| Spin_3
    a, b = sp_siegel(2), spin_vector(5)
    W, S = a.lfactors[0].vector(), b.lfactors[0].spin()
    check("Sp_4 and Spin_5 agree: 4 and spin module",
          a.moments(a.module(a.h.vector()), 8), b.moments(b.module(b.h.spin()), 8))
    check("Sp_4 and Spin_5 agree: 4 x 2 and spin x spin",
          a.moments(a.module(a.h.vector(), W), 6), b.moments(b.module(b.h.spin(), S), 6))
    # and M_{2,2} x| (SL_2 x SL_2) = C^4 x| Spin_4
    a, b = sl_grassmannian(2, 4), spin_vector(6)
    check("SL_4 and Spin_6 agree: Lambda^2(C^4) and vector",
          a.moments(a.module(ext(a.h, a.h.vector(), 2)), 8), b.moments(b.module(b.h.vector()), 8))
    say("all self-tests passed")


# ---------------------------------------------------------------------------------------
#  Systematic search for self-dual modules without a law
# ---------------------------------------------------------------------------------------
def candidates(st, hmods, lmods, extra=(), pair_extra=True):
    """Self-dual G-modules built from self-dual H-modules hmods [(name, weights)], self-dual
    L'-modules lmods [(name, tuple of factor weights)] and extra G-modules [(name, X)]:
    all products Y (x) W with Y nontrivial, and all sums of 2 of these, or of one of these
    and a module 1 (x) W inflated from L'.  With pair_extra=False the extra modules are
    tested alone, not in sums."""
    th = st.h.trivial()
    tl = tuple(f.trivial() for f in st.lfactors)
    singles = []
    for hn, hm in hmods:
        for ln, lm in [("1", tl)] + list(lmods):
            name = hn if ln == "1" else "%s (x) %s" % (hn, ln)
            singles.append((name, st.module(hm, *lm)))
    if pair_extra:
        singles += list(extra)
    lonly = [("1 (x) " + ln, st.module(th, *lm)) for ln, lm in lmods]
    pool = singles + lonly
    pairs = []
    for i in range(len(pool)):
        for j in range(i + 1, len(pool)):
            if i >= len(singles) and j >= len(singles):
                continue
            pairs.append(("%s  +  %s" % (pool[i][0], pool[j][0]), outer_sum(pool[i][1], pool[j][1])))
    if not pair_extra:
        singles += list(extra)
    return singles + pairs


def search(groups, outfile, hours, nmax=40, cap=1000000, maxdim=400, first=12):
    """groups: list of (Stabilizer, candidate list).  For each group, pass 1 targets
    degree `first`; time and size limits can stop a candidate sooner.  If no candidate
    fails, pass 2 targets `nmax` and pushes the candidates as
    far as the time allows.  The time budget is shared among the groups, unused time
    passing on to the next ones.  Everything is written to outfile as it comes."""
    out = open(outfile, "w")

    def say(s=""):
        print(s, flush=True)
        out.write(s + "\n")
        out.flush()

    def line(nm, X, c, rep, t, status):
        say("%s   [dimension %d, %.1f s]" % (nm, dimension(X), t))
        say("    c = %s" % c)
        if rep['first_bad']:
            s = "    NO LAW: the Hankel matrix of size %d has inertia %s" % (rep['first_bad'], rep['inertia'])
            if rep['witness']:
                p, val = rep['witness']
                s += "; p = %s gives L(p^2) = %s" % (poly_string(p), val)
            say(s)
        else:
            s = "    positive semidefinite through size %d, definite through size %d" % (
                rep['tested'], rep['definite_up_to'])
            if rep['singular_at']:
                s += ", singular from size %d" % rep['singular_at']
            say(s)
        say("    attained degree %d of target %d; stop: %s" % (
            status['attained_degree'], status['target_degree'], status['stop_reason']))

    start = time.time()
    end = start + 3600.0 * hours
    say("Search for self-dual modules without a law, for stabilizers of highest weight vectors")
    say("started %s; budget %.2f hours, degrees up to %d, at most %d irreducibles, dimension <= %d"
        % (time.strftime("%Y-%m-%d %H:%M"), hours, nmax, cap, maxdim))
    say("=" * 90)
    selftest(say)
    summary = []
    for gi, (st, cands) in enumerate(groups):
        gend = time.time() + max(0.0, end - time.time()) / (len(groups) - gi)
        cands = sorted([cd for cd in cands if dimension(cd[1]) <= maxdim], key=lambda cd: dimension(cd[1]))
        say()
        say("=" * 90)
        say(st.describe())
        say("%d candidate modules" % len(cands))
        say("=" * 90)
        results = {}
        p1end = time.time() + 0.5 * max(0.0, gend - time.time())
        say("Pass 1, degrees up to %d" % first)
        for i, (nm, X) in enumerate(cands):
            dl = time.time() + max(0.0, p1end - time.time()) / (len(cands) - i)
            t0 = time.time()
            status = {}
            c = st.moments(X, first, deadline=dl, cap=cap, status=status)
            rep = hankel_report(c)
            results[nm] = (c, rep, X)
            line(nm, X, c, rep, time.time() - t0, status)
        bad = [(results[nm][1]['first_bad'], dimension(results[nm][2]), nm) for nm in results
               if results[nm][1]['first_bad']]
        if not bad and cands:
            say()
            say("Pass 2, degrees up to %d" % nmax)
            for i, (nm, X) in enumerate(cands):
                dl = time.time() + max(0.0, gend - time.time()) / (len(cands) - i)
                t0 = time.time()
                status = {}
                c = st.moments(X, nmax, deadline=dl, cap=cap, status=status)
                if len(c) > len(results[nm][0]):
                    rep = hankel_report(c)
                    results[nm] = (c, rep, X)
                    line(nm, X, c, rep, time.time() - t0, status)
                    if rep['first_bad']:
                        bad.append((rep['first_bad'], dimension(X), nm))
        bad.sort()
        say()
        if bad:
            say("SUMMARY for %s: %d modules without a law; first ones (Hankel size, dimension):"
                % (st.name, len(bad)))
            for b in bad[:5]:
                say("    size %d, dimension %d: %s" % b)
            summary.append((st.name, len(cands), len(bad), "no law for %s (size %d, dim %d)" % (bad[0][2], bad[0][0], bad[0][1])))
        else:
            sizes = [results[nm][1]['tested'] for nm in results] or [0]
            low, high = min(sizes), max(sizes)
            say("SUMMARY for %s: no obstruction found; attained Hankel sizes %d--%d"
                % (st.name, low, high))
            summary.append((st.name, len(cands), 0,
                            "no obstruction found (attained sizes %d--%d)" % (low, high)))
    say()
    say("=" * 90)
    say("FINAL SUMMARY")
    for s in summary:
        say("  %-32s %4d candidates, %3d without a law: %s" % s)
    say("finished %s, total %.2f hours" % (time.strftime("%Y-%m-%d %H:%M"), (time.time() - start) / 3600))
    out.close()


# ---------------------------------------------------------------------------------------
#  Irreducible modules
# ---------------------------------------------------------------------------------------
def fundamentals(f):
    """Weights of the fundamental modules of a factor, with their highest weights."""
    n, k = f.n, f.kind
    V = f.vector()
    out = []
    if k in ('A', 'GL'):
        for i in range(1, n if k == 'A' else n + 1):
            out.append(ext(f, V, i))
        if k == 'GL':
            out.append(dual(f, f.det(1)))       # det^-1, so that every weight is reachable
    elif k == 'C':
        prev = f.trivial()
        for i in range(1, n + 1):
            E = ext(f, V, i)
            if i >= 2:
                E = Counter(E)
                E.subtract(ext(f, V, i - 2))
                E = +E
            out.append(E)
    elif k == 'B':
        for i in range(1, n):
            out.append(ext(f, V, i))
        out.append(f.spin())
    else:  # D
        for i in range(1, n - 1):
            out.append(ext(f, V, i))
        if n >= 2:
            out.append(f.spin(1))
            out.append(f.spin(0))
        elif n == 1:
            out.append(Counter({(1,): 1}))
            out.append(Counter({(-1,): 1}))
    res = []
    for M in out:
        d = decompose(Group([f]), Counter({(w,): m for w, m in M.items()}))
        top = max(d, key=lambda w: sum(abs(x) for x in w[0]))
        res.append((top[0], M))
    return res


_irrep_cache = {}


def irrep(f, mu):
    """Weights of the irreducible module of highest weight mu, by peeling off the lower
    components of V_{mu - omega} (x) V_omega for a fundamental weight omega."""
    key = (f.kind, f.n, tuple(mu))
    if key in _irrep_cache:
        return _irrep_cache[key]
    mu = tuple(mu)
    if mu == f.zero:
        res = f.trivial()
    else:
        res = None
        for om, M in fundamentals(f):
            rest = f.normalize(tuple(a - b for a, b in zip(mu, om)))
            s, r = f.dot(rest)
            if s == 1 and r == rest:
                W = tensor(f, irrep(f, rest), M)
                d = decompose(Group([f]), Counter({(w,): m for w, m in W.items()}))
                assert d[(mu,)] == 1, (mu, d)
                W = Counter(W)
                for (nu,), m in d.items():
                    if nu != mu:
                        for w, k in irrep(f, nu).items():
                            W[w] -= m * k
                assert min(W.values()) >= 0
                res = +W
                break
        assert res is not None, mu
    _irrep_cache[key] = res
    return res


def sp_parabolic(n):
    """Sym^2(C^n) x| GL_n, the Siegel parabolic subgroup of Sp_2n itself."""
    h, l = Factor('C', n), Factor('GL', n)
    return Stabilizer("Sym^2(C^%d) x| GL_%d" % (n, n), h, tuple([1] * n), [l],
                      lambda mu: (tuple(mu),))
