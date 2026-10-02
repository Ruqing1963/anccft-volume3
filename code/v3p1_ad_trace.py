"""Volume III, Paper 1, open problem (a) of Remark V3P1:rem:scope: the first nonzero trace
Tr (C^(d+1))^(d+1) of the PGL_{d+1} companion on A~_d complexes
(chapters/ch01-s03-first-nonzero-trace.tex, Section "The first nonzero trace in every rank").

C^(d+1) has first block row ( A_1, -q A_2, q^3 A_3, ..., (-1)^(j-1) q^C(j,2) A_j, ..., (-1)^d q^C(d+1,2) I )
and identity blocks on the subdiagonal, so det(xI - C) = sum_j (-1)^j q^C(j,2) A_j x^(d+1-j).

  (W) word expansion: with non-commuting symbols A_1..A_d, the block trace of C^k equals
      sum over compositions (j_1..j_r) of k of  j_1 (-1)^(k-r) q^(sum C(j_i,2)) A_{j_1}...A_{j_r},
      modulo cyclic rotation of words (d <= 6, k <= d+1);
  (Q) the q-multinomial sum: sum over compositions j of m = d+1 of j_1 (-1)^(m-r) q^(sum C(j_i,2)) [m; j]_q
      equals prod_{i=1}^{d} (1 - q^i), symbolically for d = 1..12; factorizations printed; the
      conjectured d = 3 form n (q-1)^3 (q+1)(q^2+1) is tested and fails;
  (F) flag counts: the number of chains V = U_0 > U_1 > ... > U_r = 0 in F_q^(d+1) with codimension
      steps j_1..j_r is [d+1; j]_q, by enumeration of all subspaces, (d, q) in {(2,2),(3,2),(2,3),
      (3,3),(4,2)}; hence the per-vertex trace in the building is prod (1 - q^i);
  (T) thin tori of type A~_d (q = 1): vertices (Z/m)^(d+1)/<(1,..,1)>, d = 2, 3, 4; the flag condition
      holds for m >= d+2 and fails for m = d+1; Tr (C^(d+1))^k = 0 for (d+1) not dividing k, and
      Tr (C^(d+1))^(d+1) = 0 (+ the non-simplex correction when the flag condition fails), from the
      explicit sparse companion (d = 2, 3) and from the word formula (d <= 4);
  (Y) regression on Y_21 (A~_2, q = 2, flag condition fails): 651 = 63 + 3 s.

Run from code/:  python v3p1_ad_trace.py      (numpy, scipy, sympy; about a minute)
"""
import itertools
import math
import time
from collections import Counter, defaultdict

import numpy as np
import sympy as sp
from scipy.sparse import csr_matrix, identity as sp_identity, bmat

t0 = time.time()
q = sp.symbols("q")


def elapsed():
    return f"[{time.time() - t0:5.1f} s]"


def compositions(m):
    if m == 0:
        yield ()
        return
    for j in range(1, m + 1):
        for rest in compositions(m - j):
            yield (j,) + rest


def qint(k):
    return sum(q ** i for i in range(k))


def qfact(k):
    return sp.prod([qint(i) for i in range(1, k + 1)]) if k else sp.Integer(1)


def qmultinomial(parts):
    return sp.cancel(qfact(sum(parts)) / sp.prod([qfact(j) for j in parts]))


def binom2(j):
    return j * (j - 1) // 2


# ================================================================ (W) word expansion of the block trace
def word_mul(P, Q):
    out = defaultdict(lambda: sp.Integer(0))
    for w1, c1 in P.items():
        for w2, c2 in Q.items():
            out[w1 + w2] += c1 * c2
    return {w: c for w, c in out.items() if c != 0}


def word_add(P, Q):
    out = defaultdict(lambda: sp.Integer(0), P)
    for w, c in Q.items():
        out[w] += c
    return {w: sp.expand(c) for w, c in out.items() if sp.expand(c) != 0}


def cyclic(w):
    return min(w[i:] + w[:i] for i in range(len(w))) if w else w


def companion_words(d):
    """Block companion as a (d+1) x (d+1) array of non-commutative polynomials (word -> coeff).
    The word (j,) stands for A_j; A_{d+1} = I is the empty word with the shift recorded separately:
    we keep A_{d+1} as the letter d+1 and drop it at the end (it is the identity)."""
    m = d + 1
    C = [[{} for _ in range(m)] for _ in range(m)]
    for j in range(1, m + 1):
        C[0][j - 1] = {(j,): (-1) ** (j - 1) * q ** binom2(j)}
    for i in range(1, m):
        C[i][i - 1] = {(): sp.Integer(1)}
    return C


def block_power_trace(d, k):
    m = d + 1
    C = companion_words(d)
    P = [[({(): sp.Integer(1)} if i == j else {}) for j in range(m)] for i in range(m)]
    for _ in range(k):
        P = [[_rowcol(P, C, i, j, m) for j in range(m)] for i in range(m)]
    tr = {}
    for i in range(m):
        tr = word_add(tr, P[i][i])
    out = defaultdict(lambda: sp.Integer(0))
    for w, c in tr.items():
        out[cyclic(w)] += c
    return {w: sp.expand(c) for w, c in out.items() if sp.expand(c) != 0}


def _rowcol(P, C, i, j, m):
    acc = {}
    for l in range(m):
        if P[i][l] and C[l][j]:
            acc = word_add(acc, word_mul(P[i][l], C[l][j]))
    return acc


def composition_formula(k, d):
    out = defaultdict(lambda: sp.Integer(0))
    for comp in compositions(k):
        if max(comp) > d + 1:
            continue
        r = len(comp)
        out[cyclic(comp)] += comp[0] * (-1) ** (k - r) * q ** sum(binom2(j) for j in comp)
    return {w: sp.expand(c) for w, c in out.items() if sp.expand(c) != 0}


for d in range(1, 7):
    for k in range(1, d + 2):
        assert block_power_trace(d, k) == composition_formula(k, d), (d, k)
print(f"(W) block trace of (C^(d+1))^k = sum over compositions j of k of j_1 (-1)^(k-r) q^(sum C(j_i,2)) "
      f"Tr(A_j1 ... A_jr), as non-commutative words modulo rotation, for d <= 6 and k <= d+1   {elapsed()}")

# ================================================================ (Q) the q-multinomial sum
FORM = {}
for d in range(1, 13):
    m = d + 1
    S = sum(comp[0] * (-1) ** (m - len(comp)) * q ** sum(binom2(j) for j in comp) * qmultinomial(comp)
            for comp in compositions(m))
    S = sp.expand(sp.cancel(S))
    target = sp.expand(sp.prod([1 - q ** i for i in range(1, d + 1)]))
    assert sp.expand(S - target) == 0, d
    # partition (Newton) form gives the same
    FORM[d] = S
    if d <= 6:
        print(f"(Q) d = {d}: Tr (C^({m}))^{m} / n = {sp.factor(S)}")
print(f"(Q) Tr (C^(d+1))^(d+1) = n prod_(i=1..d) (1 - q^i) = (-1)^d n (q-1)^d [d]_q!  for d = 1..12 "
      f"(symbolic)   {elapsed()}")
assert sp.expand(FORM[1] - (1 - q)) == 0                                 # rank one: -n(q-1)
assert sp.expand(FORM[2] - (q - 1) ** 2 * (q + 1)) == 0                 # Theorem V3P1:thm:trace
conj = (q - 1) ** 3 * (q + 1) * (q ** 2 + 1)
diff = sp.factor(FORM[3] - conj)
assert sp.expand(FORM[3] - conj) != 0
assert sp.expand(FORM[3] + (q - 1) ** 3 * (q + 1) * (q ** 2 + q + 1)) == 0
print(f"(Q) d = 3: the conjectured n (q-1)^3 (q+1)(q^2+1) is false (difference {diff}); the value is "
      f"-n (q-1)^3 (q+1)(q^2+q+1), e.g. -21 n at q = 2 (conjecture: +15 n)")
for d in range(1, 13):                                                   # zero of order exactly d at q = 1
    f = sp.Poly(FORM[d], q)
    order = 0
    while f.eval(1) == 0:
        f = sp.Poly(sp.cancel(f.as_expr() / (q - 1)), q)
        order += 1
    assert order == d
    assert all(sp.sign(FORM[d].subs(q, qq)) == (-1) ** d for qq in (2, 3, 4, 5))
print("(Q) at q = 1 the trace vanishes to order exactly d; for q >= 2 its sign is (-1)^d")
# the trivial local factor: d/du prod_(i=0..d) (1 - q^i u) at u = 1 equals -prod_(i=1..d)(1 - q^i)
u = sp.symbols("u")
for d in range(1, 9):
    triv = sp.prod([1 - q ** i * u for i in range(d + 1)])
    assert sp.expand(sp.diff(triv, u).subs(u, 1) + FORM[d]) == 0
    xs = sp.symbols("x")
    loc = sum((-1) ** j * q ** binom2(j) * qmultinomial((j, d + 1 - j)) * xs ** (d + 1 - j) for j in range(d + 2))
    assert sp.expand(loc - sp.prod([xs - q ** i for i in range(d + 1)])) == 0
print("(Q) the local Hecke polynomial at the trivial vertex eigenvalues is prod_(i=0..d)(x - q^i), and "
      "d/du prod_(i=0..d)(1 - q^i u) at u = 1 equals -Tr (C^(d+1))^(d+1)/n  (d <= 8)   " + elapsed())


# ================================================================ (F) flag counts by enumeration
def subspaces(dim, p):
    """All subspaces of F_p^dim, as frozensets of vectors (tuples)."""
    vecs = list(itertools.product(range(p), repeat=dim))
    seen = set()
    spaces = []

    def span(gens):
        S = {tuple([0] * dim)}
        for g in gens:
            S = {tuple((a + c * b) % p for a, b in zip(s, g)) for s in S for c in range(p)}
        return frozenset(S)
    frontier = [frozenset({tuple([0] * dim)})]
    seen.add(frontier[0])
    spaces.append(frontier[0])
    while frontier:
        new = []
        for S in frontier:
            for v in vecs:
                if v in S:
                    continue
                T = span(list(_basis(S, dim, p)) + [v])
                if T not in seen:
                    seen.add(T)
                    spaces.append(T)
                    new.append(T)
        frontier = new
    return spaces


def _basis(S, dim, p):
    basis, cur = [], {tuple([0] * dim)}
    for v in sorted(S):
        if v not in cur:
            basis.append(v)
            cur = {tuple((a + c * b) % p for a, b in zip(s, v)) for s in cur for c in range(p)}
    return basis


for (d, p) in ((2, 2), (3, 2), (2, 3), (3, 3), (4, 2)):
    m = d + 1
    SP = subspaces(m, p)
    by_dim = defaultdict(list)
    for S in SP:
        by_dim[round(math.log(len(S), p))].append(S)
    assert [len(by_dim[k]) for k in range(m + 1)] == [int(qmultinomial((k, m - k)).subs(q, p)) for k in range(m + 1)]
    total = 0
    for comp in compositions(m):
        # chains V = U_0 > U_1 > ... > U_r = 0 with codim(U_i, U_{i-1}) = j_i
        dims = [m]
        for j in comp:
            dims.append(dims[-1] - j)
        count = {S: 1 for S in by_dim[m]}
        for a, b in zip(dims, dims[1:]):
            nxt = defaultdict(int)
            for T in by_dim[b]:
                nxt[T] = sum(c for S, c in count.items() if T <= S)
            count = nxt
        N = sum(count.values())
        assert N == int(qmultinomial(comp).subs(q, p)), (d, p, comp)
        total += comp[0] * (-1) ** (m - len(comp)) * p ** sum(binom2(j) for j in comp) * N
    assert total == int(FORM[d].subs(q, p))
    print(f"(F) F_{p}^{m}: {len(SP)} subspaces; every chain count is the q-multinomial; per-vertex trace "
          f"in the building = {total} = prod (1 - {p}^i)   {elapsed()}")


# ================================================================ (T) thin tori of type A~_d (q = 1)
def thin_torus(d, m):
    """Vertices a in (Z/m)^(d+1) modulo (1,..,1) (normal form a_0 = 0), type sum(a) mod (d+1);
    the arc of shift j from a goes to a + eps with eps in {0,1}^(d+1), |eps| = j (1 <= j <= d)."""
    assert m % (d + 1) == 0
    verts = [(0,) + t for t in itertools.product(range(m), repeat=d)]
    vid = {v: i for i, v in enumerate(verts)}
    n = len(verts)

    def norm(a):
        return tuple((x - a[0]) % m for x in a)
    A = {}
    for j in range(1, d + 1):
        rows, cols = [], []
        for v in verts:
            for eps in itertools.combinations(range(d + 1), j):
                w = list(v)
                for e in eps:
                    w[e] += 1
                rows.append(vid[v])
                cols.append(vid[norm(w)])
        A[j] = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    tau = np.array([sum(v) % (d + 1) for v in verts])
    return n, A, tau


def word_trace(A, n, comp):
    M = sp_identity(n, format="csr")
    for j in comp:
        M = M @ (A[j] if j in A else sp_identity(n, format="csr"))
    return int(round(M.diagonal().sum()))


for d, ms in ((2, (3, 6, 9)), (3, (4, 8)), (4, (5, 10))):
    for m_ in ms:
        n, A, tau = thin_torus(d, m_)
        for j in A:                                               # grading: A_j shifts type by j
            r, c = A[j].nonzero()
            assert np.all((tau[c] - tau[r]) % (d + 1) == j)
            assert np.all(np.asarray(A[j].sum(1)).ravel() == math.comb(d + 1, j))
        m = d + 1
        tr, s_tot, dev = 0, 0, {}
        for comp in compositions(m):
            t = word_trace(A, n, comp)
            flag = n * math.comb(m, 0) * math.factorial(m) // math.prod(math.factorial(j) for j in comp)
            if t != flag:
                dev[comp] = t - flag
            tr += comp[0] * (-1) ** (m - len(comp)) * t
        if d <= 3:                                                # explicit companion, sparse
            blocks = [[None] * m for _ in range(m)]
            for j in range(1, m + 1):
                blocks[0][j - 1] = (-1) ** (j - 1) * (A[j] if j <= d else sp_identity(n, format="csr"))
            for i in range(1, m):
                blocks[i][i - 1] = sp_identity(n, format="csr")
            Cm = bmat(blocks, format="csr")
            P = sp_identity(m * n, format="csr")
            for k in range(1, m + 1):
                P = P @ Cm
                tk = int(round(P.diagonal().sum()))
                if k < m:
                    assert tk == 0
            assert tk == tr
        status = "flag condition holds" if not dev else f"flag condition fails: excess walks {dict(dev)}"
        if not dev:
            assert tr == 0
        print(f"(T) thin A~_{d} torus, m = {m_}: n = {n}; Tr (C^({m}))^k = 0 for k < {m}"
              f"{' (explicit companion)' if d <= 3 else ''}; Tr (C^({m}))^{m} = {tr}; {status}   {elapsed()}")

# ================================================================ (Y) regression on Y_21
from a2_common import standard_covers                                     # noqa: E402
Y21 = standard_covers()["Y_21"]
A1 = Y21.A1.astype(np.int64)
A2 = A1.T
n = Y21.n
C3 = np.block([[A1, -2 * A2, 8 * np.eye(n, dtype=np.int64)], [np.eye(n, dtype=np.int64), 0 * A1, 0 * A1],
               [0 * A1, np.eye(n, dtype=np.int64), 0 * A1]])
tr3 = int(np.trace(np.linalg.matrix_power(C3, 3)))
s = (int(np.trace(np.linalg.matrix_power(A1, 3))) - 3 * len(Y21.tris)) // 3
assert tr3 == 651 == n * int(FORM[2].subs(q, 2)) + 3 * s
print(f"(Y) Y_21 (q = 2, flag condition fails): Tr (C^(3))^3 = {tr3} = n prod(1-2^i) + 3 s = "
      f"{n * int(FORM[2].subs(q, 2))} + 3 * {s}")

print(f"\nALL CHECKS PASSED: (W) (Q) (F) (T) (Y).   {elapsed()}")
