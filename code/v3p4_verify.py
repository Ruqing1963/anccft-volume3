"""Verification battery for Volume III, Paper 4 (chapters/ch04-geodesic-edge-operator.tex).

  (R) the geodesic edge operator L_E: row and column sums q^2, block-cyclic in the tail type,
      and the intertwining relations T L = q^2 S, L S^t = q^2 T^t, L T^t = T^t A_1 - Q,
      L Q = q T^t A_2 - q S^t, S Q = (q+1) A_2, T Q = (q+1) A_1;
  (U) unitary remainder: W = ker S cap ker T cap ker Q^t is L_E^t-invariant and
      ||L_E^t f|| = sqrt(q) ||f|| on W; rank Psi = 3n - 6, dim W = 4n + 6;
  (F) factorization: spec L_E = {q^2 w^k} u {reciprocal roots of nontrivial P_j}
      u spec(sqrt(q) U); every eigenvalue has |lambda| >= sqrt(q); the five covers are
      Ramanujan and their nontrivial spectrum lies on |lambda| in {q, sqrt(q)};
  (X) exact integer identity on Y_21:
      det(I - u L_E) = (1-64u^3)(1-8u^3)^18 (1-2u^3+8u^6)^9 (1+5u^3+8u^6)^6 at 8 points;
  (K) ker Psi is the 6-dimensional trivial part; the exceptional curve
      a(lambda) = (q+1) lambda + q^2/lambda^2, |lambda| = sqrt(q), carries a root of modulus
      sqrt(q) and misses the tempered region;
  (D) local unitarity for A~_d: J - Inc(PG(d,q)) has sums q^d and singular value
      q^((d-1)/2) on the complement of constants (d = 2, 3, 4; q = 2, 3);
  (S) level statistics of the remainder of Y_168: 16 distinct values, 93% zero spacings;
      perturbation U e^{i eps V}: ratio statistic and small-spacing fraction (seeded);
  (B) Brody: b = 1 is the GOE Wigner surmise; b = 2 is not the GUE surmise.

Run:  python code/v3p4_verify.py      (numpy only)
"""
import itertools
import math
import time

import numpy as np

from a2_common import bareiss_det, standard_covers

t0 = time.time()
COV = standard_covers()
q = 2
KAPPA = q * q + q + 1


def ops(X):
    E, idx, n = X.edges, X.eidx, X.n
    m = len(E)
    tris = [tuple(t) for t in X.tris]
    chambers = set(frozenset(t) for t in tris)
    out = {}
    for (a, b) in E:
        out.setdefault(a, []).append(b)
    L = np.zeros((m, m), dtype=np.int64)
    S = np.zeros((n, m), dtype=np.int64)
    T = np.zeros((n, m), dtype=np.int64)
    Q = np.zeros((m, n), dtype=np.int64)
    for (x, y), i in idx.items():
        S[x, i] = T[y, i] = 1
        for z in out[y]:
            if frozenset((x, y, z)) not in chambers:
                L[i, idx[(y, z)]] = 1
    for (a, b, c) in tris:                       # a -> b -> c -> a
        Q[idx[(b, c)], a] += 1
        Q[idx[(c, a)], b] += 1
        Q[idx[(a, b)], c] += 1
    return L, S, T, Q


def null_basis(M, tol=1e-9):
    """Orthonormal basis of ker M (float SVD)."""
    u, s, vt = np.linalg.svd(M.astype(float))
    r = int((s > tol * max(1.0, s.max())).sum())
    return vt[r:].T


def same_multiset(u, v, tol):
    u, v = sorted(u, key=lambda z: (round(z.real, 5), round(z.imag, 5))), list(v)
    if len(u) != len(v):
        return False
    for x in u:
        k = int(np.argmin(np.abs(np.array(v) - x)))
        if abs(v[k] - x) > tol:
            return False
        v.pop(k)
    return True


DATA = {}
for name, X in COV.items():
    n = X.n
    L, S, T, Q = ops(X)
    A1 = X.A1
    A2 = A1.T
    # ------------------------------------------------------------ (R) structure
    assert set(L.sum(1)) == {q * q} and set(L.sum(0)) == {q * q}
    tail = np.array([X.tau[e[0]] for e in X.edges])
    for i, j in zip(*np.nonzero(L)):
        assert tail[j] == (tail[i] + 1) % 3
    assert (T @ L == q * q * S).all() and (L @ S.T == q * q * T.T).all()
    assert (L @ T.T == T.T @ A1 - Q).all() and (L @ Q == q * T.T @ A2 - q * S.T).all()
    assert (S @ Q == (q + 1) * A2).all() and (T @ Q == (q + 1) * A1).all()
    # ------------------------------------------------------------ (U) unitary remainder
    Psi = np.hstack([S.T, T.T, Q])
    W = null_basis(Psi.T)
    assert W.shape[1] == 4 * n + 6 and np.linalg.matrix_rank(Psi.astype(float)) == 3 * n - 6
    LtW = L.T.astype(float) @ W
    Ured = W.T @ LtW / math.sqrt(q)
    assert np.abs(LtW - W @ (W.T @ LtW)).max() < 1e-10               # invariance
    assert np.abs(Ured.T @ Ured - np.eye(W.shape[1])).max() < 1e-10   # isometry
    # ------------------------------------------------------------ (F) factorization
    evL = np.linalg.eigvals(L.astype(float))
    evW = math.sqrt(q) * np.linalg.eigvals(Ured.T)                     # compression of L_E
    a = np.linalg.eigvals(A1.astype(float))
    vert = []
    triv = 0
    for aj in a:
        if abs(abs(aj) - KAPPA) < 1e-8:
            triv += 1
            vert.append(q * q * aj / KAPPA)
            continue
        vert += list(np.roots([1, -aj, q * np.conj(aj), -q ** 3]))
    assert triv == 3
    assert same_multiset(evL, np.r_[vert, evW], 1e-6), name
    mods = np.abs(evL)
    assert mods.min() > math.sqrt(q) - 1e-8
    nontriv = mods[np.abs(mods - q * q) > 1e-6]
    assert np.all((np.abs(nontriv - q) < 1e-6) | (np.abs(nontriv - math.sqrt(q)) < 1e-6))
    DATA[name] = dict(n=n, E=len(X.edges), rank=3 * n - 6, dimW=4 * n + 6,
                      q_circle=int((np.abs(mods - q) < 1e-6).sum()),
                      sq_circle=int((np.abs(mods - math.sqrt(q)) < 1e-6).sum()),
                      distinct=len(set(np.round(evW ** 3, 6))))
    # ------------------------------------------------------------ (K) kernel of Psi
    K = null_basis(Psi)
    assert K.shape[1] == 6
    for v in K.T:                                  # each (phi1,phi2,phi3) is in the trivial orbit
        for part in v.reshape(3, n):
            chars = np.array([[np.exp(2j * math.pi * k * t / 3) for t in X.tau] for k in range(3)])
            coef = chars.conj() @ part / n
            assert np.abs(part - coef @ chars).max() < 1e-9
print("(R) L_E: sums q^2 = 4, block-cyclic in tail type, six intertwining relations "
      "(five covers)")
print("(U) W = ker S ^ ker T ^ ker Q^t: L_E^t-invariant, ||L_E^t f|| = sqrt(2)||f||; "
      "rank Psi = 3n-6, dim W = 4n+6")
print("(F) spec L_E = {4w^k} u roots of nontrivial P_j u spec(compression); all |lambda| >= "
      "sqrt 2; Ramanujan, two circles")
for name, d in DATA.items():
    print(f"    {name}: |E| = {d['E']}, |lambda|=q: {d['q_circle']}, |lambda|=sqrt q: "
          f"{d['sq_circle']}, distinct lambda^3 on W: {d['distinct']}")
print("(K) ker Psi = 6-dimensional, spanned by the trivial characters 1, w^tau, w^2tau")

# ---------------------------------------------------------------- (X) exact identity on Y_21
Y21 = COV["Y_21"]
L21 = ops(Y21)[0]
m = L21.shape[0]
for u in [1, 2, 3, -1, -2, 5, 7, -3]:
    M = (np.eye(m, dtype=np.int64) - u * L21).tolist()
    lhs = bareiss_det(M)
    rhs = ((1 - 64 * u ** 3) * (1 - 8 * u ** 3) ** 18 * (1 - 2 * u ** 3 + 8 * u ** 6) ** 9
           * (1 + 5 * u ** 3 + 8 * u ** 6) ** 6)
    assert lhs == rhs, u
print("(X) det(I - u L_E(Y_21)) = (1-64u^3)(1-8u^3)^18(1-2u^3+8u^6)^9(1+5u^3+8u^6)^6 "
      "at u = 1,2,3,-1,-2,5,7,-3 (exact)")

# ---------------------------------------------------------------- (K') exceptional curve
rng = np.random.default_rng(0)
for qq in (2, 3, 4, 5):
    for al in np.linspace(0, 2 * math.pi, 37):
        lam = math.sqrt(qq) * np.exp(1j * al)
        aa = (qq + 1) * lam + qq * qq / lam ** 2
        assert abs(lam ** 3 - aa * lam ** 2 + qq * np.conj(aa) * lam - qq ** 3) < 1e-9
        assert abs(np.conj(aa) - (lam ** 2 + qq * (qq + 1) / lam)) < 1e-9
    for _ in range(2000):                        # tempered a: all roots have modulus q
        z1, z2 = np.exp(1j * rng.uniform(0, 2 * math.pi, 2))
        aa = qq * (z1 + z2 + 1 / (z1 * z2))
        r = np.abs(np.roots([1, -aa, qq * np.conj(aa), -qq ** 3]))
        assert np.abs(r - qq).max() < 1e-6
print("(K') exceptional curve a(lambda), |lambda| = sqrt q, has P_a(1/lambda) = 0; tempered a "
      "have all roots of modulus q (q = 2..5)")


# ---------------------------------------------------------------- (D) local unitarity, A~_d
def pg_incidence(d, qq):
    pts = []
    for v in itertools.product(range(qq), repeat=d + 1):
        if any(v):
            first = next(x for x in v if x)
            inv = pow(first, -1, qq)
            w = tuple((inv * x) % qq for x in v)
            if w not in pts:
                pts.append(w)
    P = np.array(pts)
    return ((P @ P.T) % qq == 0).astype(np.int64)   # hyperplane h = point^perp


for d, qq in [(2, 2), (2, 3), (3, 2), (3, 3), (4, 2)]:
    Inc = pg_incidence(d, qq)                      # rows: hyperplanes, cols: points
    Np = Inc.shape[0]
    M = np.ones((Np, Np), dtype=np.int64) - Inc
    assert set(M.sum(0)) == {qq ** d} and set(M.sum(1)) == {qq ** d}
    P0 = np.eye(Np) - np.ones((Np, Np)) / Np
    for G in (M.T @ M, M @ M.T):
        assert np.abs(P0 @ G @ P0 - qq ** (d - 1) * P0).max() < 1e-8
print("(D) J - Inc(PG(d,q)): sums q^d, and M^tM = MM^t = q^(d-1) on 1-perp "
      "for (d,q) = (2,2),(2,3),(3,2),(3,3),(4,2)")

# ---------------------------------------------------------------- (S) level statistics
Y = COV["Y_168"]
L, S, T, Q = ops(Y)
W = null_basis(np.hstack([S.T, T.T, Q]).T)
tail = np.array([Y.tau[e[0]] for e in Y.edges])
Pr = W @ W.T
B = Pr[:, tail == 0]                               # projections of type-0 unit vectors
uu, ss, _ = np.linalg.svd(B, full_matrices=False)
W0 = uu[:, ss > 1e-8]                              # W cap C^{E_0}
assert np.abs(W0.T @ (np.eye(len(tail))[:, tail != 0])).max() < 1e-8
U3 = W0.T @ np.linalg.matrix_power(L.T.astype(float), 3) @ W0 / q ** 1.5
N = U3.shape[0]
assert N == 226 and np.abs(U3.T @ U3 - np.eye(N)).max() < 1e-10


def spacing_stats(ph):
    ph = np.sort(np.mod(ph, 2 * math.pi))
    s = np.diff(np.r_[ph, ph[0] + 2 * math.pi])
    s1, s2 = s, np.roll(s, -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.minimum(s1, s2) / np.maximum(s1, s2)
    return np.nanmean(r), float(np.mean(s < 0.1 * s.mean())), float(np.mean(s < 1e-6 * s.mean()))


ph0 = np.angle(np.linalg.eigvals(U3))
distinct0 = len(set(np.round(ph0, 6)))
_, small0, zero0 = spacing_stats(ph0)
assert distinct0 == 16 and abs(zero0 - 210 / 226) < 1e-9
print(f"(S) Y_168, W cap C^(E_0): N = {N}, distinct eigenphases {distinct0}, exact zero "
      f"spacings {zero0:.4f} = 210/226")


def expiH(H, eps):
    w, v = np.linalg.eigh(H)
    return (v * np.exp(1j * eps * w)) @ v.conj().T


STATS = {}
rng = np.random.default_rng(1)
for kind in ("GUE", "GOE"):
    for eps in (1e-3, 1e-2, 1e-1, 1.0, 3.0):
        R, Z = [], []
        for _ in range(12):
            A = rng.standard_normal((N, N)) + (1j * rng.standard_normal((N, N)) if kind == "GUE" else 0)
            V = (A + A.conj().T) / (2 * math.sqrt(N))
            r, small, _ = spacing_stats(np.angle(np.linalg.eigvals(U3 @ expiH(V, eps))))
            R.append(r)
            Z.append(small)
        STATS[(kind, eps)] = (float(np.mean(R)), float(np.mean(Z)))
        print(f"    {kind} eps = {eps:6.3f}: <r> = {np.mean(R):.4f}, "
              f"fraction of spacings < 0.1 mean = {np.mean(Z):.3f}")
for kind in ("GUE", "GOE"):
    assert STATS[(kind, 1e-3)][1] > 0.9 and STATS[(kind, 1e-2)][1] > 0.9
    assert abs(STATS[(kind, 3.0)][0] - 0.5996) < 0.02           # GUE value, even for real V
print("    reference <r>: Poisson 0.3863, GOE 0.5307, GUE 0.5996")


# ---------------------------------------------------------------- (B) Brody
def brody(s, b):
    al = math.gamma((b + 2) / (b + 1)) ** (b + 1)
    return (b + 1) * al * s ** b * np.exp(-al * s ** (b + 1))


s = np.linspace(0, 40, 400001)
goe = math.pi / 2 * s * np.exp(-math.pi * s ** 2 / 4)
gue = 32 / math.pi ** 2 * s ** 2 * np.exp(-4 * s ** 2 / math.pi)
for b in (0.0, 1.0, 2.0):                                        # normalization, unit mean
    p = brody(s, b)
    assert abs(np.trapz(p, s) - 1) < 1e-3 and abs(np.trapz(s * p, s) - 1) < 1e-3
assert np.abs(brody(s, 1.0) - goe).max() < 1e-12
dev = float(np.abs(brody(s, 2.0) - gue).max())
assert dev > 0.05
print(f"(B) Brody b = 1 equals the GOE surmise; b = 2 differs from the GUE surmise by "
      f"{dev:.4f} in sup norm")

print(f"\nALL CHECKS PASSED: (R) (U) (F) (X) (K) (K') (D) (S) (B).   [{time.time() - t0:.0f} s]")
