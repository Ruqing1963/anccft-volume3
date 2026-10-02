"""Breaking the arithmetic degeneracy: the crossover to GUE for the unitary remainder of
Y_168 (Volume III, Paper 4, chapters/ch04-s06-gue-crossover.tex; open problem (d) of Section 6).

  (A) U_3: the real orthogonal 226 x 226 remainder of (L_E^t)^3 on W cap C^{E_0}, rebuilt
      exactly as in v3p4_verify.py (16 distinct eigenphases, atom 210/226 at s = 0).
  (B) crossover: U_eps = U_3 exp(i eps V) for four kinds of Hermitian V (GUE-type, GOE-type,
      geometric on-site, geometric local 0/+-1), eps on a 31-point log grid 1e-4..10, 20
      samples each: ratio statistic <r>, fraction of spacings below 0.1 of the mean, the
      crossover window (eps_half, eps_sat), and the first-order prediction from the
      eigenvalues of P_k V P_k on the 16 degenerate eigenspaces.
  (C) saturated regime: unfolded spacing distribution, Brody MLE, KS and KL distances to the
      GOE and GUE surmises and to the Brody b = 2 curve; Haar references at N = 226.
  (D) symmetry: trace obstruction kappa(V) shows U_eps has no antiunitary time reversal for
      generic V; two time-reversal symmetric families with the same unperturbed spectrum
      (S_0 = O Lambda O^t from the real Schur form, and U_3 with T-real V, T = Z K) give the
      GOE value of <r>.
  (E) deck group (Z/2)^3 x Z/7: the translation action on edges commutes with L_E, S, T, Q and
      with U_3; 56 character blocks of sizes 6 + 55 x 4 = 226; within-block structure (simple
      except in the Z/7 sector), the 6 spectral classes of blocks, their arithmetic
      (Q(sqrt(-7))), the lifts of the automorphisms x -> x+1, x -> 2x of X_3 to Y_168, whose
      orbits on characters are exactly the 6 classes; deck-equivariant perturbations
      (block-diagonal V, pull-backs from X_3) give superposition statistics, not GUE.

Run from code/:  python v3p4_quantum_chaos.py      (numpy + scipy; seeded; about 6 minutes)
"""
import itertools
import math
import time

import numpy as np
from scipy.linalg import schur
from scipy.optimize import minimize_scalar
from scipy.special import erf

from a2_common import standard_covers, relator_matrix, nullspace_mod   # no checks run on import
from a2_common import T as TRIPLES, GIDX as A2GIDX

t0 = time.time()
q = 2
TWO_PI = 2 * math.pi
R_POISSON, R_GOE, R_GUE = 0.3863, 0.5307, 0.5996
rng = np.random.default_rng(2026)


def elapsed():
    return f"[{time.time() - t0:5.0f} s]"


# ================================================================ (A) the remainder U_3
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
    u, s, vt = np.linalg.svd(M.astype(float))
    r = int((s > tol * max(1.0, s.max())).sum())
    return vt[r:].T


Y = standard_covers()["Y_168"]
n = Y.n
L, S, T, Q = ops(Y)
m = L.shape[0]
W = null_basis(np.hstack([S.T, T.T, Q]).T)
tail = np.array([Y.tau[e[0]] for e in Y.edges])
Pr = W @ W.T
B0 = Pr[:, tail == 0]
uu, ss, _ = np.linalg.svd(B0, full_matrices=False)
W0 = uu[:, ss > 1e-8]                              # orthonormal basis of W cap C^{E_0}
U3 = W0.T @ np.linalg.matrix_power(L.T.astype(float), 3) @ W0 / q ** 1.5
N = U3.shape[0]
assert N == 226 and np.abs(U3.T @ U3 - np.eye(N)).max() < 1e-10
MEAN_SP = TWO_PI / N


def circle_stats(ph):
    """(<r>, fraction of spacings < 0.1 mean, unfolded spacings) on the circle."""
    ph = np.sort(np.mod(ph, TWO_PI))
    s = np.diff(np.r_[ph, ph[0] + TWO_PI])
    s1, s2 = s, np.roll(s, -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.minimum(s1, s2) / np.maximum(s1, s2)
    return float(np.nanmean(r)), float(np.mean(s < 0.1 * s.mean())), s / s.mean()


def line_stats(ph):
    """Same for a spectrum on an interval (no wrap-around)."""
    ph = np.sort(ph)
    s = np.diff(ph)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.minimum(s[:-1], s[1:]) / np.maximum(s[:-1], s[1:])
    return float(np.nanmean(r)), float(np.mean(s < 0.1 * s.mean())), s / s.mean()


def clusters(ph, tol=1e-6):
    """Representatives and multiplicities of the distinct values of a phase list."""
    ph = np.sort(np.mod(ph, TWO_PI))
    reps, mult = [ph[0]], [1]
    for x in ph[1:]:
        if x - reps[-1] < tol:
            mult[-1] += 1
        else:
            reps.append(x)
            mult.append(1)
    if len(reps) > 1 and reps[0] + TWO_PI - reps[-1] < tol:   # wrap-around
        mult[0] += mult.pop()
        reps.pop()
    return np.array(reps), np.array(mult)


ph0 = np.angle(np.linalg.eigvals(U3))
reps0, mult0 = clusters(ph0)
r0, small0, s0 = circle_stats(ph0)
zero0 = float(np.mean(s0 < 1e-6))
assert len(reps0) == 16 and abs(zero0 - 210 / 226) < 1e-9
print(f"(A) U_3 on W cap C^(E_0) of Y_168: N = {N}, real orthogonal; {len(reps0)} distinct "
      f"eigenphases, multiplicities {sorted(mult0.tolist(), reverse=True)}; atom at s = 0 of mass "
      f"{zero0:.4f} = 210/226; mean spacing 2 pi / N = {MEAN_SP:.5f}  {elapsed()}")

# eigenspace bases of U_3 (orthonormal, complex) for the 16 clusters
EIG_U3 = []
for th in reps0:
    u_, s_, vt_ = np.linalg.svd(U3 - np.exp(1j * th) * np.eye(N))
    EIG_U3.append(vt_[s_ < 1e-8].conj().T)
assert sum(B.shape[1] for B in EIG_U3) == N
assert all(abs(B.shape[1] - mu) == 0 for B, mu in zip(EIG_U3, mult0))

# ================================================================ (E1) deck group action
MODS = (2, 2, 2, 7)
G = sorted(itertools.product(range(2), range(2), range(2), range(7)))
GIDX = {g: i for i, g in enumerate(G)}
assert n == 3 * len(G) and all(Y.tau[i] == i % 3 for i in range(n))


def gadd(a, b):
    return tuple((u + v) % mm for u, v, mm in zip(a, b, MODS))


def perms(g):
    """Vertex and edge permutation of the deck translation (a, s) -> (a + g, s)."""
    pv = np.array([3 * GIDX[gadd(G[i // 3], g)] + i % 3 for i in range(n)])
    ep = np.array([Y.eidx[(pv[a], pv[b])] for (a, b) in Y.edges])
    return pv, ep


EP, DG = {}, {}
for g in G:
    pv, ep = perms(g)
    assert (Y.A1[np.ix_(pv, pv)] == Y.A1).all()
    assert (L[np.ix_(ep, ep)] == L).all() and (S[np.ix_(pv, ep)] == S).all()
    assert (T[np.ix_(pv, ep)] == T).all() and (Q[np.ix_(ep, pv)] == Q).all()
    assert (tail[ep] == tail).all()
    EP[g] = ep
    DG[g] = W0.T @ W0[np.argsort(ep)]                  # compression of P_g to W cap C^{E_0}
    assert np.abs(DG[g].T @ DG[g] - np.eye(N)).max() < 1e-10      # W cap C^{E_0} is invariant
    assert np.abs(DG[g] @ U3 - U3 @ DG[g]).max() < 1e-10          # U_3 commutes with the deck group
g1, g2 = (1, 0, 0, 0), (0, 0, 0, 1)
assert np.abs(DG[g1] @ DG[g2] - DG[gadd(g1, g2)]).max() < 1e-10
print("(E1) deck group A = (Z/2)^3 x Z/7 acting by translation of (a, s): preserves types and "
      "chambers, commutes with L_E, S, T, Q; the compressed action on W cap C^(E_0) is "
      "orthogonal and commutes with U_3")

# character blocks
CHARS = [(k, jj) for k in itertools.product(range(2), repeat=3) for jj in range(7)]


def chi(k, jj, g):
    return (-1) ** (k[0] * g[0] + k[1] * g[1] + k[2] * g[2]) * np.exp(2j * math.pi * jj * g[3] / 7)


BLOCK = {}
for (k, jj) in CHARS:
    Pi = sum(np.conj(chi(k, jj, g)) * DG[g] for g in G) / len(G)
    w, v = np.linalg.eigh(Pi)
    assert np.all((np.abs(w) < 1e-8) | (np.abs(w - 1) < 1e-8))
    B = v[:, w > 0.5]
    Uchi = B.conj().T @ U3 @ B
    assert np.abs(U3 @ B - B @ Uchi).max() < 1e-8                 # block is U_3-invariant
    BLOCK[(k, jj)] = (B, np.angle(np.linalg.eigvals(Uchi)))
dims = {key: B.shape[1] for key, (B, _) in BLOCK.items()}
assert dims[((0, 0, 0), 0)] == 6 and all(d == 4 for key, d in dims.items() if key != ((0, 0, 0), 0))
assert sum(dims.values()) == N
# the isotypic components are mutually orthogonal and span everything
Ball = np.hstack([B for (B, _) in BLOCK.values()])
assert np.abs(Ball.conj().T @ Ball - np.eye(N)).max() < 1e-8
print("(E2) 56 character blocks of U_3: dim 6 for chi = 1 (the type-0 remainder of X_3) and 4 "
      "for the 55 nontrivial characters; 6 + 55*4 = 226")

# within-block and cross-block structure
def ctype(key):
    """Order of the character: 1 (trivial), 2 (factors through (Z/2)^3), 7 (through Z/7), 14."""
    k, jj = key
    return "1" if (k == (0, 0, 0) and jj == 0) else ("2" if jj == 0 else ("7" if k == (0, 0, 0) else "14"))


within_deg = {}                              # block -> multiplicity pattern if not simple
min_gap_within = []
cross = {k: set() for k in range(16)}       # cluster -> set of blocks containing it
classes = {}
for key, (B, ph) in BLOCK.items():
    rp, mu = clusters(ph)
    if mu.max() > 1:
        within_deg[key] = sorted(mu.tolist(), reverse=True)
    if len(rp) > 1:
        gaps = np.diff(np.r_[rp, rp[0] + TWO_PI])
        min_gap_within.append(gaps.min())
    for x in ph:
        kk = int(np.argmin(np.abs(np.exp(1j * reps0) - np.exp(1j * x))))
        assert abs(np.exp(1j * reps0[kk]) - np.exp(1j * x)) < 1e-6
        cross[kk].add(key)
    sig = tuple(np.round(np.sort(np.mod(ph, TWO_PI)), 5))
    classes.setdefault(sig, []).append(key)
n_blocks_per_value = [len(cross[k]) for k in range(16)]
n_within = len(within_deg)
within_kinds = sorted(set(ctype(key) for key in within_deg), key=int)
within_patterns = sorted(set(tuple(p) for p in within_deg.values()))
# 226 phases, 16 values: the degeneracy is overwhelmingly across blocks
cross_pairs = sum(mu * (mu - 1) // 2 for mu in mult0)                      # coincident pairs, total
within_pairs = sum(sum(x * (x - 1) // 2 for x in p) for p in within_deg.values())
print(f"(E3) within-block degeneracies: {n_within} of 56 blocks are not simple (character orders "
      f"{{{','.join(within_kinds)}}}, multiplicity patterns {within_patterns}); the other "
      f"{56 - n_within} blocks have simple spectrum, smallest within-block gap "
      f"{min(min_gap_within):.4f} rad = {min(min_gap_within) / MEAN_SP:.1f} mean spacings; of the "
      f"{cross_pairs} coincident pairs of eigenphases, {within_pairs} lie inside one block and "
      f"{cross_pairs - within_pairs} across blocks; every one of the 16 values occurs in between "
      f"{min(n_blocks_per_value)} and {max(n_blocks_per_value)} blocks; the 56 block spectra form "
      f"{len(classes)} distinct multisets")
for sig, keys in sorted(classes.items(), key=lambda kv: -len(kv[1])):
    kinds = sorted(set(ctype(key) for key in keys), key=int)
    deg = ", ".join(f"{math.degrees(x):7.2f}" for x in sig)
    print(f"     class of {len(keys):2d} blocks (character orders {{{','.join(kinds)}}}): phases "
          f"{deg} deg")
# arithmetic of the Z/7 sector (the pull-back of Y_21): with pi = (1 + sqrt(-7))/2, the prime
# above 2 in Q(sqrt(-7)), the values of lambda^3 are 2 pi, 2 pi-bar, pi^3, pi-bar^3, i.e. the
# phases are +-phi and +-3 phi with phi = arg pi = arctan(sqrt 7).  In terms of mu = 2^{3/2} lambda:
# trivial block (mu^2 + 5 mu + 8)^3, square-class blocks (mu - 2 pi)^3 (mu - pi-bar^3), and the
# conjugate for the non-square class; the order-2 blocks have mu^4 + 11 mu^2 + 64.
phi7 = math.atan(math.sqrt(7))
sector7 = sorted(np.mod([phi7, -phi7, 3 * phi7, -3 * phi7], TWO_PI))
present = [reps0[np.argmin(np.abs(np.exp(1j * reps0) - np.exp(1j * x)))] for x in sector7]
assert np.abs(np.exp(1j * np.array(present)) - np.exp(1j * np.array(sector7))).max() < 1e-6
for key, (B, ph) in BLOCK.items():
    mu = 2 ** 1.5 * np.exp(1j * ph)
    coef = np.poly(mu)
    if ctype(key) == "1":
        assert np.abs(coef - np.poly(np.roots([1, 5, 8]).tolist() * 3)).max() < 1e-6
    elif ctype(key) == "2":
        assert np.abs(coef - np.array([1, 0, 11, 0, 64])).max() < 1e-6
    elif ctype(key) == "7":
        pi7 = (1 + 1j * math.sqrt(7)) / 2
        for roots in ([2 * pi7] * 3 + [np.conj(pi7) ** 3], [2 * np.conj(pi7)] * 3 + [pi7 ** 3]):
            if np.abs(coef - np.poly(roots)).max() < 1e-6:
                break
        else:
            raise AssertionError(key)
print("     arithmetic: Z/7-sector phases are +-phi, +-3phi with phi = arg((1+sqrt(-7))/2) = "
      f"{math.degrees(phi7):.3f} deg; block polynomials of mu = 2^(3/2) lambda: (mu^2+5mu+8)^3 "
      "(trivial), (mu-2pi)^3(mu-pibar^3) and conjugate (order 7), mu^4+11mu^2+64 (order 2)")

# ---------------------------------------------------------------- (E6) lifted automorphisms
# x -> x+1 and x -> 2x preserve the triangle list T of the presentation, hence are automorphisms
# of X_3 fixing the three vertices.  Lift to Y_168: (a, s) -> (phi(a) + c_s, s) with phi in Aut(A)
# and constants c_s (the voltage assignment uses a spanning tree, which the automorphism moves).
RM = relator_matrix()
NN7, NN2 = nullspace_mod(RM, 7), nullspace_mod(RM, 2)
THETA = [tuple(NN2[j][i] for j in range(3)) + (NN7[0][i],) for i in range(21)]
th = lambda s, x: THETA[A2GIDX[(s, x)]]
for (i, j) in Y.edges:                               # the edge structure of the voltage cover
    a, s = G[i // 3], i % 3
    assert j % 3 == (s + 1) % 3 and any(gadd(a, th(s, x)) == G[j // 3] for x in range(7))
MATS2 = [np.array(M).reshape(3, 3) for M in itertools.product(range(2), repeat=9)]
MATS2 = [M for M in MATS2 if round(abs(np.linalg.det(M))) % 2 == 1]
assert len(MATS2) == 168


def lift(pi):
    sols = []
    for M2 in MATS2:
        for k7 in range(1, 7):
            phi = lambda a: tuple((M2 @ np.array(a[:3])) % 2) + ((k7 * a[3]) % 7,)
            delta = []
            for s in range(3):
                ds = {tuple((u - v) % mm for u, v, mm in zip(phi(th(s, x)), th(s, pi(x)), MODS))
                      for x in range(7)}
                if len(ds) != 1:
                    break
                delta.append(ds.pop())
            else:
                tot = (0, 0, 0, 0)
                for d in delta:
                    tot = gadd(tot, d)
                if tot == (0, 0, 0, 0):
                    sols.append((M2, k7, delta))
    return sols


AUT = {}
for name, pi in (("x+1", lambda x: (x + 1) % 7), ("2x", lambda x: (2 * x) % 7)):
    assert all(tuple(pi(v) for v in tr) in set(TRIPLES) for tr in TRIPLES)
    sols = lift(pi)
    assert len(sols) == 1
    M2, k7, delta = sols[0]
    phi = lambda a, M2=M2, k7=k7: tuple((M2 @ np.array(a[:3])) % 2) + ((k7 * a[3]) % 7,)
    c = [(0, 0, 0, 0)]
    for s in range(2):
        c.append(tuple((u - v) % mm for u, v, mm in zip(c[-1], delta[s], MODS)))
    pv = np.array([3 * GIDX[gadd(phi(G[i // 3]), c[i % 3])] + i % 3 for i in range(n)])
    ep = np.array([Y.eidx[(pv[a], pv[b])] for (a, b) in Y.edges])
    assert (L[np.ix_(ep, ep)] == L).all() and (Q[np.ix_(ep, pv)] == Q).all()
    assert (S[np.ix_(pv, ep)] == S).all() and (T[np.ix_(pv, ep)] == T).all()
    Dt = W0.T @ W0[np.argsort(ep)]
    assert np.abs(Dt @ U3 - U3 @ Dt).max() < 1e-8
    for g in (g1, g2):
        assert np.abs(Dt @ DG[g] @ Dt.T - DG[phi(g)]).max() < 1e-8   # normalizes the deck group
    AUT[name] = (phi, k7, M2)
    order2 = next(k for k in range(1, 8) if (np.linalg.matrix_power(M2, k) % 2 == np.eye(3)).all())
    print(f"(E6) automorphism x -> {name} of X_3 lifts to Y_168 (unique lift fixing the fibre over "
          f"the base point up to deck translations): acts on A by phi_7 = {k7} on Z/7 and by an "
          f"element of order {order2} of GL(3,2) on (Z/2)^3; commutes with U_3, normalizes A")


def char_after(key, phi):
    """chi o phi^{-1}."""
    k, jj = key
    vals = {phi(g): chi(k, jj, g) for g in G}
    for key2 in CHARS:
        if all(abs(chi(key2[0], key2[1], h) - vals[h]) < 1e-9 for h in G):
            return key2
    raise RuntimeError(key)


parent = {key: key for key in CHARS}


def find(x):
    while parent[x] != x:
        x = parent[x]
    return x


for key in CHARS:
    for (phi, _, _) in AUT.values():
        parent[find(key)] = find(char_after(key, phi))
ORBITS = {}
for key in CHARS:
    ORBITS.setdefault(find(key), []).append(key)
spec_class = {key: sig for sig, keys in classes.items() for key in keys}
assert sorted(len(v) for v in ORBITS.values()) == [1, 3, 3, 7, 21, 21]
assert all(len({spec_class[key] for key in orb}) == 1 for orb in ORBITS.values())
assert len(ORBITS) == len(classes)
print("     the group generated by the two lifts has orbits of sizes 1, 3, 3, 7, 21, 21 on the 56 "
      "characters, and these are exactly the 6 spectral classes: every cross-block coincidence of "
      "U_3 is explained by Aut(X_3) (x -> x+1, x -> 2x generate Z/7 x| Z/3)")
for kk in range(16):
    kinds = sorted(set(ctype(key) for key in cross[kk]), key=int)
    print(f"     value {math.degrees(reps0[kk]):8.3f} deg: multiplicity {mult0[kk]:2d}, in "
          f"{len(cross[kk]):2d} blocks (orders {{{','.join(kinds)}}})")

# edge orbits of the deck group (21 = 7 per type), for pull-back perturbations
orbit = np.full(m, -1)
for e in range(m):
    if orbit[e] < 0:
        orb = sorted(set(int(EP[g][e]) for g in G))
        assert len(orb) == len(G)
        orbit[orb] = orb[0]
ORB_IDS = {o: i for i, o in enumerate(sorted(set(orbit.tolist())))}
assert len(ORB_IDS) == 21
orbit_id = np.array([ORB_IDS[o] for o in orbit])

# ================================================================ (D1) symmetric partners of U_3
Rm, O = schur(U3, output="real")                   # U_3 = O R O^t, R block diagonal (normal)
Lam = np.zeros(N, dtype=complex)
Dref = np.ones(N)
Rbd = np.zeros_like(Rm)
i = 0
while i < N:
    if i + 1 < N and abs(Rm[i + 1, i]) > 1e-8:
        a, b = Rm[i, i], Rm[i, i + 1]
        assert abs(Rm[i + 1, i + 1] - a) < 1e-8 and abs(Rm[i + 1, i] + b) < 1e-8
        Lam[i], Lam[i + 1] = a + 1j * b, a - 1j * b
        Dref[i + 1] = -1.0
        Rbd[i:i + 2, i:i + 2] = Rm[i:i + 2, i:i + 2]
        i += 2
    else:
        Lam[i] = Rm[i, i]
        Rbd[i, i] = Rm[i, i]
        i += 1
assert np.abs(Rm - Rbd).max() < 1e-8 and np.abs(O @ Rbd @ O.T - U3).max() < 1e-8
S0 = (O * Lam) @ O.T                              # symmetric unitary, same spectrum as U_3
Z = (O * Dref) @ O.T                              # real symmetric involution, Z U_3 Z = U_3^{-1}
Yr = Dref[:, None] * Rbd                          # D R: real symmetric involution
assert np.abs(S0 - S0.T).max() < 1e-10 and np.abs(S0.conj().T @ S0 - np.eye(N)).max() < 1e-10
assert np.abs(Z @ U3 @ Z - U3.T).max() < 1e-10 and np.abs(Z @ Z - np.eye(N)).max() < 1e-10
assert np.abs(Yr - Yr.T).max() < 1e-10 and np.abs(Yr @ Yr - np.eye(N)).max() < 1e-10
phS = np.angle(np.linalg.eigvals(S0))
assert np.abs(np.sort(np.mod(phS, TWO_PI)) - np.sort(np.mod(ph0, TWO_PI))).max() < 1e-8
assert np.abs(U3 - U3.T).max() > 0.5              # U_3 itself is far from symmetric
EIG_S0 = []
for th in reps0:
    cols = np.abs(Lam - np.exp(1j * th)) < 1e-8
    EIG_S0.append(O[:, cols].astype(complex))
assert sum(B.shape[1] for B in EIG_S0) == N
print(f"(D1) real Schur form U_3 = O R O^t; S_0 = O Lambda O^t is a symmetric unitary with the "
      f"same spectrum (max |U_3 - U_3^t| = {np.abs(U3 - U3.T).max():.3f}); Z = O D O^t satisfies "
      f"Z U_3 Z = U_3^(-1), Z^2 = 1  {elapsed()}")


# ================================================================ perturbation families
def normalize(V):
    """Scale a Hermitian V so that Tr V^2 = N (eigenvalues of root-mean-square 1)."""
    return V * math.sqrt(N / float(np.real(np.trace(V @ V))))


E0 = np.flatnonzero(tail == 0)
by_tail, by_head = {}, {}
for e in E0:
    a, b = Y.edges[e]
    by_tail.setdefault(a, []).append(e)
    by_head.setdefault(b, []).append(e)
PAIRS = [(e, f) for grp in list(by_tail.values()) + list(by_head.values())
         for e, f in itertools.combinations(grp, 2)]
PAIRS = np.array(PAIRS)
assert len(PAIRS) == 2 * 56 * 21
BLIST = [B for (B, _) in BLOCK.values()]


def V_gue(rng):
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    return normalize((A + A.conj().T) / 2)


def V_goe(rng):
    A = rng.standard_normal((N, N))
    return normalize((A + A.T) / 2)


def V_diag(rng):
    """Random on-site phases d_e on the edges (a random U(1) gauge potential), compressed:
    V = W_0^t diag(d) W_0.  diag(d) itself does not preserve W, so exp(i eps diag(d)) would
    leave W; the compression is Hermitian and exp(i eps V) is unitary on W cap C^{E_0}."""
    d = rng.uniform(-math.pi, math.pi, size=m)
    return normalize(W0.T @ (d[:, None] * W0))


def V_local(rng):
    """Random symmetric 0/+-1 operator supported on pairs of type-0 edges that share a vertex
    (tail or head), plus a 0/+-1 diagonal, compressed to W cap C^{E_0}."""
    M = np.zeros((m, m))
    vals = rng.integers(-1, 2, size=len(PAIRS)).astype(float)
    M[PAIRS[:, 0], PAIRS[:, 1]] = vals
    M[PAIRS[:, 1], PAIRS[:, 0]] = vals
    M[E0, E0] = rng.integers(-1, 2, size=len(E0)).astype(float)
    return normalize(W0.T @ M @ W0)


def V_block(rng):
    """Deck-equivariant: independent GUE blocks on the 56 isotypic components."""
    V = np.zeros((N, N), dtype=complex)
    for B in BLIST:
        d = B.shape[1]
        A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        V += B @ ((A + A.conj().T) / 2) @ B.conj().T
    return normalize(V)


def V_pull(rng):
    """Deck-invariant on-site potential: d_e constant on the 21 deck orbits of edges, i.e. the
    pull-back of an on-site potential on X_3 (7 free parameters on the type-0 edges)."""
    x = rng.standard_normal(21)
    d = x[orbit_id]
    return normalize(W0.T @ (d[:, None] * W0))


def V_treal(rng):
    """Real symmetric V with T V T^{-1} = U_3 V U_3^{-1} for T = Z K: V = O (A + Y A Y) O^t."""
    A = rng.standard_normal((N, N))
    A = (A + A.T) / 2
    return normalize(O @ (A + Yr @ A @ Yr) @ O.T)


def expiH(w, v, eps):
    return (v * np.exp(1j * eps * w)) @ v.conj().T


def spectrum(fam, w, v, eps):
    if fam == "symS0":
        Eh = expiH(w, v, eps / 2)
        return np.angle(np.linalg.eigvals(Eh @ S0 @ Eh))
    return np.angle(np.linalg.eigvals(U3 @ expiH(w, v, eps)))


def first_order_gaps(bases, V):
    """Consecutive differences of the eigenvalues of P_k V P_k, pooled over the clusters."""
    gaps = []
    for B in bases:
        mu = np.sort(np.linalg.eigvalsh(B.conj().T @ V @ B))
        gaps += list(np.diff(mu))
    return np.array(gaps)


# ================================================================ (D2) trace obstruction
def kappa(U, V, b=1, c=2):
    Ub, Uc = np.linalg.matrix_power(U, b), np.linalg.matrix_power(U, c)
    return np.trace(V @ Ub @ V @ Uc @ V) - np.trace(V @ Uc @ V @ Ub @ V)


Vt = V_goe(rng)
kap = abs(kappa(U3, Vt))
kap_t = abs(kappa(U3, V_treal(rng)))
kap_s = abs(kappa(S0, Vt))
assert kap > 1e-3 and kap_t < 1e-8 and kap_s < 1e-8
# explicit check of the time-reversal relations T U T^-1 = U^-1 for both symmetric families
w_, v_ = np.linalg.eigh(Vt)
Eh = expiH(w_, v_, 0.35)
Us = Eh @ S0 @ Eh
assert np.abs(Us - Us.T).max() < 1e-10                      # K U'_eps K = U'^{-1}_eps
Vtr = V_treal(rng)
w_, v_ = np.linalg.eigh(Vtr)
Ue = U3 @ expiH(w_, v_, 0.7)
assert np.abs(Z @ Ue.conj() @ Z - Ue.conj().T).max() < 1e-10  # (Z K) U_eps (Z K)^-1 = U_eps^-1
w_, v_ = np.linalg.eigh(Vt)
Ue = U3 @ expiH(w_, v_, 0.7)
assert np.abs(Z @ Ue.conj() @ Z - Ue.conj().T).max() > 0.1   # ... but not for generic real V
print(f"(D2) trace obstruction kappa(V) = Tr(V U V U^2 V) - Tr(V U^2 V U V): |kappa| = {kap:.3f} "
      f"for a GOE-type V with U = U_3 (no antiunitary time reversal of the family), "
      f"{kap_t:.1e} for T-real V (T = Z K), {kap_s:.1e} for U = S_0 and real V")

# ================================================================ (B) the crossover
EPS = np.logspace(-4, 1, 26)                      # 26 points, ratio 10^0.2 between neighbours
NS = 20
FAMS = {"GUE": V_gue, "GOE": V_goe, "onsite": V_diag, "local": V_local,
        "symS0": V_goe, "block": V_block}
RMEAN, FMEAN, GAPS = {}, {}, {}
for fam, gen in FAMS.items():
    R = np.zeros((NS, len(EPS)))
    F = np.zeros((NS, len(EPS)))
    gaps = []
    for s_ in range(NS):
        V = gen(rng)
        w, v = np.linalg.eigh(V)
        gaps.append(first_order_gaps(EIG_S0 if fam == "symS0" else EIG_U3, V))
        for k, eps in enumerate(EPS):
            R[s_, k], F[s_, k], _ = circle_stats(spectrum(fam, w, v, eps))
    RMEAN[fam], FMEAN[fam], GAPS[fam] = R.mean(0), F.mean(0), np.concatenate(gaps)
    print(f"     family {fam:6s} done  {elapsed()}")


def crossover(fam):
    r, f, gaps = RMEAN[fam], FMEAN[fam], GAPS[fam]
    # observed eps_half: log-interpolated crossing of the small-spacing fraction through 0.5
    k = int(np.argmax(f < 0.5))
    assert k > 0 and f[k - 1] >= 0.5
    t = (f[k - 1] - 0.5) / (f[k - 1] - f[k])
    eps_half = float(np.exp(np.log(EPS[k - 1]) + t * (np.log(EPS[k]) - np.log(EPS[k - 1]))))
    # observed eps_sat: first grid point at which <r> comes within 0.01 of its saturation value
    # (the mean over eps >= 2); <r> rises monotonically up to there, and later points fluctuate
    # by the sampling error of about 0.004
    r_sat = float(r[EPS >= 2].mean())
    k_sat = int(np.argmax(r >= r_sat - 0.01))
    eps_sat = float(EPS[k_sat])
    # first-order prediction: within-cluster splittings eps*gap; fraction below 0.1 mean spacing
    fine = np.logspace(-4, 1, 2001)
    f_pred = np.array([(eps * gaps < 0.1 * MEAN_SP).sum() / (NS * N) for eps in fine])
    eps_half_pred = float(fine[np.argmax(f_pred < 0.5)])
    eps_one = MEAN_SP / float(gaps.mean())            # mean first-order splitting = mean spacing
    return dict(eps_half=eps_half, eps_half_pred=eps_half_pred, eps_sat=eps_sat, eps_one=eps_one,
                r_sat=r_sat, gap_mean=float(gaps.mean()), gap_med=float(np.median(gaps)))


CROSS = {fam: crossover(fam) for fam in FAMS}
print(f"(B) crossover window ({len(EPS)} eps values 1e-4..10, {NS} samples; Tr V^2 = N for every V):")
print("     family   <r> at eps = 1e-3, 1e-2, 0.025, 0.063, 0.1, 0.25, 0.63, 1, 2.5, 10 | "
      "frac(<0.1 mean) at the same eps")
sel = [np.argmin(np.abs(EPS - e)) for e in (1e-3, 1e-2, 0.025, 0.063, 0.1, 0.25, 0.63, 1, 2.5, 10)]
for fam in FAMS:
    rr = " ".join(f"{RMEAN[fam][k]:.3f}" for k in sel)
    ff = " ".join(f"{FMEAN[fam][k]:.3f}" for k in sel)
    print(f"     {fam:6s}   {rr} | {ff}")
print("     family   eps_half(obs)  eps_half(1st order)  ratio | eps_sat(obs)  eps_1 = (2pi/N)/mean gap"
      "  ratio | <r>_sat")
for fam, c in CROSS.items():
    print(f"     {fam:6s}   {c['eps_half']:.4f}         {c['eps_half_pred']:.4f}          "
          f"{c['eps_half'] / c['eps_half_pred']:.2f} | {c['eps_sat']:.4f}        {c['eps_one']:.4f}"
          f"                   {c['eps_sat'] / c['eps_one']:.2f} | {c['r_sat']:.4f}")
print(f"     naive scale 2 pi / N = {MEAN_SP:.4f}; mean first-order gaps: "
      + ", ".join(f"{fam} {c['gap_mean']:.4f}" for fam, c in CROSS.items()))
for fam in ("GUE", "GOE", "onsite", "local"):
    c = CROSS[fam]
    assert FMEAN[fam][EPS <= 1e-2].min() > 0.9                         # still 93% clustered
    assert 1 / 3 < c["eps_half"] / c["eps_half_pred"] < 3                # first order predicts eps_half
    # saturation needs splittings of a few mean spacings; observed eps_sat / eps_1 in [1.5, 3.3]
    # (grid-resolved to a factor 10^0.2), so the assertion is a factor 4, not 3
    assert 1 / 4 < c["eps_sat"] / c["eps_one"] < 4
    assert abs(c["r_sat"] - R_GUE) < 0.015                              # GUE at saturation
assert abs(CROSS["symS0"]["r_sat"] - R_GOE) < 0.015                     # GOE for the symmetric family
assert 1 / 3 < CROSS["symS0"]["eps_half"] / CROSS["symS0"]["eps_half_pred"] < 3
assert CROSS["block"]["r_sat"] < R_GOE - 0.05                           # superposition, not Wigner-Dyson
print("     checks: frac(<0.1 mean) > 0.9 for eps <= 1e-2; eps_half within a factor 3 (observed: "
      "within 10%) and eps_sat within a factor 4 (observed: 1.5-3.3) of the first-order scales; "
      "<r>_sat within 0.015 of 0.5996 (GUE, GOE, onsite, local), of 0.5307 (symS0); "
      "block-diagonal V below 0.48")


# ================================================================ (C) saturated spectra
def haar_unitary(rng):
    A = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / math.sqrt(2)
    Qm, Rq = np.linalg.qr(A)
    d = np.diag(Rq)
    return Qm * (d / np.abs(d))


def haar_orthogonal(rng):
    Qm, Rq = np.linalg.qr(rng.standard_normal((N, N)))
    return Qm * np.sign(np.diag(Rq))


EPS_SAT, NS2 = 3.0, 40
SAT = {}
for fam, gen in list(FAMS.items()) + [("treal", V_treal), ("pull", V_pull)]:
    rs, fs, sp, nz = [], [], [], 0
    for s_ in range(NS2):
        V = gen(rng)
        w, v = np.linalg.eigh(V)
        ph = spectrum(fam, w, v, EPS_SAT)
        r, f, s = circle_stats(ph)
        rs.append(r), fs.append(f), sp.append(s)
        nz += int((s < 1e-6).sum())
    SAT[fam] = dict(r=float(np.mean(rs)), f=float(np.mean(fs)), s=np.concatenate(sp), zeros=nz)
for ref in ("CUE", "COE", "O(N)"):
    rs, sp = [], []
    for s_ in range(NS2):
        if ref == "CUE":
            r, f, s = circle_stats(np.angle(np.linalg.eigvals(haar_unitary(rng))))
        elif ref == "COE":
            Uh = haar_unitary(rng)
            r, f, s = circle_stats(np.angle(np.linalg.eigvals(Uh.T @ Uh)))
        else:
            ph = np.angle(np.linalg.eigvals(haar_orthogonal(rng)))
            r, f, s = line_stats(ph[(ph > 1e-9) & (ph < math.pi - 1e-9)])
        rs.append(r), sp.append(s)
    SAT[ref] = dict(r=float(np.mean(rs)), f=float("nan"), s=np.concatenate(sp), zeros=0)


def brody_alpha(b):
    return math.gamma((b + 2) / (b + 1)) ** (b + 1)


def cdf_brody(s, b):
    return 1 - np.exp(-brody_alpha(b) * s ** (b + 1))


def cdf_goe(s):
    return 1 - np.exp(-math.pi * s ** 2 / 4)


def cdf_gue(s):
    return erf(2 * s / math.sqrt(math.pi)) - (4 * s / math.pi) * np.exp(-4 * s ** 2 / math.pi)


def cdf_poisson(s):
    return 1 - np.exp(-s)


def ks(s, F):
    s = np.sort(s)
    nn = len(s)
    Fs = F(s)
    return float(max(np.abs(np.arange(1, nn + 1) / nn - Fs).max(), np.abs(np.arange(nn) / nn - Fs).max()))


def kl(s, F, edges=np.r_[np.arange(0, 5, 0.1), np.inf]):
    p, _ = np.histogram(s, bins=edges)
    p = p / p.sum()
    qq = np.diff(F(np.minimum(edges, 1e9)))
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / qq[mask])))


def brody_mle(s):
    s = s[s > 1e-12]
    nll = lambda b: -np.sum(np.log(b + 1) + np.log(brody_alpha(b)) + b * np.log(s)
                            - brody_alpha(b) * s ** (b + 1))
    return float(minimize_scalar(nll, bounds=(0.01, 6.0), method="bounded").x)


print(f"(C) saturated regime eps = {EPS_SAT} ({NS2} samples, {NS2 * N} unfolded spacings each; "
      f"Haar references at N = {N}):")
print("     family   <r>    zeros  Brody b   KS:Poisson  GOE    GUE    Brody2  Brody(b) | KL: GOE    GUE")
FIT = {}
for fam, d in SAT.items():
    s = d["s"]
    b = brody_mle(s)
    row = dict(b=b, ks_p=ks(s, cdf_poisson), ks_goe=ks(s, cdf_goe), ks_gue=ks(s, cdf_gue),
               ks_b2=ks(s, lambda x: cdf_brody(x, 2.0)), ks_bb=ks(s, lambda x: cdf_brody(x, b)),
               kl_goe=kl(s, cdf_goe), kl_gue=kl(s, cdf_gue))
    FIT[fam] = row
    print(f"     {fam:6s}   {d['r']:.4f} {d['zeros']:5d}  {b:5.2f}     {row['ks_p']:.4f}      "
          f"{row['ks_goe']:.4f} {row['ks_gue']:.4f} {row['ks_b2']:.4f}  {row['ks_bb']:.4f}   | "
          f"{row['kl_goe']:.4f} {row['kl_gue']:.4f}")
sup_b2_gue = float(np.abs(cdf_brody(np.linspace(0, 6, 60001), 2.0) - cdf_gue(np.linspace(0, 6, 60001))).max())
print(f"     sup |F_Brody2 - F_GUE| = {sup_b2_gue:.4f} (CDFs); the surmises are N = 2 formulas, so the "
      f"Haar rows calibrate the finite-N and finite-sample distances")
# The maximum-likelihood Brody parameter of a GUE-type spectrum is NOT 2: for the four
# non-symmetric families and for Haar CUE at N = 226 it is 1.5 +- 0.05.  The requested window
# (1.6, 2.4) is therefore replaced by the observed one, (1.3, 1.7); the KS distance to the GUE
# surmise is an order of magnitude below the distance to the Brody b = 2 curve.
for fam in ("GUE", "GOE", "onsite", "local"):
    assert abs(SAT[fam]["r"] - R_GUE) < 0.015 and SAT[fam]["zeros"] == 0
    assert 1.3 < FIT[fam]["b"] < 1.7
    assert FIT[fam]["ks_gue"] < FIT[fam]["ks_b2"] / 3 and FIT[fam]["ks_gue"] < FIT[fam]["ks_goe"]
    assert FIT[fam]["kl_gue"] < FIT[fam]["kl_goe"]
for fam in ("symS0", "treal"):
    assert abs(SAT[fam]["r"] - R_GOE) < 0.015
    assert 0.8 < FIT[fam]["b"] < 1.1
    assert FIT[fam]["ks_goe"] < FIT[fam]["ks_gue"] and FIT[fam]["kl_goe"] < FIT[fam]["kl_gue"]
assert abs(SAT["CUE"]["r"] - R_GUE) < 0.015 and abs(SAT["COE"]["r"] - R_GOE) < 0.015
assert abs(SAT["O(N)"]["r"] - R_GUE) < 0.015 and 1.3 < FIT["CUE"]["b"] < 1.7
assert SAT["block"]["r"] < R_GOE - 0.05 and SAT["pull"]["r"] < R_GOE - 0.05
print("     checks: non-symmetric families: <r> within 0.015 of GUE, Brody MLE b in (1.3, 1.7) "
      "(as for Haar CUE; b = 2 is not the GUE value), KS to the GUE surmise < 1/3 of the KS to the "
      "Brody b = 2 curve and below the KS to GOE, KL likewise; symmetric families (symS0, treal): "
      "<r> within 0.015 of GOE, b in (0.8, 1.1); Haar O(N) half-spectrum is GUE-like; "
      "deck-equivariant V (block, pull) stay below 0.48")

# ================================================================ (E4) deck-equivariant perturbations
# the perturbed spectrum of a deck-equivariant family is the union of the 56 block spectra;
# count how many of the 16 x (blocks) cross-block coincidences survive at saturation
for fam in ("block", "pull"):
    print(f"(E4) deck-equivariant {fam:5s}: <r>_sat = {SAT[fam]['r']:.4f}, fraction of spacings "
          f"< 0.1 mean = {SAT[fam]['f']:.3f}, exact zero spacings {SAT[fam]['zeros']} of "
          f"{NS2 * N}  (superposition of 56 blocks: Poisson 0.3863, GOE 0.5307)")
V = V_block(rng)
assert all(np.abs(DG[g] @ V - V @ DG[g]).max() < 1e-8 for g in (g1, g2))
V = V_pull(rng)
assert all(np.abs(DG[g] @ V - V @ DG[g]).max() < 1e-8 for g in (g1, g2))
w, v = np.linalg.eigh(V)
Ue = U3 @ expiH(w, v, 1.0)
for key, (B, _) in list(BLOCK.items())[:8]:
    assert np.abs(Ue @ B - B @ (B.conj().T @ Ue @ B)).max() < 1e-8   # blocks stay invariant
V = V_gue(rng)
assert max(np.abs(DG[g] @ V - V @ DG[g]).max() for g in (g1, g2)) > 0.1  # generic V does not
print("(E5) block-diagonal and pull-back V commute with the deck group and U_eps preserves the 56 "
      "blocks; a generic V does not commute")

# (E7) pull-backs of on-site potentials from X_3 keep an exact degeneracy: in every sample the
# three blocks of each class {chi, chi^2, chi^4} of characters of order 7 stay isospectral
# (8 triply degenerate values, i.e. 16 zero spacings per sample).  Observed, not explained.
assert SAT["pull"]["zeros"] == 16 * NS2
V = V_pull(rng)
w, v = np.linalg.eigh(V)
Ue = U3 @ expiH(w, v, 1.0)
rp, mu = clusters(np.angle(np.linalg.eigvals(Ue)))
assert sorted(mu.tolist(), reverse=True)[:9] == [3] * 8 + [1]
for kk in np.flatnonzero(mu > 1):
    who = [key for key, (B, _) in BLOCK.items()
           if np.min(np.abs(np.exp(1j * np.angle(np.linalg.eigvals(B.conj().T @ Ue @ B)))
                            - np.exp(1j * rp[kk]))) < 1e-6]
    js = sorted(key[1] for key in who)
    assert all(key[0] == (0, 0, 0) for key in who) and js in ([1, 2, 4], [3, 5, 6])
assert SAT["block"]["zeros"] == 0
print(f"(E7) pull-back on-site V (7 parameters on the type-0 edges of X_3): at eps = 1 the "
      f"spectrum has 8 triple eigenphases, each shared by the three order-7 blocks of one class "
      f"{{chi, chi^2, chi^4}} ({SAT['pull']['zeros']} = 16 x {NS2} zero spacings at eps = 3); "
      f"independent GUE blocks (block) lift every coincidence ({SAT['block']['zeros']} zero "
      f"spacings) but give <r> = {SAT['block']['r']:.4f} (Poisson 0.3863)")

print(f"\nALL CHECKS PASSED: (A) (B) (C) (D1) (D2) (E1)-(E7).   {elapsed()}")
