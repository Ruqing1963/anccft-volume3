"""Verification battery for Volume III, Paper 5 (chapters/ch05-chamber-games.tex).

  (R) reduction: the chamber public-goods payoff equals the additive pairwise payoff
      (r/3 - 1) D_2 s_i + (r/3) sum_j w_ij s_j, with w_ij = q+1 on edges (Y_21, Y_168, T_9);
  (Dm) switching any vertex from D to C changes its own payoff by D_2 (r/3 - 1);
  (Z) zero-noise thresholds: single-mutant threshold 3 kappa/(kappa-1); boundary margins
      delta(A) of stars, chambers, edges; star of Y_168 stable exactly for r > 7/3;
      5/3 is not of the form 3 kappa/(kappa + delta) with delta an integer (q = 2);
  (MC) Fermi dynamics (K = 0.1): cooperator fraction on Y_168, a random 14-regular graph
      and the thin torus T_12 (seeded);
  (P) rock-paper-scissors: d/dt log(x1 x2 x3) = (a-b)(1 - 3 sigma); replicator-mutator
      Hopf point mu_H = (b-a)/18, stable cycle with amplitude^2 ~ (1 - mu/mu_H), unique
      interior equilibrium on a grid;
  (E) |F|/|V| = kappa(q+1)/3 and the thresholds as functions of q;
  (NL1)-(NL5) nonlinear chamber games (Section 5): pairwise iff f affine; exact potential and
      the type-stratified Nash window 3/2 <= r <= 3; exhaustive best cages on Y_21 and on random
      tripartite / random 21-regular hypergraphs; closed-form star cages (q = 1, 2); Fermi
      dynamics of the threshold game on Y_21, T_12, Y_168 and a random hypergraph;
  (NC1)-(NC6) chiral games (Section 6): Monderer-Shapley circulation 2/7 on Y_21 and a strict
      improvement 6-cycle (no potential); Lyapunov function for potential mean fields; Hopf at
      K_H = 1/4 (supply chain) and 0.170582 (three-body), period 2 pi/sqrt 3; unique equilibrium;
      flat interfaces of T_12; shrinking class-density oscillations on larger complexes;
  (NA1)-(NA5) clocks and anchors (Section 7): the gauge clock reaches the twisted ground state;
      linear response X_m(omega) against the forced ODE; coherence with a synchronized clock;
      zero-noise spreading from anchored balls (building, level-14 cover) and walls (thin);
      finite-noise anchors on Y_168 and T_12.

Run:  python code/v3p5_verify.py      (numpy only)
"""
import math
import time
from fractions import Fraction

import numpy as np

from a2_common import standard_covers, thin_torus

t0 = time.time()
COV = standard_covers()
Y21, Y84, Y168 = COV["Y_21"], COV["Y_84"], COV["Y_168"]
T9, T12 = thin_torus(9), thin_torus(12)


def weights(X):
    W = np.zeros((X.n, X.n))
    for t in X.tris:
        for a in t:
            for b in t:
                if a != b:
                    W[a, b] += 1
    D2 = np.bincount(np.array(X.tris).ravel(), minlength=X.n).astype(float)
    return W, D2


def chamber_payoff(X, s, r):
    pi = np.zeros(X.n)
    for t in X.tris:
        nc = sum(s[v] for v in t)
        for v in t:
            pi[v] += r / 3 * nc - s[v]
    return pi


# ---------------------------------------------------------------- (R) reduction, (Dm)
rng = np.random.default_rng(0)
for X in (Y21, Y168, T9):
    W, D2 = weights(X)
    adj = (X.A1 + X.A1.T) > 0
    assert set(W[adj].astype(int)) == {X.q + 1} and not W[~adj].any()
    assert set(D2.astype(int)) == {(X.q ** 2 + X.q + 1) * (X.q + 1)}
    for _ in range(10):
        s = rng.integers(0, 2, X.n).astype(float)
        r = rng.uniform(1, 5)
        red = (r / 3 - 1) * D2 * s + r / 3 * W @ s
        assert np.allclose(chamber_payoff(X, s, r), red)
        i = rng.integers(X.n)
        s0, s1 = s.copy(), s.copy()
        s0[i], s1[i] = 0, 1
        assert abs(chamber_payoff(X, s1, r)[i] - chamber_payoff(X, s0, r)[i]
                   - D2[i] * (r / 3 - 1)) < 1e-9
print("(R) chamber PGG payoff = (r/3-1) D_2 s_i + (r/3) (q+1) N_C(i) on Y_21, Y_168, T_9; "
      "w = q+1 on edges, 0 elsewhere")
print("(Dm) own switch D -> C changes own payoff by D_2 (r/3 - 1) in every configuration")

# ---------------------------------------------------------------- (Z) zero-noise thresholds
KAP = 7


def delta(adj, A):
    inA = np.zeros(len(adj), bool)
    inA[list(A)] = True
    NA = adj[:, inA].sum(1)
    return min(int(NA[i] - NA[j]) for i in np.nonzero(inA)[0] for j in np.nonzero(adj[i] & ~inA)[0])


def norm_payoff(adj, s, r, kap):
    return (r / 3 - 1) * s + r / (3 * kap) * (adj @ s)


# single mutants on Y_168
adj = (Y168.A1 + Y168.A1.T) > 0
rstar = 3 * KAP / (KAP - 1)
for r, d_wins in [(rstar - 0.01, True), (rstar + 0.01, False)]:
    s = np.ones(Y168.n)
    s[0] = 0
    p = norm_payoff(adj, s, r, KAP)
    nb = np.nonzero(adj[0])[0]
    assert (p[0] > p[nb].max()) == d_wins
    s = np.zeros(Y168.n)
    s[0] = 1
    p = norm_payoff(adj, s, r, KAP)
    assert (p[0] > p[nb].max()) == (not d_wins)
DEL = {}
for X in (Y21, Y84, Y168, T12):
    a = (X.A1 + X.A1.T) > 0
    DEL[X.name] = dict(
        star=sorted({delta(a, [v] + list(np.nonzero(a[v])[0])) for v in range(X.n)}),
        chamber=sorted({delta(a, list(t)) for t in X.tris}),
        edge=sorted({delta(a, list(e)) for e in X.edges}))
assert DEL["Y_168"] == dict(star=[2], chamber=[0], edge=[-1])
assert DEL["Y_84"]["star"] == [0] and DEL["Y_21"]["star"] == [-6]
assert DEL["T_12"]["star"] == [1]
strip = [v for v in range(T12.n) if v // 12 < 6]
aT = (T12.A1 + T12.A1.T) > 0
assert delta(aT, strip) == 2
# the star of Y_168 is stable (no boundary C loses) exactly for r > 21/9 = 7/3
star = [0] + list(np.nonzero(adj[0])[0])
s = np.zeros(Y168.n)
s[star] = 1
for r, stable in [(7 / 3 + 0.005, True), (7 / 3 - 0.005, False)]:
    p = norm_payoff(adj, s, r, KAP)
    worst = min(p[i] - p[j] for i in star for j in np.nonzero(adj[i] & (s == 0))[0])
    assert (worst > 0) == stable
# 5/3 is not a zero-noise cluster threshold for q = 2
assert all(Fraction(21, 7 + d) != Fraction(5, 3) for d in range(-6, 14))
print(f"(Z) single-mutant threshold 3k/(k-1) = {rstar}; delta: {DEL}")
print("    T_12 half-torus strip delta = 2; star of Y_168 stable iff r > 7/3; 5/3 is not "
      "21/(7+delta) for integer delta")


# ---------------------------------------------------------------- (MC) Fermi dynamics
def random_regular(n, d, seed):
    """Circulant C_n(1..d/2) randomized by 40|E| degree-preserving double-edge swaps."""
    rg = np.random.default_rng(seed)
    E = {tuple(sorted((i, (i + k) % n))) for i in range(n) for k in range(1, d // 2 + 1)}
    E = list(E)
    S = set(E)
    for _ in range(40 * len(E)):
        i, j = rg.integers(len(E), size=2)
        (a, b), (c, e) = E[i], E[j]
        if len({a, b, c, e}) < 4:
            continue
        n1, n2 = tuple(sorted((a, c))), tuple(sorted((b, e)))
        if n1 in S or n2 in S:
            continue
        S -= {E[i], E[j]}
        S |= {n1, n2}
        E[i], E[j] = n1, n2
    A = np.zeros((n, n))
    for a, b in E:
        A[a, b] = A[b, a] = 1
    assert set(A.sum(1)) == {d}
    return A


def fermi_runs(W, D2, r, K, R=60, T=400, seed=2, x0=0.5):
    rg = np.random.default_rng(seed)
    n = len(D2)
    nbrs = [np.nonzero(W[i])[0] for i in range(n)]
    maxd = max(len(x) for x in nbrs)
    NB = np.array([np.r_[x, [x[0]] * (maxd - len(x))] for x in nbrs]).astype(int)
    deg = np.array([len(x) for x in nbrs])
    S = (rg.random((R, n)) < x0).astype(float)
    ar = np.arange(R)
    for _ in range(T * n):
        i = rg.integers(0, n, R)
        j = NB[i, (rg.random(R) * deg[i]).astype(int)]
        si, sj = S[ar, i], S[ar, j]
        pi_i = ((r / 3 - 1) * D2[i] * si + r / 3 * np.einsum('ij,ij->i', W[i], S)) / D2[i]
        pi_j = ((r / 3 - 1) * D2[j] * sj + r / 3 * np.einsum('ij,ij->i', W[j], S)) / D2[j]
        p = 1 / (1 + np.exp(np.clip((pi_i - pi_j) / K, -50, 50)))
        adopt = rg.random(R) < p
        S[ar[adopt], i[adopt]] = sj[adopt]
    return float(S.mean())


GRAPHS = {"Y_168": weights(Y168), "T_12": weights(T12),
          "RRG14": (3 * random_regular(168, 14, 1), np.full(168, 21.0))}
MC = {}
for name, (W, D2) in GRAPHS.items():
    MC[name] = {r: fermi_runs(W, D2, r, K=0.1) for r in (2.7, 2.9, 3.0, 3.1)}
    print(f"(MC) K = 0.1, {name:6s}: " + "  ".join(f"r={r}: {v:.2f}" for r, v in MC[name].items()))
assert MC["Y_168"][2.7] < 0.05 and MC["RRG14"][2.7] < 0.05 and MC["T_12"][2.7] > 0.2
assert MC["Y_168"][3.1] > 0.9 and MC["RRG14"][3.1] > 0.9


# ---------------------------------------------------------------- (P) rock-paper-scissors
def rps(a, b):
    return np.array([[0, a, -b], [-b, 0, a], [a, -b, 0]], float)


def field(y, A, mu):
    x = np.r_[y, 1 - y.sum()]
    Ax = A @ x
    return (x * (Ax - x @ Ax) + mu * (1 - 3 * x))[:2]


def rk4(y, A, mu, dt, steps):
    for _ in range(steps):
        k1 = field(y, A, mu)
        k2 = field(y + dt / 2 * k1, A, mu)
        k3 = field(y + dt / 2 * k2, A, mu)
        k4 = field(y + dt * k3, A, mu)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


for _ in range(500):                                       # log-derivative identity
    x = rng.dirichlet([1, 1, 1])
    a, b = rng.uniform(0.1, 3, 2)
    A = rps(a, b)
    Ax = A @ x
    dlogV = np.sum(Ax - x @ Ax)
    sig = x[0] * x[1] + x[1] * x[2] + x[2] * x[0]
    assert abs(dlogV - (a - b) * (1 - 3 * sig)) < 1e-12
a, b = 1.0, 2.0
A = rps(a, b)
muH = (b - a) / 18
J = np.zeros((2, 2))
yb = np.array([1 / 3, 1 / 3])
for k in range(2):
    e = np.zeros(2)
    e[k] = 1e-7
    J[:, k] = (field(yb + e, A, muH) - field(yb - e, A, muH)) / 2e-7
ev = np.linalg.eigvals(J)
assert np.abs(ev.real).max() < 1e-6 and abs(abs(ev[0].imag) - math.sqrt(3) * (a + b) / 6) < 1e-6
AMP = {}
for fr in (0.5, 0.9, 0.99, 1.05):
    amps = []
    for y0 in ([0.34, 0.33], [0.8, 0.1]):
        y = rk4(np.array(y0), A, fr * muH, 0.02, 250000)   # t = 5000
        xs = []
        for _ in range(1000):                               # sample one more stretch
            y = rk4(y, A, fr * muH, 0.02, 5)
            xs.append(y[0])
        amps.append(np.abs(np.array(xs) - 1 / 3).max())
    assert abs(amps[0] - amps[1]) < 1e-3                    # same attractor from both sides
    AMP[fr] = amps[0]
assert AMP[1.05] < 1e-4 and AMP[0.99] < AMP[0.9] < AMP[0.5]
assert abs(AMP[0.99] ** 2 / 0.01 - AMP[0.9] ** 2 / 0.1) < 0.05
for aa, bb, fr in [(1, 2, 0.5), (1, 3, 0.2), (0.5, 2, 0.8), (2, 1, 1.0)]:   # unique equilibrium
    Ai, mu = rps(aa, bb), fr * abs(bb - aa) / 18 + (0.01 if aa > bb else 0)
    g = np.linspace(0.002, 0.998, 400)
    X1, X2 = np.meshgrid(g, g)
    ok = X1 + X2 < 0.999
    pts = np.stack([X1[ok], X2[ok]], 1)
    F = np.array([field(p, Ai, mu) for p in pts])
    cand = pts[np.linalg.norm(F, axis=1) < 5e-3]
    assert len(cand) and np.abs(cand - 1 / 3).max() < 0.02
print(f"(P) d/dt log(x1x2x3) = (a-b)(1-3 sigma); mu_H = (b-a)/18 with frequency sqrt3(a+b)/6; "
      f"cycle amplitudes {dict((k, round(v, 4)) for k, v in AMP.items())}; unique equilibrium")

# ---------------------------------------------------------------- (E) combinatorial table
for q in (2, 3, 4, 5, 7, 8):
    kap = q * q + q + 1
    FV = Fraction(kap * (q + 1), 3)
    rs = Fraction(3 * kap, kap - 1)
    assert FV == Fraction((q + 1) * (q * q + q + 1), 3)
    print(f"(E) q={q}: |F|/|V| = {float(FV):.3f}, D_2 = {kap * (q + 1)}, mean-field r_c = 3, "
          f"single-mutant 3k/(k-1) = {float(rs):.4f}, spec 1+2/(q+1) = {1 + 2 / (q + 1):.4f}")

# ================================================================ (NL) nonlinear chamber games
PHI = {"linear": [0, 1, 2, 3], "threshold": [0, 0, 2, 3], "quadratic": [0, 1, 4, 9]}


def nl_payoff(tris, n, s, f):
    """pi_i = sum over chambers at i of f(n_C)/3 - s_i."""
    pi = np.zeros(n)
    for t in tris:
        nc = int(sum(s[v] for v in t))
        for v in t:
            pi[v] += f[nc] / 3 - s[v]
    return pi


# (NL1) multilinear expansion: f(n) = c1 n + c2 e2 + c3 e3 on {0,1}^3, pairwise iff c2 = c3 = 0
for name, phi in PHI.items():
    f = np.array(phi, float)
    c1, c2, c3 = f[1] - f[0], f[2] - 2 * f[1] + f[0], f[3] - 3 * f[2] + 3 * f[1] - f[0]
    for bits in [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]:
        a, b, c = bits
        assert abs(f[a + b + c] - (c1 * (a + b + c) + c2 * (a * b + a * c + b * c) + c3 * a * b * c)) < 1e-12
    assert (c2 == 0 and c3 == 0) == (name == "linear")
    if name == "threshold":
        assert (c1, c2, c3) == (0, 2, -3)
# the coefficient of s_j s_k in pi_i equals c2/3 (one chamber {i,j,k}) and of s_i s_j s_k equals c3/3
f = np.array(PHI["threshold"], float)
t0nl = Y21.tris[0]
i, j, k = t0nl
base = np.zeros(21)
vals = {}
for sj in (0, 1):
    for sk in (0, 1):
        s = base.copy()
        s[j], s[k] = sj, sk
        vals[(sj, sk)] = nl_payoff(Y21.tris, 21, s, f)[i]
assert abs(vals[(1, 1)] - vals[(1, 0)] - vals[(0, 1)] + vals[(0, 0)] - 2 / 3) < 1e-12   # c2/3
print("(NL1) f = c1 n + c2 e2 + c3 e3; pairwise iff c2 = c3 = 0 iff f affine; threshold f has "
      "(c1,c2,c3) = (0,2r,-3r); the s_j s_k coefficient in pi_i is c2/3 on Y_21")

# (NL2) exact potential Phi = sum_chambers (f(n)/3 - n); type-stratified Nash window
for X in (Y21, Y168, T12):
    for _ in range(20):
        r = rng.uniform(1, 5)
        f = r * np.array(PHI["threshold"], float)
        s = rng.integers(0, 2, X.n)
        v = rng.integers(X.n)
        s1, s0 = s.copy(), s.copy()
        s1[v], s0[v] = 1, 0
        pot = lambda z: sum(f[int(sum(z[w] for w in t))] / 3 - sum(z[w] for w in t) for t in X.tris)
        assert abs((nl_payoff(X.tris, X.n, s1, f)[v] - nl_payoff(X.tris, X.n, s0, f)[v])
                   - (pot(s1) - pot(s0))) < 1e-9
    tau = np.array(X.tau)
    for r, nash in [(1.4, False), (1.6, True), (2.9, True), (3.1, False)]:
        f = r * np.array(PHI["threshold"], float)
        s = (tau != 2).astype(int)                       # D = one type class
        pi = nl_payoff(X.tris, X.n, s, f)
        ok = True
        for v in range(0, X.n, max(1, X.n // 12)):
            z = s.copy()
            z[v] = 1 - z[v]
            ok &= nl_payoff(X.tris, X.n, z, f)[v] <= pi[v] + 1e-12
        assert ok == nash, (X.name, r)
print("(NL2) exact potential verified (Y_21, Y_168, T_12); D = one type class is a Nash "
      "equilibrium exactly for 3/2 <= r <= 3 (tested at 1.4, 1.6, 2.9, 3.1)")

# (NL3) exhaustive enumeration on 21-vertex hypergraphs
def best_cage(tris, n, want_potential=None, chunk=1 << 14):
    tris = np.array([sorted(t) for t in tris])
    F = len(tris)
    D2 = int(np.bincount(tris.ravel(), minlength=n)[0])
    Inc = np.zeros((F, n))
    Inc[np.repeat(np.arange(F), 3), tris.ravel()] = 1
    P = np.array(sorted({(a, b) for t in tris for a in t for b in t if a != b}))
    best = {k: (np.inf, None) for k in PHI}
    potmax = {}
    for start in range(1, 2 ** n - 1, chunk):
        masks = np.arange(start, min(start + chunk, 2 ** n - 1), dtype=np.int64)
        B = ((masks[:, None] >> np.arange(n)) & 1).astype(np.int8)
        nA = B[:, tris[:, 0]] + B[:, tris[:, 1]] + B[:, tris[:, 2]]
        valid = (B[:, P[:, 0]] == 1) & (B[:, P[:, 1]] == 0)
        for k, phi in PHI.items():
            Phi = np.array(phi, float)[nA] @ Inc / 3.0
            d = np.where(valid, Phi[:, P[:, 0]] - Phi[:, P[:, 1]], np.inf)
            dmin = d.min(1)
            okm = dmin > 1e-12
            if okm.any():
                idx = np.argmax(np.where(okm, dmin, -np.inf))
                if D2 / dmin[idx] < best[k][0]:
                    best[k] = (D2 / dmin[idx], int(masks[idx]))
        if want_potential:
            for r in want_potential:
                psi = r * np.array(PHI["threshold"], float) / 3 - np.arange(4)
                val = psi[nA].sum(1).max()
                potmax[r] = max(potmax.get(r, -np.inf), val)
    return best, potmax


def random_regular_triples(n, D2, seed, tripartite):
    rg = np.random.default_rng(seed)
    F = n * D2 // 3
    if tripartite:
        cols = [np.repeat(np.arange(c, n, 3), D2) for c in range(3)]
        for c in cols:
            rg.shuffle(c)
        E = np.stack(cols, 1)
    else:
        st = np.repeat(np.arange(n), D2)
        rg.shuffle(st)
        E = st.reshape(-1, 3)
    for _ in range(20000):
        keys = {}
        bad = [k for k in range(F) if len(set(E[k])) < 3]
        for k in range(F):
            keys.setdefault(tuple(sorted(E[k])), []).append(k)
        bad += [ks[1] for ks in keys.values() if len(ks) > 1]
        if not bad:
            return [tuple(e) for e in E]
        for k in bad:
            jj = rg.integers(F)
            a = rg.integers(3)
            b = a if tripartite else rg.integers(3)
            E[k][a], E[jj][b] = E[jj][b], E[k][a]
    raise RuntimeError("repair failed")


CAGE = {}
bY, potY = best_cage([tuple(t) for t in Y21.tris], 21, want_potential=(2.9, 3.1))
CAGE["Y_21"] = bY
CAGE["tripartite"] = best_cage(random_regular_triples(21, 21, 0, True), 21)[0]
CAGE["random"] = best_cage(random_regular_triples(21, 21, 0, False), 21)[0]
for name, b in CAGE.items():
    print(f"(NL3) best zero-noise cage, {name:10s}: " +
          ", ".join(f"{k} {v[0]:.4f}" for k, v in b.items()))
assert abs(CAGE["Y_21"]["linear"][0] - 3) < 1e-9 and abs(CAGE["Y_21"]["threshold"][0] - 2.25) < 1e-9
for k in PHI:                                            # Y_21 is not better than the random models
    assert CAGE["Y_21"][k][0] >= min(CAGE["tripartite"][k][0], CAGE["random"][k][0]) - 1e-9
assert potY[2.9] < -1e-9                                 # every proper state is below all-D (Phi = 0)
assert potY[3.1] < 147 * (3.1 - 3) - 1e-9                # every proper state is below all-C
print("    Y_21: all-D is the unique maximum of Phi at r = 2.9, all-C at r = 3.1: stochastic "
      "stability switches at r = 3, independent of geometry")

# (NL4) star cages: closed form versus exact computation (Y_168: q = 2, T_12: q = 1)
def star_threshold(X, v, phi):
    adj = (X.A1 + X.A1.T) > 0
    A = set([v] + list(np.nonzero(adj[v])[0]))
    at = {u: [] for u in range(X.n)}
    for t in X.tris:
        for u in t:
            at[u].append(set(t))
    D2 = len(at[0])
    Ph = lambda u: sum(Fraction(phi[len(c & A)], 3) for c in at[u])
    rr = Fraction(0)
    for i in A:
        for j in np.nonzero(adj[i])[0]:
            if j not in A:
                rr = max(rr, Fraction(D2) / (Ph(i) - Ph(j)))
    return rr


def star_formula(q, phi):
    kap = q * q + q + 1
    return Fraction(3 * (q + 1) * kap,
                    (q + 1) * (phi[3] + q * phi[2] + q * q * phi[1]) - phi[2] - 2 * q * phi[1])


for X, qq in ((Y168, 2), (T12, 1)):
    for k, phi in PHI.items():
        assert star_threshold(X, 0, phi) == star_formula(qq, phi)
print("(NL4) star cages: " + "; ".join(f"q={qq}: " + ", ".join(f"{k} {star_formula(qq, phi)}"
                                                           for k, phi in PHI.items()) for qq in (1, 2)))


# (NL5) Fermi dynamics for the threshold game (K = 0.1), per-chamber normalized payoffs
def incidence(tris, n):
    lst = [[] for _ in range(n)]
    for t in tris:
        for v in t:
            lst[v].append(tuple(t))
    return np.array([[list(t) for t in x] for x in lst])


def fermi_nl(CH, phi, r, K=0.1, R=60, T=300, x0=0.5, seed=2):
    rg = np.random.default_rng(seed)
    n, D2, _ = CH.shape
    nb = [np.unique(CH[i].ravel()) for i in range(n)]
    nb = [x[x != i] for i, x in enumerate(nb)]
    md = max(len(x) for x in nb)
    NB = np.array([np.r_[x, [x[0]] * (md - len(x))] for x in nb]).astype(int)
    deg = np.array([len(x) for x in nb])
    f = np.array(phi, float) * r / 3.0
    S = (rg.random((R, n)) < x0).astype(np.int64)
    ar = np.arange(R)

    def pay(i):
        return f[S[ar[:, None, None], CH[i]].sum(-1)].sum(1) / D2 - S[ar, i]

    for _ in range(T * n):
        i = rg.integers(0, n, R)
        jn = NB[i, (rg.random(R) * deg[i]).astype(int)]
        p = 1 / (1 + np.exp(np.clip((pay(i) - pay(jn)) / K, -50, 50)))
        ad = rg.random(R) < p
        S[ar[ad], i[ad]] = S[ar[ad], jn[ad]]
    return float(S.mean())


STRUCT = {"Y_21": incidence(Y21.tris, 21), "T_12": incidence(T12.tris, 144),
          "Y_168": incidence(Y168.tris, 168),
          "RH21": incidence(random_regular_triples(168, 21, 1, False), 168)}
NLMC = {}
for name, CH in STRUCT.items():
    NLMC[name] = {r: fermi_nl(CH, PHI["threshold"], r) for r in (2.25, 2.5, 2.75, 3.0)}
    print(f"(NL5) threshold game, K = 0.1, {name:6s}: " +
          "  ".join(f"r={r}: {v:.2f}" for r, v in NLMC[name].items()))
assert NLMC["T_12"][2.5] > NLMC["Y_168"][2.5] and NLMC["T_12"][2.5] > NLMC["Y_21"][2.5]
assert abs(NLMC["Y_168"][2.5] - NLMC["RH21"][2.5]) < 0.15

# ================================================================ (NC) chiral, non-potential games
def sig(z):
    return 1 / (1 + np.exp(-z))


def sc_gain(rho, su, sd, threebody=False):
    """Per-chamber gain of cooperating for role rho (upstream member su, downstream member sd)."""
    if rho == 0:
        return 1 - 2 * su                       # technology is held up by the channel
    if rho == 1:
        return 2 * su - 1                       # manufacturing matches technology
    return (4 * su * sd - 1) if threebody else (2 * su - 1)


def roles(X):
    tau = np.array(X.tau)
    at = [[] for _ in range(X.n)]
    for t in X.tris:
        for v in t:
            at[v].append(t)
    up = np.array([[next(w for w in t if tau[w] == (tau[v] - 1) % 3) for t in at[v]] for v in range(X.n)])
    dn = np.array([[next(w for w in t if tau[w] == (tau[v] + 1) % 3) for t in at[v]] for v in range(X.n)])
    return tau, up, dn


# (NC1) Monderer-Shapley circulation on an upstream edge of Y_21
tau21, up21, dn21 = roles(Y21)


def sc_payoff(i, s):
    return float(np.mean([s[i] * sc_gain(tau21[i], s[u], s[d]) for u, d in zip(up21[i], dn21[i])]))


i = 0
j = int(up21[0][0])                             # an upstream neighbour of vertex 0
circ = set()
for _ in range(10):
    s = rng.integers(0, 2, 21)
    val = 0.0
    for (si, sj, sign) in [(1, 1, 1), (0, 1, -1), (1, 0, -1), (0, 0, 1)]:
        z = s.copy()
        z[i], z[j] = si, sj
        val += sign * sc_payoff(i, z)
    for (si, sj, sign) in [(1, 1, 1), (1, 0, -1), (0, 1, -1), (0, 0, 1)]:
        z = s.copy()
        z[i], z[j] = si, sj
        val -= sign * sc_payoff(j, z)
    circ.add(round(val, 12))
assert {abs(c) for c in circ} == {round(3 * 2 / 21, 12)}   # |(q+1) a / D_2| = 2/7
# (NC2) a 6-cycle of strict improvements inside one chamber
prof = (0, 0, 0)
seq = [prof]
for rho in (0, 1, 2, 0, 1, 2):
    su, sd = prof[(rho - 1) % 3], prof[(rho + 1) % 3]
    better = 1 if sc_gain(rho, su, sd) > 0 else 0
    assert better != prof[rho]                  # the switch is a strict improvement
    prof = tuple(better if k == rho else prof[k] for k in range(3))
    seq.append(prof)
assert seq[-1] == (0, 0, 0) and len(set(seq[:-1])) == 6
print(f"(NC1) Monderer-Shapley circulation on an upstream edge = {circ.pop():.6f} = -2/7 != 0: "
      f"not a (weighted) potential game")
print(f"(NC2) strict improvement 6-cycle {seq}: no ordinal potential")

# (NC3) potential mean fields have the Lyapunov function Phi/K + sum H(x_rho)
for _ in range(20):
    Mg = rng.normal(size=(3, 3))
    Mg = (Mg + Mg.T) / 2
    np.fill_diagonal(Mg, 0)
    bvec = rng.normal(size=3)
    K = 0.2
    Gpot = lambda x: bvec + Mg @ x
    Lyap = lambda x: (bvec @ x + x @ Mg @ x / 2) / K - np.sum(x * np.log(x) + (1 - x) * np.log(1 - x))
    x = rng.uniform(0.1, 0.9, 3)
    prev = Lyap(x)
    for _ in range(2000):
        x = x + 0.01 * (sig(Gpot(x) / K) - x)
        cur = Lyap(x)
        assert cur >= prev - 1e-12
        prev = cur
print("(NC3) symmetric (potential) mean fields: Phi/K + entropy increases along logit dynamics")


# (NC4) Hopf bifurcation of the supply-chain and three-body mean fields
def G_sc(x):
    return np.array([1 - 2 * x[2], 2 * x[0] - 1, 2 * x[1] - 1])


def G_3b(x):
    return np.array([1 - 2 * x[2], 2 * x[0] - 1, 4 * x[0] * x[1] - 1])


def mf_run(G, K, T=300, dt=0.01):
    x = np.array([0.6, 0.5, 0.5])
    xs = []
    fld = lambda y: sig(G(y) / K) - y
    for k in range(int(T / dt)):
        k1 = fld(x)
        k2 = fld(x + dt / 2 * k1)
        k3 = fld(x + dt / 2 * k2)
        k4 = fld(x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if k > T / dt / 2:
            xs.append(x[0])
    xs = np.array(xs)
    zc = np.nonzero(np.diff(np.sign(xs - xs.mean())) > 0)[0]
    per = np.diff(zc).mean() * dt if len(zc) > 2 else float("nan")
    return (xs.max() - xs.min()) / 2, per


def jac(G, K, h=1e-6):
    x = np.full(3, 0.5)
    J = np.zeros((3, 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        J[:, k] = ((sig(G(x + e) / K) - x - e) - (sig(G(x - e) / K) - x + e)) / (2 * h)
    return np.linalg.eigvals(J)


for G, KH, name in ((G_sc, 0.25, "supply chain"), (G_3b, None, "three-body")):
    assert np.allclose(G(np.full(3, 0.5)), 0)
    if KH is None:
        rts = np.roots([1, 0, 4, 8])
        KH = max(rts.real) / 4
    assert abs(max(jac(G, KH).real)) < 1e-6 and max(jac(G, KH * 0.98).real) > 0 > max(jac(G, KH * 1.02).real)
    a_in, p_in = mf_run(G, KH * 0.9)
    a_out, _ = mf_run(G, KH * 1.1)
    assert a_in > 0.05 and a_out < 1e-3
    print(f"(NC4) {name}: Hopf at K_H = {KH:.6f}; amplitude {a_in:.4f} at 0.9 K_H (period {p_in:.3f}), "
          f"{a_out:.1e} at 1.1 K_H")
assert abs(mf_run(G_sc, 0.249)[1] - 2 * math.pi / math.sqrt(3)) < 0.05
# unique equilibrium for the negative-feedback cycle: the composed map is decreasing
for K in (0.05, 0.1, 0.3):
    F = lambda x0: sig((1 - 2 * sig((2 * sig((2 * x0 - 1) / K) - 1) / K)) / K)   # x0 -> x1 -> x2 -> x0
    grid = np.linspace(0, 1, 2001)
    vals = F(grid) - grid
    assert np.all(np.diff(F(grid)) <= 1e-15) and ((vals[:-1] > 0) != (vals[1:] > 0)).sum() == 1

# (NC5) flat interfaces of the thin lattice: r > 18 / (3 phi3 + phi2 - phi1)
def cage_T(A, phi):
    adjT = (T12.A1 + T12.A1.T) > 0
    A = set(A)
    at = {u: [] for u in range(T12.n)}
    for t in T12.tris:
        for u in t:
            at[u].append(set(t))
    Ph = lambda u: sum(Fraction(phi[len(c & A)], 3) for c in at[u])
    return max(Fraction(6) / (Ph(a) - Ph(b)) for a in A for b in np.nonzero(adjT[a])[0] if b not in A)


row = [v for v in range(T12.n) if v // 12 < 6]
for k, phi in PHI.items():
    assert cage_T(row, phi) == Fraction(18, 3 * phi[3] + phi[2] - phi[1])
print("(NC5) flat interface of T_12: 9/5, 18/11, 3/5 = 18/(3 phi3 + phi2 - phi1); one lonely "
      "chamber of 6 at the boundary (thick star: 12 of 21)")


# (NC6) network logit dynamics: the class-density oscillation shrinks with size
def torus_lite(m):
    idx = lambda a, b: (a % m) * m + (b % m)
    ds = [(1, 0), (0, 1), (-1, -1)]
    tr = set()
    for a in range(m):
        for b in range(m):
            for p in range(3):
                for qd in range(3):
                    if p != qd:
                        (p1, p2), (r1, r2) = ds[p], ds[qd]
                        tr.add(frozenset((idx(a, b), idx(a + p1, b + p2), idx(a + p1 + r1, b + p2 + r2))))

    class Lite:
        pass
    X = Lite()
    X.n, X.tau, X.tris = m * m, [(v // m + v % m) % 3 for v in range(m * m)], [tuple(t) for t in tr]
    return X


def logit_std(X, K, T=120, seed=0):
    rg = np.random.default_rng(seed)
    tau, up, dn = roles(X)
    n = X.n
    s = (rg.random(n) < 0.5).astype(int)
    rec = []
    every = max(1, n // 10)
    for k in range(int(T * n)):
        v = rg.integers(n)
        g = sc_gain(tau[v], s[up[v]], s[dn[v]]).mean()
        s[v] = rg.random() < sig(g / K)
        if k % every == 0:
            rec.append(s[tau == 0].mean())
    rec = np.array(rec[len(rec) // 3:])
    return float(rec.std())


STD = {name: logit_std(X, 0.1) for name, X in
       (("Y_24", COV["Y_24"]), ("Y_168", Y168), ("T_12", torus_lite(12)), ("T_48", torus_lite(48)))}
print("(NC6) std of the class density at K = 0.1: " + ", ".join(f"{k} {v:.3f}" for k, v in STD.items()))
assert STD["Y_168"] < 0.7 * STD["Y_24"] and STD["T_48"] < 0.5 * STD["T_12"]

# ================================================================ (NA) clocks, entrainment, anchors
from a2_common import LocalBuilding


def type_edges(X):
    tau = np.array(X.tau)
    E = {(a, b) for t in X.tris for a in t for b in t if tau[b] == (tau[a] + 1) % 3}
    return np.array(sorted(E))


def gauge_clock(X, alpha, K1=1.0, Tc=100, dt=0.05, seed=0):
    """Paper 3 flux sector (identical frequencies, rotating frame) from random phases."""
    rg = np.random.default_rng(seed)
    E = type_edges(X)
    deg = np.bincount(E.ravel(), minlength=X.n)
    aa = 2 * math.pi * alpha / 3
    th = rg.uniform(0, 2 * math.pi, X.n)

    def fl(th):
        sn = np.sin(th[E[:, 1]] - th[E[:, 0]] + aa)
        out = np.zeros(X.n)
        np.add.at(out, E[:, 0], sn)
        np.add.at(out, E[:, 1], -sn)
        return K1 * out / deg

    for _ in range(int(Tc / dt)):
        k1 = fl(th)
        k2 = fl(th + dt / 2 * k1)
        k3 = fl(th + dt / 2 * k2)
        k4 = fl(th + dt * k3)
        th = th + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    tau = np.array(X.tau)
    z = np.exp(1j * th)
    ords = [abs((z * np.exp(-2j * math.pi * m * tau / 3)).mean()) for m in range(3)]
    return th, int(np.argmax(ords)), max(ords)


# (NA1) the clock locks onto the twisted ground state m* = -round(alpha) (mod 3)
for X in (COV["Y_24"], Y168):
    for alpha, mstar in ((0.3, 0), (1.2, 2), (-0.8, 1)):
        _, m, rr = gauge_clock(X, alpha)
        assert m == mstar and rr > 0.999, (X.name, alpha, m, rr)
print("(NA1) gauge clock on Y_24, Y_168 reaches the twisted ground state m* = -round(alpha): "
      "alpha = 0.3, 1.2, -0.8 -> m = 0, 2, 1 with order > 0.999")


# (NA2) mean-field linear response: amplitude eps |X_m(omega)|, X_m = (i w + I - C)^-1 f_m / (4K)
def lin_amp(K, eps, w, m):
    Cm = np.zeros((3, 3))
    Cm[0, 2], Cm[1, 0], Cm[2, 1] = -2, 2, 2
    Cm /= 4 * K
    fm = np.exp(2j * math.pi * m * np.arange(3) / 3)
    return abs(np.linalg.solve(1j * w * np.eye(3) + np.eye(3) - Cm, fm)[0]) * eps / (4 * K)


def ode_amp(K, eps, w, m, Tt=300, dt=0.005):
    x = np.full(3, 0.5)
    xs = []
    ph = 2 * math.pi * m * np.arange(3) / 3
    fl = lambda y, t: sig((G_sc(y) + eps * np.cos(w * t + ph)) / K) - y
    for k in range(int(Tt / dt)):
        t = k * dt
        k1 = fl(x, t)
        k2 = fl(x + dt / 2 * k1, t + dt / 2)
        k3 = fl(x + dt / 2 * k2, t + dt / 2)
        k4 = fl(x + dt * k3, t + dt)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if t > Tt / 2:
            xs.append(x[0])
    return (max(xs) - min(xs)) / 2


Kf, wres = 0.4, math.sqrt(3) / 1.6
LIN = {m: lin_amp(Kf, 0.02, wres, m) for m in range(3)}
for m in range(3):
    assert abs(ode_amp(Kf, 0.02, wres, m) - LIN[m]) < 0.03 * LIN[m]
assert LIN[2] > LIN[0] > LIN[1]
curve = [lin_amp(Kf, 0.02, w, 0) for w in np.linspace(0.2, 3, 57)]
assert abs(np.linspace(0.2, 3, 57)[int(np.argmax(curve))] - wres) < 0.3
print(f"(NA2) linear response at resonance w = {wres:.4f}, eps = 0.02: |x_0| = "
      + ", ".join(f"m={m}: {v:.5f}" for m, v in LIN.items()) + " (ODE within 3%); peak near the "
      "endogenous frequency")


# (NA3) coherence on the network: synchronized clock versus random offsets (K = 0.4)
def forced_std(X, K, eps, w, clock, Tt=100, seed=0):
    rg = np.random.default_rng(seed)
    tau, up, dn = roles(X)
    n = X.n
    s = (rg.random(n) < 0.5).astype(int)
    rec = []
    every = max(1, n // 10)
    for k in range(int(Tt * n)):
        v = rg.integers(n)
        g = sc_gain(tau[v], s[up[v]], s[dn[v]]).mean()
        if clock is not None:
            g += eps * math.cos(w * k / n + clock[v])
        s[v] = rg.random() < sig(g / K)
        if k % every == 0:
            rec.append(s[tau == 0].mean())
    return float(np.std(rec[len(rec) // 3:]))


th168, _, _ = gauge_clock(Y168, 0.3)
coh = {"sync": forced_std(Y168, 0.4, 0.3, wres, th168),
       "random": forced_std(Y168, 0.4, 0.3, wres, np.random.default_rng(5).uniform(0, 2 * math.pi, 168)),
       "none": forced_std(Y168, 0.4, 0.3, wres, None)}
assert coh["sync"] > 1.5 * coh["random"] and coh["sync"] > 1.5 * coh["none"]
print("(NA3) Y_168, K = 0.4: std of x_0 with synchronized clock / random offsets / no clock = "
      + " / ".join(f"{v:.3f}" for v in coh.values()))

# (NA4) anchors at zero noise: spreading thresholds from anchored balls and walls
PHI3 = {"linear": [0, 1, 2, 3], "threshold": [0, 0, 2, 3], "quadratic": [0, 1, 4, 9]}
LB = LocalBuilding(14)
dLB = LB.ball(4)
assert [sum(1 for x in dLB.values() if x == r) for r in range(5)] == [1, 14, 98, 560, 2912]
chc = {}


def chambers_at(v):
    if v not in chc:
        chc[v] = LB.chambers(v)
    return chc[v]


SPREAD = {}
for Rb in (1, 2):
    A = {v for v in dLB if dLB[v] <= Rb}
    front = [v for v in dLB if dLB[v] == Rb + 1]
    for k, phi in PHI3.items():
        Ph = {}

        def Pv(u):
            if u not in Ph:
                Ph[u] = sum(Fraction(phi[sum(1 for w in c if w in A)], 3) for c in chambers_at(u))
            return Ph[u]

        SPREAD[(Rb, k)] = max(min(Fraction(21) / (Pv(i) - Pv(j)) for i in LB.nbrs(j)
                                  if i in A and Pv(i) > Pv(j)) for j in front)
assert SPREAD[(1, "threshold")] == Fraction(63, 19) and SPREAD[(2, "threshold")] == 3
assert SPREAD[(2, "linear")] == Fraction(21, 10) and SPREAD[(2, "quadratic")] == Fraction(21, 20)
# stability of the unanchored ball of radius 3 (the ring reached from an anchored ball of radius 2)
A3 = {v for v in dLB if dLB[v] <= 3}
stab3 = {}
for k, phi in PHI3.items():
    Ph = {}

    def Pv(u):
        if u not in Ph:
            Ph[u] = sum(Fraction(phi[sum(1 for w in c if w in A3)], 3) for c in chambers_at(u))
        return Ph[u]

    stab3[k] = max(Fraction(21) / (Pv(i) - Pv(j)) for i in A3 if dLB[i] == 3
                   for j in LB.nbrs(i) if j not in A3)
assert stab3 == {"linear": Fraction(7, 3), "threshold": Fraction(63, 19), "quadratic": Fraction(63, 55)}
T24 = torus_lite(24)
adj24 = {v: set() for v in range(T24.n)}
at24 = {v: [] for v in range(T24.n)}
for t in T24.tris:
    for a in t:
        at24[a].append(set(t))
        for b in t:
            if a != b:
                adj24[a].add(b)
Aw = {v for v in range(T24.n) if v // 24 < 6}
for k, phi in PHI3.items():
    Pw = lambda u: sum(Fraction(phi[len(c & Aw)], 3) for c in at24[u])
    sp = max(min(Fraction(6) / (Pw(i) - Pw(j)) for i in adj24[j] if i in Aw and Pw(i) > Pw(j))
             for j in range(T24.n) if j // 24 == 6)
    assert sp == Fraction(18, 3 * phi[3] + phi[2] - phi[1])
print("(NA4) building (level-14 cover, balls 1,14,98,560,2912): spreading from anchored balls "
      f"R=1: {SPREAD[(1, 'linear')]}, {SPREAD[(1, 'threshold')]}, {SPREAD[(1, 'quadratic')]}; "
      f"R=2: {SPREAD[(2, 'linear')]}, {SPREAD[(2, 'threshold')]}, {SPREAD[(2, 'quadratic')]}; "
      f"unanchored ball R=3 needs {stab3['linear']}, {stab3['threshold']}, {stab3['quadratic']}; "
      "thin anchored wall spreads at 9/5, 18/11, 3/5")


# (NA5) anchors at finite noise (threshold game, K = 0.1): placement does not matter
def fermi_anchor(CH, r, anchors, K=0.1, Rr=40, Tt=300, seed=2):
    rg = np.random.default_rng(seed)
    n, D2, _ = CH.shape
    nb = [np.unique(CH[i].ravel()) for i in range(n)]
    nb = [x[x != i] for i, x in enumerate(nb)]
    md = max(len(x) for x in nb)
    NB = np.array([np.r_[x, [x[0]] * (md - len(x))] for x in nb]).astype(int)
    deg = np.array([len(x) for x in nb])
    f = np.array(PHI3["threshold"], float) * r / 3.0
    S = (rg.random((Rr, n)) < 0.5).astype(np.int64)
    ar = np.arange(Rr)
    fixed = np.zeros(n, bool)
    fixed[list(anchors)] = True
    S[:, fixed] = 1
    pay = lambda i: f[S[ar[:, None, None], CH[i]].sum(-1)].sum(1) / D2 - S[ar, i]
    for _ in range(Tt * n):
        i = rg.integers(0, n, Rr)
        jn = NB[i, (rg.random(Rr) * deg[i]).astype(int)]
        p = 1 / (1 + np.exp(np.clip((pay(i) - pay(jn)) / K, -50, 50)))
        ad = (rg.random(Rr) < p) & ~fixed[i]
        S[ar[ad], i[ad]] = S[ar[ad], jn[ad]]
    return float(S[:, ~fixed].mean())


adj168 = (Y168.A1 + Y168.A1.T) > 0
starA = [0] + list(np.nonzero(adj168[0])[0])
randA = list(np.random.default_rng(9).choice(168, 15, replace=False))
CHY = STRUCT["Y_168"]
ANC = {(lab, r): fermi_anchor(CHY, r, A) for lab, A in (("none", []), ("star", starA), ("random", randA))
       for r in (2.25, 2.5)}
ANC[("T_12 wall", 2.25)] = fermi_anchor(STRUCT["T_12"], 2.25, [v for v in range(144) if v // 12 == 0])
print("(NA5) finite-noise anchors: " + ", ".join(f"{k[0]} r={k[1]}: {v:.2f}" for k, v in ANC.items()))
assert max(ANC[("none", 2.25)], ANC[("star", 2.25)], ANC[("random", 2.25)]) < 0.05
assert ANC[("star", 2.5)] > ANC[("none", 2.5)] and abs(ANC[("star", 2.5)] - ANC[("random", 2.5)]) < 0.1
assert ANC[("T_12 wall", 2.25)] > 0.3

print(f"\nALL CHECKS PASSED: (R) (Dm) (Z) (MC) (P) (E) (NL1)-(NL5) (NC1)-(NC6) (NA1)-(NA5).   [{time.time() - t0:.0f} s]")





