"""Volume III, Paper 3, open problem (a) of Remark V3P3:rem:scope: transverse Hopf bifurcations of
the type-cluster equilibria on Y_168 (chapters/ch03-phase-oscillators.tex, Section "Transverse Hopf bifurcations").

Model (V3P3:eq:model) with omega_j = omega, K_1 = 1; cluster equations (V3P3:eq:cluster); the
Jacobian at a cluster equilibrium is (+) M(lambda) (x) I_{dim E_lambda} (+) diag(-sigma_t)
(Theorem V3P3:thm:blocks).

  (A) spectrum of A_1 on Y_168: besides kappa = 7, the Z/3-orbits of e^{i pi/3} (contains -1;
      multiplicity 7) and of 2 sqrt 2 e^{+-i phi} (multiplicity 21 each); M(e^{i pi/3}) is similar
      to the real matrix M(-1);
  (S) scan: 150 random slices (alpha, Phi), K_2 in [-3, 3] (step 0.01), all cluster equilibria by
      vectorized Newton from 36 seeds; each stable equilibrium is continued one step, and the
      first instability is classified (cluster/real, sigma_t/real, transverse/real,
      transverse/complex = Hopf);
  (L) the Hopf locus in the (Phi, K_2) plane at alpha = 0.55 (72 values of Phi);
  (P) a representative Hopf point at (K_1, alpha, Phi) = (1, 0.55, -1.40): K_2*, Omega_H and
      d Re mu / d K_2 to 25 digits (mpmath, unknowns d_0, d_1, K_2, omega); the full 168 x 168
      Jacobian has exactly 42 eigenvalues on the imaginary axis (+- i Omega_H, 21 each), the
      rotation zero, and the rest in Re < 0; 1D continuation of the spectrum through K_2*;
  (D) full-network dynamics (RK4, dt = 0.25) at the point of (P): the transient oscillation at
      Omega_H above threshold, the end state (frequency-locked, stationary in the rotating frame),
      pure character modes, and the basin radius of the equilibrium below threshold;
  (C) survey of 6 representative Hopf points of (S): distance delta_next to the next crossing of
      another eigenvalue, and the end state of a run at delta_next/5 (T = 1500..4000).
  output: data/v3p3_hopf_search.json, v3p3_hopf_locus.png, v3p3_hopf_dynamics.png.

Run from code/:  python v3p3_hopf_search.py [--quick]   (numpy, mpmath, matplotlib; seeded;
                 about 25 min; --quick: 12 slices, 12 values of Phi, 3 survey points, a few minutes)
"""
import itertools
import json
import math
import os
import sys
import time
from collections import Counter

import mpmath as mp
import numpy as np

from a2_common import standard_covers

t0 = time.time()
TP = 2 * math.pi
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT, exist_ok=True)
KAP = 7
RES = {}
QUICK = "--quick" in sys.argv                    # smoke test: 12 slices, 12 values of Phi, 3 survey points


def elapsed():
    return f"[{time.time() - t0:6.0f} s]"


# ================================================================ (A) spectrum of A_1
Y = standard_covers()["Y_168"]
A1 = Y.A1.astype(float)
lam = np.linalg.eigvals(A1)
assert int((np.abs(lam) < 1e-8).sum()) == 18
reps = [l for l in lam if abs(l) > 1e-8 and -math.pi / 3 + 1e-6 <= np.angle(l) < math.pi / 3 + 1e-6]
groups = {}
for l in reps:
    groups.setdefault((round(l.real, 6), round(l.imag, 6)), []).append(l)
DIST = np.array([np.mean(groups[k]) for k in sorted(groups)])          # full double precision
MULT = np.array([len(groups[k]) for k in sorted(groups)])
KAPI = int(np.argmin(np.abs(DIST - KAP)))
ONE = int(np.argmin(np.abs(DIST - np.exp(1j * math.pi / 3))))
LP = int(np.argmin(np.abs(DIST - 2 * math.sqrt(2) * np.exp(0.1622j))))
LM = int(np.argmin(np.abs(DIST - 2 * math.sqrt(2) * np.exp(-0.1622j))))
assert abs(DIST[KAPI] - 7) < 1e-9 and MULT[KAPI] == 1
assert abs(DIST[ONE] - np.exp(1j * math.pi / 3)) < 1e-9 and MULT[ONE] == 7
assert abs(abs(DIST[LP]) - 2 * math.sqrt(2)) < 1e-9 and MULT[LP] == MULT[LM] == 21
assert abs(DIST[LM] - np.conj(DIST[LP])) < 1e-9 and len(DIST) == 4
PHI_L = float(np.angle(DIST[LP]))
big, unit = lam[np.abs(np.abs(lam) - 2 * math.sqrt(2)) < 1e-6], lam[np.abs(np.abs(lam) - 1) < 1e-6]
assert len(big) == 126 and len(unit) == 21
assert np.abs(np.abs(big ** 3 - 20) - 4 * math.sqrt(7)).max() < 1e-11      # lambda^3 = 4(5 +- sqrt(-7))
assert np.abs((big ** 3).real - 20).max() < 1e-11 and np.abs(unit ** 3 + 1).max() < 1e-12
print(f"(A) Y_168: A_1 has 0^18, kappa = 7, the orbit of e^(i pi/3) (contains -1) with multiplicity 7, "
      f"and the orbits of 2 sqrt 2 e^(+-i phi), phi = {PHI_L:.6f} = {PHI_L / math.pi:.6f} pi, "
      f"multiplicity 21 each   {elapsed()}")


# ---------------------------------------------------------------- cluster equilibria (vectorized)
def red(X, K1, K2, a, Phi):
    d0, d1 = X[..., 0], X[..., 1]
    d = [d0, d1, -d0 - d1]
    f = [0.5 * K1 * (np.sin(d[t] + a) - np.sin(d[t - 1] + a)) + K2 * np.sin(d[t] - d[t - 1] + Phi)
         for t in range(3)]
    return np.stack([f[1] - f[0], f[2] - f[1]], -1)


def newton(X, K1, K2, a, Phi, it=40):
    for _ in range(it):
        F = red(X, K1, K2, a, Phi)
        J = np.empty(X.shape[:-1] + (2, 2))
        for k in range(2):
            e = np.zeros(2)
            e[k] = 1e-7
            J[..., :, k] = (red(X + e, K1, K2, a, Phi) - red(X - e, K1, K2, a, Phi)) / 2e-7
        det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
        det = np.where(np.abs(det) < 1e-14, 1e-14, det)
        dx0 = (J[..., 1, 1] * F[..., 0] - J[..., 0, 1] * F[..., 1]) / det
        dx1 = (-J[..., 1, 0] * F[..., 0] + J[..., 0, 0] * F[..., 1]) / det
        X = X - np.clip(np.stack([dx0, dx1], -1), -0.5, 0.5)
    ok = np.abs(red(X, K1, K2, a, Phi)).max(-1) < 1e-11
    return np.mod(X + math.pi, TP) - math.pi, ok


def coeffs(X, K1, K2, a, Phi):
    d = [X[..., 0], X[..., 1], -X[..., 0] - X[..., 1]]
    pt = [K1 * np.cos(d[t] + a) / (2 * KAP) for t in range(3)]
    ct = [K2 * np.cos(d[t] - d[t - 1] + Phi) / KAP for t in range(3)]
    al = [pt[t] + ct[t] for t in range(3)]
    ga = [pt[t - 1] + ct[t] for t in range(3)]
    sg = np.stack([KAP * (al[t] + ga[t]) for t in range(3)], -1)
    return al, ga, sg


def blocks(X, K1, K2, a, Phi):
    """X (B, 2) -> eigenvalues (B, 4, 3) of the four M(lambda) (rotation zero set to -inf), sigma (B, 3)."""
    al, ga, sg = coeffs(X, K1, K2, a, Phi)
    B = X.shape[0]
    M = np.zeros((B, len(DIST), 3, 3), complex)
    for t in range(3):
        M[:, :, t, t] = -sg[:, t][:, None]
        M[:, :, t, (t + 1) % 3] += al[t][:, None] * DIST[None, :]
        M[:, :, t, (t - 1) % 3] += ga[t][:, None] * np.conj(DIST)[None, :]
    ev = np.linalg.eigvals(M)
    kz = np.argmin(np.abs(ev[:, KAPI, :]), -1)
    ev[np.arange(B), KAPI, kz] = -np.inf
    return ev, sg


def leading(ev, sg):
    B = ev.shape[0]
    flat = ev.reshape(B, -1)
    i = np.argmax(flat.real, -1)
    mu = flat[np.arange(B), i]
    s = (-sg).max(-1)
    use_s = s > mu.real
    return np.where(use_s, s, mu), np.where(use_s, -1, i // 3)


SEEDS = np.array([[u, v] for u in np.linspace(-math.pi, math.pi, 7)[:-1]
                  for v in np.linspace(-math.pi, math.pi, 7)[:-1]])
K2GRID = np.linspace(-3, 3, 601)


def dedupe(xs):
    keep = []
    for x in xs:
        if all(np.abs(np.mod(x - y + math.pi, TP) - math.pi).max() > 1e-6 for y in keep):
            keep.append(x)
    return np.array(keep)


def kind(blk, mu):
    if blk == -1:
        return "sigma_t (real)"
    if blk == KAPI:
        return "cluster (real)"
    return "transverse, complex (Hopf)" if abs(mu.imag) > 1e-6 else "transverse, real"


def first_instabilities(alpha, Phi, K1=1.0):
    """Every stable cluster equilibrium on the K_2 grid, continued one step; returns the events
    (K_2 before, x before, leading mu after, block after)."""
    a = TP * alpha / 3
    X = np.broadcast_to(SEEDS, (len(K2GRID),) + SEEDS.shape).copy()
    X, ok = newton(X, K1, np.broadcast_to(K2GRID[:, None], X.shape[:-1]), a, Phi)
    events = []
    for i in range(len(K2GRID) - 1):
        xs = X[i][ok[i]]
        if not len(xs):
            continue
        xs = dedupe(xs)
        ev, sg = blocks(xs, K1, K2GRID[i], a, Phi)
        mu, _ = leading(ev, sg)
        st = mu.real < -1e-9
        if not st.any():
            continue
        x0 = xs[st]
        x1, ok1 = newton(x0.copy(), K1, np.full(len(x0), K2GRID[i + 1]), a, Phi)
        ev1, sg1 = blocks(x1, K1, K2GRID[i + 1], a, Phi)
        mu1, blk1 = leading(ev1, sg1)
        close = np.abs(np.mod(x1 - x0 + math.pi, TP) - math.pi).max(-1) < 0.1
        for j in np.flatnonzero(ok1 & close & (mu1.real > 0)):
            events.append((K2GRID[i], x0[j], mu1[j], int(blk1[j])))
    return events


def crit_mu(x, K2, alpha, Phi, blk):
    ev, _ = blocks(x[None], 1.0, K2, TP * alpha / 3, Phi)
    e = ev[0, blk]
    return e[np.argmax(e.real)]


def refine(alpha, Phi, K2lo, x, blk):
    """Bisection on K_2 along the branch through x (stable at K2lo, unstable at K2lo + 0.01)."""
    a = TP * alpha / 3
    lo, hi = K2lo, K2lo + 0.01
    for _ in range(50):
        mid = (lo + hi) / 2
        xm, ok = newton(x[None].copy(), 1.0, np.array([mid]), a, Phi)
        assert ok[0]
        if crit_mu(xm[0], mid, alpha, Phi, blk).real < 0:
            lo, x = mid, xm[0]
        else:
            hi = mid
    K2 = (lo + hi) / 2
    x = newton(x[None].copy(), 1.0, np.array([K2]), a, Phi)[0][0]
    mu = crit_mu(x, K2, alpha, Phi, blk)
    sl = (crit_mu(newton(x[None].copy(), 1.0, np.array([K2 + 1e-6]), a, Phi)[0][0], K2 + 1e-6, alpha, Phi, blk).real
          - crit_mu(newton(x[None].copy(), 1.0, np.array([K2 - 1e-6]), a, Phi)[0][0], K2 - 1e-6, alpha, Phi, blk).real) / 2e-6
    return K2, x, mu, sl


# M(e^{i pi/3}) is similar to the real matrix M(-1): spectrum closed under conjugation
rng = np.random.default_rng(3)
for _ in range(20):
    X = rng.uniform(-math.pi, math.pi, (1, 2))
    al, ga, sg = coeffs(X, 1.0, rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-3, 3))
    for l in (np.exp(1j * math.pi / 3), -1.0):
        M = np.diag(-sg[0]).astype(complex)
        for t in range(3):
            M[t, (t + 1) % 3] += al[t][0] * l
            M[t, (t - 1) % 3] += ga[t][0] * np.conj(l)
        e = np.sort_complex(np.linalg.eigvals(M))
        if l == -1.0:
            assert np.abs(np.sort_complex(e1) - e).max() < 1e-9
        e1 = e
    assert np.abs(np.sort_complex(e) - np.sort_complex(np.conj(e))).max() < 1e-9

# ================================================================ (S) scan of first instabilities
NS = 12 if QUICK else 150
rng = np.random.default_rng(2026)
allev, hopf_pts = Counter(), []
for trial in range(NS):
    alpha, Phi = rng.uniform(0, 1), rng.uniform(-math.pi, math.pi)
    seen = set()
    for (K2, x, mu, blk) in first_instabilities(alpha, Phi):
        key = (round(K2, 4), round(mu.real, 6), round(mu.imag, 6), blk)
        if key in seen:                                   # Z/3-rotated copies of one equilibrium
            continue
        seen.add(key)
        k = kind(blk, mu)
        allev[k] += 1
        if k.startswith("transverse, complex"):
            hopf_pts.append((alpha, Phi, K2, x, blk))
    if trial % 50 == 49:
        print(f"(S) {trial + 1} slices   {elapsed()}", flush=True)
tot = sum(allev.values())
print(f"(S) {NS} slices (alpha, Phi) x 601 values of K_2: {tot} losses of stability of stable cluster "
      f"equilibria (Z/3 copies counted once): " + ", ".join(f"{k}: {v}" for k, v in allev.most_common()))
hblk = Counter(DIST[b] for (_, _, _, _, b) in hopf_pts)
assert allev["transverse, complex (Hopf)"] > 0 and set(hblk) <= {DIST[LP], DIST[LM]}
print(f"(S) every Hopf event is in a block 2 sqrt 2 e^(+-i phi) (none in the e^(i pi/3) block); "
      f"K_2 at the Hopf events in [{min(h[2] for h in hopf_pts):.2f}, {max(h[2] for h in hopf_pts):.2f}]   {elapsed()}")
RES["scan"] = dict(slices=NS, events=dict(allev), hopf_K2_range=[min(h[2] for h in hopf_pts), max(h[2] for h in hopf_pts)])

# ================================================================ (L) Hopf locus at alpha = 0.55
ALPHA = 0.55
LOC = []
NPHI = 12 if QUICK else 72
for Phi in np.linspace(-math.pi, math.pi, NPHI + 1)[:-1]:
    seen = set()
    for (K2, x, mu, blk) in first_instabilities(ALPHA, Phi):
        if not kind(blk, mu).startswith("transverse, complex"):
            continue
        K2s, xs_, mus, sl = refine(ALPHA, Phi, K2, x, blk)
        key = round(K2s, 6)
        if key in seen:
            continue
        seen.add(key)
        LOC.append((Phi, K2s, abs(mus.imag), sl))
print(f"(L) alpha = {ALPHA}: {len(LOC)} Hopf points on {NPHI} values of Phi; Omega_H in "
      f"[{min(l[2] for l in LOC):.4f}, {max(l[2] for l in LOC):.4f}], every slope d Re mu/d K_2 > 0: "
      f"{all(l[3] > 0 for l in LOC)}   {elapsed()}")
RES["locus"] = [dict(Phi=l[0], K2=l[1], Omega_H=l[2], slope=l[3]) for l in LOC]

# ================================================================ (P) the representative point
PHI0 = -1.40
evs = [e for e in first_instabilities(ALPHA, PHI0) if kind(e[3], e[2]).startswith("transverse, complex")]
assert evs
K2c, xc, muc, blkc = evs[0]
K2s, xs, mus, slope = refine(ALPHA, PHI0, K2c, xc, blkc)
LAMc = DIST[blkc]
a0 = TP * ALPHA / 3

# 25 digits: unknowns (d_0, d_1, K_2, omega); equations red = 0 and det(M(lambda) - i omega) = 0
mp.mp.dps = 40
Ych = [g for g in itertools.product(range(2), range(2), range(2), range(7))]
G = sorted(Ych)
assert all(Y.tau[i] == i % 3 for i in range(Y.n))


def mp_lambda():
    """The eigenvalue lambda of A_1 to 40 digits, from a 3 x 3 character block of A_1 whose
    double-precision spectrum contains it."""
    for k in itertools.product(range(2), repeat=3):
        for j in range(7):
            chi = lambda g: (-1) ** (k[0] * g[0] + k[1] * g[1] + k[2] * g[2]) * mp.expjpi(mp.mpf(2 * j * g[3]) / 7)
            B = mp.matrix(3, 3)
            for s in range(3):
                i0 = s                                   # vertex (0, s)
                for jj in np.flatnonzero(A1[i0]):
                    B[s, (s + 1) % 3] += chi(G[jj // 3])
            ev = mp.eig(B)[0]
            for e in ev:
                if abs(complex(e) - LAMc) < 1e-9:
                    return e
    raise RuntimeError


LMP = mp_lambda()


def mp_M(d0, d1, K2, l):
    d = [d0, d1, -d0 - d1]
    a = 2 * mp.pi * mp.mpf(ALPHA) / 3
    Phi = mp.mpf(PHI0)
    pt = [mp.cos(d[t] + a) / (2 * KAP) for t in range(3)]
    ct = [K2 * mp.cos(d[t] - d[t - 1] + Phi) / KAP for t in range(3)]
    al = [pt[t] + ct[t] for t in range(3)]
    ga = [pt[t - 1] + ct[t] for t in range(3)]
    M = mp.matrix(3, 3)
    for t in range(3):
        M[t, t] = -KAP * (al[t] + ga[t])
        M[t, (t + 1) % 3] += al[t] * l
        M[t, (t - 1) % 3] += ga[t] * mp.conj(l)
    return M


def mp_eqs(d0, d1, K2, w):
    d = [d0, d1, -d0 - d1]
    a = 2 * mp.pi * mp.mpf(ALPHA) / 3
    Phi = mp.mpf(PHI0)
    f = [mp.mpf(1) / 2 * (mp.sin(d[t] + a) - mp.sin(d[t - 1] + a)) + K2 * mp.sin(d[t] - d[t - 1] + Phi) for t in range(3)]
    D = mp.det(mp_M(d0, d1, K2, LMP) - 1j * w * mp.eye(3))
    return [f[1] - f[0], f[2] - f[1], mp.re(D), mp.im(D)]


sol = mp.findroot(mp_eqs, (mp.mpf(xs[0]), mp.mpf(xs[1]), mp.mpf(K2s), mp.mpf(mus.imag)))
d0s, d1s, K2mp, OMH = sol
assert abs(float(K2mp) - K2s) < 1e-8 and abs(float(OMH) - mus.imag) < 1e-8
res_mp = max(abs(v) for v in mp_eqs(d0s, d1s, K2mp, OMH))
assert res_mp < mp.mpf(10) ** -30


def mp_crit(K2):
    g = lambda u, v: [e for e in mp_eqs(u, v, K2, 0)[:2]]
    u, v = mp.findroot(g, (d0s, d1s))
    ev = mp.eig(mp_M(u, v, K2, LMP))[0]
    return max(ev, key=lambda z: mp.re(z) if abs(mp.im(z) - OMH) < 1e-3 else -mp.inf)


hh = mp.mpf(10) ** -15
SLOPE = (mp.re(mp_crit(K2mp + hh)) - mp.re(mp_crit(K2mp - hh))) / (2 * hh)
assert abs(float(SLOPE) - slope) < 1e-4 * abs(slope) and SLOPE > 0
psi_s = np.array([0.0, float(d0s), float(d0s + d1s)])
# rotation frequency and the full Jacobian
class Net:
    def __init__(self, X):
        self.n = X.n
        self.tau = np.array(X.tau)
        self.src = np.array([e[0] for e in X.edges])
        self.dst = np.array([e[1] for e in X.edges])
        self.T = np.array(X.tris)
        self.deg = np.bincount(np.r_[self.src, self.dst], minlength=self.n)
        self.deg2 = np.bincount(self.T.ravel(), minlength=self.n)

    def field(self, th, K1, K2, a, Phi):
        phi = th[self.dst] - th[self.src] + a
        f = np.bincount(self.src, np.sin(phi), self.n) - np.bincount(self.dst, np.sin(phi), self.n)
        A, B, C = (th[self.T[:, i]] for i in range(3))
        g = (np.bincount(self.T[:, 0], np.sin(B + C - 2 * A + Phi), self.n)
             + np.bincount(self.T[:, 1], np.sin(C + A - 2 * B + Phi), self.n)
             + np.bincount(self.T[:, 2], np.sin(A + B - 2 * C + Phi), self.n))
        return K1 * f / self.deg + K2 * g / self.deg2

    def jac(self, th, p, h=1e-6):
        J = np.zeros((self.n, self.n))
        for k in range(self.n):
            e = np.zeros(self.n)
            e[k] = h
            J[:, k] = (self.field(th + e, *p) - self.field(th - e, *p)) / (2 * h)
        return J

    def orders(self, th):
        return np.array([abs(np.exp(1j * th[self.tau == t]).mean()) for t in range(3)])


NET = Net(Y)
PST = (1.0, float(K2mp), a0, PHI0)
th_s = psi_s[NET.tau]
v_s = NET.field(th_s, *PST)
assert np.ptp(v_s) < 1e-12
OMEGA_ROT = float(v_s.mean())
Jf = NET.jac(th_s, PST)
wf = np.linalg.eigvals(Jf)
on_axis = np.abs(wf.real) < 1e-6
n_plus = int((on_axis & (np.abs(wf.imag - float(OMH)) < 1e-6)).sum())
n_minus = int((on_axis & (np.abs(wf.imag + float(OMH)) < 1e-6)).sum())
n_zero = int((on_axis & (np.abs(wf) < 1e-6)).sum())
gap = float(wf.real[~on_axis].max())
assert n_plus == n_minus == 21 and n_zero == 1 and on_axis.sum() == 43 and gap < -1e-3
print(f"(P) Hopf point (K_1, K_2*, alpha, Phi) = (1, {mp.nstr(K2mp, 25)}, {ALPHA}, {PHI0}), in the block "
      f"lambda = 2 sqrt 2 e^({'+' if LAMc.imag > 0 else '-'}i phi):")
print(f"    cluster equilibrium d = ({mp.nstr(d0s, 20)}, {mp.nstr(d1s, 20)}), rotation Omega - omega = "
      f"{OMEGA_ROT:.10f}; Hopf frequency Omega_H = Im mu = {mp.nstr(OMH, 25)}; transversality "
      f"d Re mu / d K_2 = {mp.nstr(SLOPE, 20)}; residual {mp.nstr(res_mp, 3)}")
print(f"    full 168 x 168 Jacobian: {n_plus} eigenvalues at +i Omega_H, {n_minus} at -i Omega_H, {n_zero} at 0 "
      f"(rotation), all others Re <= {gap:.5f}: a Hopf point of multiplicity 21 (+ 21 conjugate)   {elapsed()}")
RES["point"] = dict(K1=1.0, K2=mp.nstr(K2mp, 30), alpha=ALPHA, Phi=PHI0, d0=mp.nstr(d0s, 25), d1=mp.nstr(d1s, 25),
                    Omega_H=mp.nstr(OMH, 30), slope=mp.nstr(SLOPE, 25), Omega_rot=OMEGA_ROT, gap=gap,
                    multiplicity=21, lam=[LAMc.real, LAMc.imag])

# 1D continuation of the leading eigenvalues through K_2*
CONT = []
xcur = np.array([float(d0s), float(d1s)])
blkconj = LM if blkc == LP else LP
for K2 in np.linspace(float(K2mp) - 0.4, float(K2mp) + 0.03, 87):     # the branch folds near K_2 = -0.505
    xcur = newton(xcur[None].copy(), 1.0, np.array([K2]), a0, PHI0)[0][0]
    ev, sg = blocks(xcur[None], 1.0, K2, a0, PHI0)
    e = ev[0, blkc]
    mu = e[np.argmax(e.real)]
    jc = int(np.argmin(np.abs(ev[0, blkconj] - np.conj(mu))))          # the conjugate partner in M(lambda-bar)
    others = np.r_[np.delete(ev[0].ravel(), [blkc * 3 + np.argmax(e.real), blkconj * 3 + jc]), -sg[0]]
    others = others[np.isfinite(others.real)]
    CONT.append((K2, mu.real, mu.imag, others.real.max()))
CONT = np.array(CONT)
assert CONT[CONT[:, 0] <= float(K2mp), 3].max() < 0    # the rest of the spectrum is stable up to K_2*


def next_crossing(x, K2h, alpha, Phi, blk, dmax=0.3, step=2.5e-4):
    """Distance from K_2* to the first crossing of any eigenvalue other than the Hopf pair."""
    a = TP * alpha / 3
    bconj = LM if blk == LP else LP
    for dk in np.arange(step, dmax, step):
        x, ok = newton(x[None].copy(), 1.0, np.array([K2h + dk]), a, Phi)
        x = x[0]
        if not ok[0]:
            return dk, "fold"
        ev, sg = blocks(x[None], 1.0, K2h + dk, a, Phi)
        e = ev[0, blk]
        k = int(np.argmax(e.real))
        jc = int(np.argmin(np.abs(ev[0, bconj] - np.conj(e[k]))))
        flat = ev[0].ravel().copy()
        flat[[blk * 3 + k, bconj * 3 + jc]] = -np.inf
        flat[~np.isfinite(flat.real)] = -np.inf
        i = int(np.argmax(flat.real))
        if max(flat[i].real, (-sg[0]).max()) > 0:
            if (-sg[0]).max() > flat[i].real:
                return dk, "sigma_t"
            name = {KAPI: "cluster", ONE: "e^(i pi/3) block, real" if abs(flat[i].imag) < 1e-9 else "e^(i pi/3) block, complex"}
            return dk, name.get(i // 3, "2 sqrt 2 block")
    return dmax, "none"


D_NEXT, W_NEXT = next_crossing(np.array([float(d0s), float(d1s)]), float(K2mp), ALPHA, PHI0, blkc)
print(f"    next crossing of another eigenvalue after K_2*: delta = {D_NEXT:.4f} ({W_NEXT}); the Hopf pair is "
      f"the only unstable direction for 0 < delta < {D_NEXT:.4f}")
assert 0.002 < D_NEXT < 0.01
RES["point"]["next_crossing"] = [D_NEXT, W_NEXT]
assert np.all(np.diff(np.sign(CONT[:, 1]))[np.abs(np.diff(np.sign(CONT[:, 1]))) > 0] > 0)
RES["continuation"] = CONT.tolist()

# ================================================================ (D) dynamics on Y_168
G56 = sorted(itertools.product(range(2), range(2), range(2), range(7)))
AV = [G56[i // 3] for i in range(Y.n)]


def setup(dK, Kc=None, alpha=ALPHA, Phi=PHI0, x0=None):
    Kc = float(K2mp) if Kc is None else Kc
    a = TP * alpha / 3
    p = (1.0, Kc + dK, a, Phi)
    x0 = np.array([float(d0s), float(d1s)]) if x0 is None else x0
    xe = newton(x0[None].copy(), 1.0, np.array([Kc + dK]), a, Phi)[0][0]
    the = np.array([0.0, xe[0], xe[0] + xe[1]])[NET.tau]
    return p, the, NET.jac(the, p)


def crit_pert(Jx, seed, k=42):
    w, V = np.linalg.eig(Jx)
    ci = [i for i in np.argsort(-w.real) if abs(w[i]) > 1e-8][:k]
    rng = np.random.default_rng(seed)
    c = rng.normal(size=len(ci)) + 1j * rng.normal(size=len(ci))
    v = np.real(V[:, ci] @ c)
    return v / (np.linalg.norm(v) / math.sqrt(Y.n)), w, V, ci


def simulate(p, th, T, dt=0.25, rec=8):
    f = lambda x: NET.field(x, *p)
    ts, X = [], []
    for s in range(int(round(T / dt))):
        k1 = f(th)
        k2 = f(th + dt / 2 * k1)
        k3 = f(th + dt / 2 * k2)
        k4 = f(th + dt * k3)
        th = th + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if s % rec == 0:
            ts.append((s + 1) * dt)
            X.append(th.copy())
    return np.array(ts), np.array(X), th


def deviation(X, the):
    dl = np.angle(np.exp(1j * (X - the)))
    dl = np.angle(np.exp(1j * (dl - np.angle(np.exp(1j * dl).mean(-1))[..., None])))
    return np.sqrt((dl ** 2).mean(-1)), dl


def end_state(p, th, dl):
    fr = NET.field(th, *p)
    return dict(freq_spread=float(np.ptp(fr)), shape_change=float(np.abs(dl[-1] - dl[-25]).max()),
                r=NET.orders(th).tolist(), A=float(np.sqrt((dl[-1] ** 2).mean())), Omega=float(fr.mean()))


DYN = {}
# (D1) transient above threshold: delta = +0.0005 (Re mu = slope * delta), eps = 1e-4, T = 6000
dK1 = 0.0005
p, the, Jx = setup(dK1)
pert, w, V, ci = crit_pert(Jx, 11)
ts, X, thf = simulate(p, the + 1e-4 * pert, 6000)
A, dl = deviation(X, the)
# complex coordinate of the deviation in one critical mode (left eigenvector projection)
Wl = np.linalg.inv(V)
z = dl @ Wl[ci[0]]
RT = np.array([[abs(np.exp(1j * x[NET.tau == t]).mean()) for t in range(3)] for x in X])
es = end_state(p, thf, dl)
mu_lin = complex(np.linalg.eigvals(Jx)[np.argmax(np.linalg.eigvals(Jx).real)])     # exact Re mu at delta


def window(a_lo, a_hi):
    i0, i1 = int(np.argmax(A > a_lo)), int(np.argmax(A > a_hi))
    rate = math.log(A[i1] / A[i0]) / (ts[i1] - ts[i0])
    turn = np.unwrap(np.angle(z[i0:i1 + 1]))
    return rate, abs(turn[-1] - turn[0]) / (ts[i1] - ts[i0])


r_lin, w_lin = window(2e-4, 1e-3)
r_mid, w_mid = window(3e-3, 1e-2)
r_nl, w_nl = window(1e-2, 3e-2)
print(f"(D1) delta = +{dK1}: linearization mu = {mu_lin.real:.5f} {mu_lin.imag:+.5f}i. Growth rate of the rms "
      f"deviation A and rotation rate of the critical-mode coordinate: {r_lin:.5f}, {w_lin:.5f} for A in "
      f"[2e-4, 1e-3]; {r_mid:.5f}, {w_mid:.5f} for A in [3e-3, 1e-2]; {r_nl:.5f}, {w_nl:.5f} for A in "
      f"[1e-2, 3e-2]. The growth accelerates with the amplitude (destabilizing nonlinearity). End state: "
      f"A = {es['A']:.4f}, r = ({', '.join(f'{x:.4f}' for x in es['r'])}), frequency spread "
      f"{es['freq_spread']:.1e}, shape change {es['shape_change']:.1e}   {elapsed()}")
assert abs(r_lin - mu_lin.real) < 0.1 * mu_lin.real and abs(w_lin - abs(mu_lin.imag)) < 0.05 * abs(mu_lin.imag)
assert r_nl > r_mid > 1.2 * r_lin
assert es["freq_spread"] < 1e-9 and es["shape_change"] < 1e-9 and es["A"] > 0.2
DYN["transient"] = dict(delta=dK1, mu=[mu_lin.real, mu_lin.imag], windows=[[2e-4, 1e-3, r_lin, w_lin],
                        [3e-3, 1e-2, r_mid, w_mid], [1e-2, 3e-2, r_nl, w_nl]], end=es)
TR = dict(ts=ts, A=A, z=z, RT=RT)

# (D2) generic and pure-character perturbations at delta = +0.001, +0.002 (exactly the 42 Hopf modes
# are unstable there; at +0.004 a real eigenvalue of another block has crossed as well)
CH = [(k, j) for k in itertools.product(range(2), repeat=3) for j in range(7)]


def chi_vec(c):
    k, j = c
    return np.array([(-1) ** (k[0] * g[0] + k[1] * g[1] + k[2] * g[2]) * np.exp(2j * math.pi * j * g[3] / 7) for g in AV])


for dK in ((0.002,) if QUICK else (0.001, 0.002)):
    p, the, Jx = setup(dK)
    assert (np.linalg.eigvals(Jx).real > 1e-9).sum() == 42
    out = []
    for seed in (1, 2, 3):
        pert, *_ = crit_pert(Jx, seed)
        ts_, X_, th_ = simulate(p, the + 1e-3 * pert, 4000, rec=40)
        out.append(end_state(p, th_, deviation(X_, the)[1]))
    nchar, cends = 0, []
    for c in CH:
        v = chi_vec(c)
        B = np.stack([v * (NET.tau == s) for s in range(3)], 1) / math.sqrt(56)
        Jc = B.conj().T @ Jx @ B
        assert np.abs(Jx @ B - B @ Jc).max() < 1e-6                 # deck-group invariance
        wc, Uc = np.linalg.eig(Jc)
        kk = np.argmax(wc.real)
        if wc[kk].real <= 1e-9:                     # (the trivial character carries the rotation zero)
            continue
        nchar += 1
        if len(cends) < 3:
            vec = np.real(B @ Uc[:, kk])
            vec /= np.linalg.norm(vec) / math.sqrt(Y.n)
            ts_, X_, th_ = simulate(p, the + 1e-3 * vec, 4000, rec=40)
            cends.append(end_state(p, th_, deviation(X_, the)[1]))
    assert nchar == 42
    for e in out + cends:
        assert e["freq_spread"] < 1e-9 and e["shape_change"] < 1e-9 and e["A"] > 0.2
    print(f"(D2) delta = +{dK}: 3 generic perturbations in the 42-dimensional critical space and 3 pure "
          f"character modes (of the {nchar} unstable characters) all end in frequency-locked states that are "
          f"stationary in the rotating frame: A = " + ", ".join(f"{e['A']:.3f}" for e in out + cends)
          + "; r_0 = " + ", ".join(f"{e['r'][0]:.3f}" for e in out + cends) + f"   {elapsed()}")
    DYN[f"above_{dK}"] = dict(generic=out, characters=cends, unstable_characters=nchar)

# (D3) basin radius below threshold
BAS = []
for dK in (-0.002, -0.008, -0.032):
    p, the, Jx = setup(dK)
    pert, *_ = crit_pert(Jx, 5)
    lo, hi = 1e-3, 0.3
    for _ in range(6 if QUICK else 9):
        mid = math.sqrt(lo * hi)
        ts_, X_, th_ = simulate(p, the + mid * pert, 2500, rec=100)
        if deviation(X_[-1:], the)[0][-1] < mid / 5:
            lo = mid
        else:
            hi = mid
    BAS.append((dK, lo, hi))
ex = np.polyfit(np.log([-b[0] for b in BAS]), np.log([math.sqrt(b[1] * b[2]) for b in BAS]), 1)[0]
assert all(b[2] < 0.1 for b in BAS) and BAS[0][2] < BAS[1][1] < BAS[1][2] < BAS[2][1]
print("(D3) below threshold the equilibrium is stable, but perturbations of size eps_c escape to a "
      "locked state: " + ", ".join(f"delta = {b[0]}: eps_c in [{b[1]:.4f}, {b[2]:.4f}]" for b in BAS)
      + f"; eps_c -> 0 as delta -> 0-, fitted exponent {ex:.2f}   {elapsed()}")
DYN["basin"] = dict(points=BAS, exponent=ex)
RES["dynamics"] = DYN

# ================================================================ (C) criticality survey
SUR = []
seen = set()
for (alpha, Phi, K2, x, blk) in hopf_pts:
    key = (round(alpha, 3), round(Phi, 3))
    if key in seen or len(SUR) >= (3 if QUICK else 6):
        continue
    seen.add(key)
    K2h, xh, muh, slh = refine(alpha, Phi, K2, x, blk)
    dn, wn = next_crossing(xh, K2h, alpha, Phi, blk)
    dK = dn / 5                                             # well inside the window where only the Hopf pair is unstable
    p, the, Jx = setup(dK, Kc=K2h, alpha=alpha, Phi=Phi, x0=xh)
    w = np.linalg.eigvals(Jx)
    if (w.real > 1e-9).sum() != 42:
        continue
    T = float(min(4000, max(1500, 6 / (slh * dK))))
    pert, *_ = crit_pert(Jx, 7)
    ts_, X_, th_ = simulate(p, the + 1e-3 * pert, T, rec=40)
    Aend, dl_ = deviation(X_, the)
    e = end_state(p, th_, dl_)
    if e["freq_spread"] > 1e-6:
        cls = "drifting"
    elif e["shape_change"] > 1e-6:
        cls = "periodic, small" if e["A"] < 0.1 else "periodic, large"
    else:
        cls = "locked, stationary" if e["A"] > 0.05 else "back to equilibrium"
    SUR.append(dict(alpha=alpha, Phi=Phi, K2=K2h, Omega_H=abs(muh.imag), slope=slh, outcome=cls, A=e["A"],
                    r=e["r"], next_crossing=dn, next_kind=wn, delta=dK))
print(f"(C) {len(SUR)} distinct Hopf points of (S), each run at delta = (distance to the next crossing)/5 from "
      f"a perturbation of size 1e-3 in the critical space: outcomes " + str(dict(Counter(s['outcome'] for s in SUR)))
      + f"; Omega_H in [{min(s['Omega_H'] for s in SUR):.4f}, {max(s['Omega_H'] for s in SUR):.4f}]   {elapsed()}")
assert len(SUR) >= (2 if QUICK else 5)
for s in SUR:
    print(f"    alpha = {s['alpha']:.4f}, Phi = {s['Phi']:+.4f}, K_2* = {s['K2']:+.6f}, Omega_H = {s['Omega_H']:.5f}, "
          f"slope {s['slope']:.3f}, run at delta = {s['delta']:.4f}, next other crossing at delta = "
          f"{s['next_crossing']:.4f} ({s['next_kind']}): {s['outcome']} (A = {s['A']:.3f}, r = {np.round(s['r'], 4).tolist()})")
dn_all = [s["next_crossing"] for s in SUR]
print(f"    distance from K_2* to the next crossing of another eigenvalue: {min(dn_all):.4f} .. {max(dn_all):.4f} "
      f"(kinds {dict(Counter(s['next_kind'] for s in SUR))}); every run is at delta = (that distance)/5, where "
      f"exactly the 42 Hopf modes are unstable")
RES["survey"] = SUR

# ================================================================ output
with open(os.path.join(OUT, "v3p3_hopf_search.json"), "w") as fh:
    json.dump(RES, fh, indent=1, default=float)

import matplotlib                                           # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                             # noqa: E402

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
L = np.array([(l[0], l[1], l[2]) for l in LOC])
sc = ax[0].scatter(L[:, 0], L[:, 1], c=L[:, 2], s=14, cmap="viridis")
ax[0].plot([PHI0], [float(K2mp)], "r*", ms=12, label="point (P)")
fig.colorbar(sc, ax=ax[0], label=r"$\Omega_H$")
ax[0].set_xlabel(r"$\Phi$"), ax[0].set_ylabel(r"$K_2^*$"), ax[0].legend(fontsize=8)
ax[0].set_title(rf"Hopf locus, $K_1=1$, $\alpha={ALPHA}$", fontsize=10)
ax[1].plot(CONT[:, 0], CONT[:, 1], label=r"Re $\mu$ (critical block)")
ax[1].plot(CONT[:, 0], CONT[:, 2], label=r"Im $\mu$")
ax[1].plot(CONT[:, 0], CONT[:, 3], "--", label=r"max Re, all other eigenvalues (without $\bar\mu$)")
ax[1].axhline(0, color="k", lw=0.5), ax[1].axvline(float(K2mp), color="r", lw=0.6)
ax[1].set_xlabel(r"$K_2$"), ax[1].legend(fontsize=8)
ax[1].set_title(rf"continuation at $\Phi={PHI0}$", fontsize=10)
for a_ in ax:
    a_.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v3p3_hopf_locus.png"), dpi=150)

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
ax[0].semilogy(TR["ts"], TR["A"])
ax[0].set_xlabel("t"), ax[0].set_ylabel("rms deviation A(t)")
ax[0].set_title(rf"$\delta=+{dK1}$ above $K_2^*$", fontsize=10)
for t in range(3):
    ax[1].plot(TR["ts"], 1 - TR["RT"][:, t], label=f"$1-r_{t}$")
ax[1].set_yscale("log"), ax[1].set_xlabel("t"), ax[1].legend(fontsize=8)
ax[1].set_title("class order parameters", fontsize=10)
gg = TR["A"] < 0.05
zz = TR["z"][gg] / np.abs(TR["z"][gg]).max()
ax[2].plot(zz.real, zz.imag, lw=0.8)
ax[2].set_aspect("equal"), ax[2].set_xlabel("Re z"), ax[2].set_ylabel("Im z")
ax[2].set_title("critical-mode coordinate during growth (normalized)", fontsize=10)
for a_ in ax:
    a_.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v3p3_hopf_dynamics.png"), dpi=150)
print(f"\nALL CHECKS PASSED: (A) (S) (L) (P) (D1) (D2) (D3) (C).   {elapsed()}")
