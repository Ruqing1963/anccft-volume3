"""Verification battery for Volume III, Paper 3 (chapters/ch03-phase-oscillators.tex).

  (G) gauge data: the uniform connection a*1_E has flux 3a on every chamber, the type
      gauge chi = (2 pi/3) tau shifts it by 2 pi/3, and the flat twists are counted by
      |H_1(Y;Z)| (Y_21: 64);
  (L) triangle lemma: max of cos x1 + cos x2 + cos x3 on x1 + x2 + x3 = S mod 2 pi;
  (V) frustrated ground states: V >= -|E| cos(2 pi beta/3), attained by the twisted state,
      and never beaten by random descents (Y_21, T_6);
  (J) Jacobian at every twisted state = -c_m Delta_0 (Y_21, Y_24, T_6), relative
      equilibria rotating at omega + K_2 sin Phi;
  (F) frequency sum rule  mean(theta') = omega + K_2 <m_Delta>_F  at random states;
  (R) the three-cluster system is a gradient flow modulo rotation: its reduced Jacobian is
      symmetric at every equilibrium found;
  (B) block decomposition of the Jacobian at type-cluster equilibria (Y_21, Y_168);
  (O) Ott-Antonsen: Lyapunov inequality for K_2 = 0, no chimera limit in the K_2 = 0 OA
      system, and non-closure of the OA manifold for K_2 != 0 (finite populations);
  (N) numerics: stable cluster equilibria never have exactly one transversally stable
      class; the one-class instability ends in a frozen state; at large K_2 on Y_168 a
      solitary state (<= 2 drifting oscillators) and a disordered state (every class
      mostly drifting), neither class-selective;
  (E) Euler characteristic: chi(Y) = n for q = 2, chi(T_6) = 0.

Run:  python code/v3p3_verify.py      (numpy only)
"""
import math
import time
from collections import Counter

import numpy as np

from a2_common import standard_covers, thin_torus, torsion

t0 = time.time()
COV = standard_covers()
Y21, Y24, Y168 = COV["Y_21"], COV["Y_24"], COV["Y_168"]
T6 = thin_torus(6)
TWO_PI = 2 * math.pi


class Net:
    """theta'_j = omega + K1/deg(j) sum_k sin(theta_k - theta_j + A_jk)
                 + K2/deg2(j) sum_{chambers (j,k,l)} sin(theta_k + theta_l - 2 theta_j + Phi),
    with A = a on every type-increasing edge (A_kj = -A_jk)."""

    def __init__(self, X):
        self.X, self.n = X, X.n
        self.tau = np.array(X.tau)
        self.src = np.array([e[0] for e in X.edges])
        self.dst = np.array([e[1] for e in X.edges])
        self.T = np.array(X.tris)
        self.deg = np.bincount(np.r_[self.src, self.dst], minlength=self.n)
        self.deg2 = np.bincount(self.T.ravel(), minlength=self.n)
        self.kappa = int(self.deg[0]) // 2

    def corner_args(self, th, Phi):
        A, B, C = (th[self.T[:, i]] for i in range(3))
        return np.stack([B + C - 2 * A, C + A - 2 * B, A + B - 2 * C]) + Phi

    def field(self, th, K1, K2, a, Phi, om=0.0):
        phi = th[self.dst] - th[self.src] + a
        f = np.zeros(self.n)
        np.add.at(f, self.src, np.sin(phi))
        np.add.at(f, self.dst, -np.sin(phi))
        g = np.zeros(self.n)
        s = np.sin(self.corner_args(th, Phi))
        for i in range(3):
            np.add.at(g, self.T[:, i], s[i])
        return om + K1 * f / self.deg + K2 * g / self.deg2

    def energy(self, th, a):
        return -np.cos(th[self.dst] - th[self.src] + a).sum()

    def jac(self, th, p, h=1e-6):
        J = np.zeros((self.n, self.n))
        for k in range(self.n):
            e = np.zeros(self.n)
            e[k] = h
            J[:, k] = (self.field(th + e, *p) - self.field(th - e, *p)) / (2 * h)
        return J

    def rk4(self, th, dt, steps, p):
        for _ in range(steps):
            k1 = self.field(th, *p)
            k2 = self.field(th + dt / 2 * k1, *p)
            k3 = self.field(th + dt / 2 * k2, *p)
            k4 = self.field(th + dt * k3, *p)
            th = th + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return th

    def orders(self, th):
        return np.array([abs(np.exp(1j * th[self.tau == t]).mean()) for t in range(3)])


NETS = {X.name: Net(X) for X in (Y21, Y24, Y168, T6)}

# ---------------------------------------------------------------- (G) gauge data
for X in (Y21, Y24, T6):
    E = len(X.edges)
    a = 0.37
    flux = X.d2.T @ (a * np.ones(E))                         # delta A on chambers
    assert np.allclose(flux, 3 * a)
    chi = TWO_PI / 3 * np.array(X.tau)
    dchi = X.d1.T @ chi                                       # chi_k - chi_j on j -> k
    assert np.allclose(np.mod(dchi - TWO_PI / 3 + 1e-9, TWO_PI), 1e-9)
H1 = torsion(Y21.d2.tolist())                                 # Tor coker d2 = H_1 (b_1 = 0)
assert H1 == [2] * 6 and math.prod(H1) == 64
print(f"(G) uniform connection: flux 3a on every chamber; type gauge shifts a by 2pi/3; "
      f"H_1(Y_21) = {H1} -> 64 gauge classes per flux")

# ---------------------------------------------------------------- (L) triangle lemma
rng = np.random.default_rng(0)
for S in rng.uniform(-math.pi, math.pi, 200):
    g = np.linspace(-math.pi, math.pi, 721)
    X1, X2 = np.meshgrid(g, g)
    vals = np.cos(X1) + np.cos(X2) + np.cos(S - X1 - X2)
    assert vals.max() <= 3 * math.cos(S / 3) + 1e-12
    assert vals.max() >= 3 * math.cos(S / 3) - 1e-4
print("(L) max(cos x1 + cos x2 + cos x3 | sum = S mod 2pi) = 3 cos(S/3) for |S| <= pi")


# ---------------------------------------------------------------- (V) ground states
def descend(N, th, a, steps=4000, dt=0.2):
    for _ in range(steps):
        th = th + dt * N.field(th, 1.0, 0.0, a, 0.0)
    return th


for name in ("Y_21", "T_6"):
    N = NETS[name]
    E = len(N.src)
    for alpha in (0.0, 0.3, -0.45, 0.5, 1.2):
        a = TWO_PI * alpha / 3
        beta = alpha - math.floor(alpha + 0.5)
        bound = -E * math.cos(TWO_PI * beta / 3)
        m = -math.floor(alpha + 0.5)
        tw = TWO_PI * m / 3 * N.tau
        assert abs(N.energy(tw, a) - bound) < 1e-9
        best = min(N.energy(descend(N, rng.uniform(0, TWO_PI, N.n), a), a)
                   for _ in range(6))
        assert best >= bound - 1e-8
    print(f"(V) {name}: twisted state attains -|E|cos(2 pi beta/3); 30 random descents "
          f"never go below it")

# ---------------------------------------------------------------- (J) twisted states
for name in ("Y_21", "Y_24", "T_6"):
    N = NETS[name]
    kap = N.kappa
    for (K1, K2, alpha, Phi) in [(1.0, 0.7, 0.3, 0.4), (-0.8, 1.3, 0.1, 2.0),
                                 (1.0, 0.0, 0.5, 0.0)]:
        a = TWO_PI * alpha / 3
        p = (K1, K2, a, Phi)
        for m in range(3):
            th = TWO_PI * m / 3 * N.tau
            v = N.field(th, *p)
            assert np.allclose(v, K2 * math.sin(Phi))           # rotation at K2 sin Phi
            c = (K1 * math.cos(TWO_PI * (m + alpha) / 3) + 2 * K2 * math.cos(Phi)) / (2 * kap)
            assert np.abs(N.jac(th, p) + c * N.X.L0).max() < 1e-7
print("(J) Jacobian at each twisted state = -c_m Delta_0; Omega - omega = K2 sin Phi "
      "(Y_21, Y_24, T_6)")
ev21 = np.round(np.linalg.eigvalsh(Y21.L0.astype(float)), 8)
assert Counter(ev21.tolist()) == Counter({0.0: 1, 14.0: 18, 21.0: 2})
# the worked example of Section V3P3:sec:example: K1 = 1, K2 = 0.2, alpha = 0.3, Phi = pi/4
p_ex = (1.0, 0.2, TWO_PI * 0.3 / 3, math.pi / 4)
C_EX = {0: 0.07799, 1: -0.04505, 2: 0.027669}
for m in range(3):
    c = (math.cos(TWO_PI * (m + 0.3) / 3) + 0.4 * math.cos(math.pi / 4)) / 14
    assert abs(c - C_EX[m]) < 1e-6
    ev = np.sort(np.linalg.eigvals(NETS["Y_21"].jac(TWO_PI * m / 3 * NETS["Y_21"].tau, p_ex)).real)
    want = np.sort([0.0] + [-14 * c] * 18 + [-21 * c] * 2)
    assert np.abs(ev - want).max() < 1e-6
print(f"    Y_21 example: c_m = {C_EX}; Theta_0, Theta_2 stable, Theta_1 unstable; "
      f"Omega - omega = {0.2 * math.sin(math.pi / 4):.6f}")

# ---------------------------------------------------------------- (F) frequency sum rule
for name in ("Y_21", "Y_168", "T_6"):
    N = NETS[name]
    F = len(N.T)
    for _ in range(20):
        th = rng.uniform(0, TWO_PI, N.n)
        K1, K2, a, Phi, om = rng.normal(size=5)
        mD = np.sin(N.corner_args(th, Phi)).mean(axis=0)       # m_Delta
        lhs = N.field(th, K1, K2, a, Phi, om).mean()
        assert abs(lhs - (om + K2 * mD.mean())) < 1e-12
    assert 3 * F == N.n * int(N.deg2[0])
print("(F) (1/n) sum theta' = omega + K2 <m_Delta>_F at random states (Y_21, Y_168, T_6)")


# ---------------------------------------------------------------- cluster system
def cluster_field(psi, K1, K2, a, Phi):
    out = np.zeros(3)
    for t in range(3):
        dp, dm = psi[(t + 1) % 3] - psi[t], psi[(t - 1) % 3] - psi[t]
        out[t] = 0.5 * K1 * (math.sin(dp + a) + math.sin(dm - a)) + K2 * math.sin(dp + dm + Phi)
    return out


def cluster_jac(psi, p, h=1e-7):
    J = np.zeros((3, 3))
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        J[:, k] = (cluster_field(psi + e, *p) - cluster_field(psi - e, *p)) / (2 * h)
    return J


def sigmas(psi, K1, K2, a, Phi):
    s = np.zeros(3)
    for t in range(3):
        dp, dm = psi[(t + 1) % 3] - psi[t], psi[(t - 1) % 3] - psi[t]
        s[t] = 0.5 * K1 * (math.cos(dp + a) + math.cos(dm - a)) + 2 * K2 * math.cos(dp + dm + Phi)
    return s


def newton(x, p, it=60):
    """Zero of the reduced field (d_0, d_1) -> (d_0', d_1')."""
    def red(x):
        v = cluster_field(np.array([0.0, x[0], x[0] + x[1]]), *p)
        return np.array([v[1] - v[0], v[2] - v[1]])
    for _ in range(it):
        f = red(x)
        if np.abs(f).max() < 1e-13:
            return x, True
        J = np.zeros((2, 2))
        for k in range(2):
            e = np.zeros(2)
            e[k] = 1e-7
            J[:, k] = (red(x + e) - red(x - e)) / 2e-7
        try:
            x = x - np.linalg.solve(J, f)
        except np.linalg.LinAlgError:
            return x, False
    return x, np.abs(red(x)).max() < 1e-11


def cluster_equilibria(p, starts, rng):
    sols = []
    for _ in range(starts):
        x, ok = newton(rng.uniform(-math.pi, math.pi, 2), p)
        if not ok:
            continue
        x = np.mod(x + math.pi, TWO_PI) - math.pi
        if any(np.abs(np.mod(x - y + math.pi, TWO_PI) - math.pi).max() < 1e-6 for y in sols):
            continue
        sols.append(x)
    return [np.array([0.0, x[0], x[0] + x[1]]) for x in sols]


def is_twisted(psi):
    d = np.array([psi[1] - psi[0], psi[2] - psi[1], psi[0] - psi[2]])
    return np.allclose(np.cos(d - d[0]), 1, atol=1e-6)


# ---------------------------------------------------------------- (R) gradient structure
P = np.eye(3) - np.ones((3, 3)) / 3
nR = 0
for _ in range(150):
    p = (rng.choice([1.0, -1.0]), rng.uniform(-2, 2), rng.uniform(-math.pi, math.pi),
         rng.uniform(-math.pi, math.pi))
    K1, K2, a, Phi = p
    for psi in cluster_equilibria(p, 6, rng):
        R = P @ cluster_jac(psi, p) @ P
        assert np.abs(R - R.T).max() < 1e-6
        nR += 1
    # the field minus the gradient of U is parallel to (1,1,1) at random points
    psi = rng.uniform(0, TWO_PI, 3)
    U = lambda s: (-0.5 * K1 * sum(math.cos(s[(t + 1) % 3] - s[t] + a) for t in range(3))
                   - K2 / 3 * sum(math.cos(s[(t + 1) % 3] + s[(t - 1) % 3] - 2 * s[t] + Phi)
                                  for t in range(3)))
    grad = np.array([(U(psi + h) - U(psi - h)) / 2e-6 for h in 1e-6 * np.eye(3)])
    res = cluster_field(psi, *p) + grad
    assert np.ptp(res) < 1e-7
print(f"(R) reduced three-cluster Jacobian symmetric at {nR} equilibria; field = -grad U + "
      f"(mean) (1,1,1)")


# ---------------------------------------------------------------- (B) block decomposition
def block_spectrum(N, psi, p):
    K1, K2, a, Phi = p
    kap, q = N.kappa, N.X.q
    d = [psi[1] - psi[0], psi[2] - psi[1], psi[0] - psi[2]]
    pt = [K1 * math.cos(d[t] + a) / (2 * kap) for t in range(3)]
    ct = [K2 * math.cos(d[t] - d[(t - 1) % 3] + Phi) / kap for t in range(3)]
    al = [pt[t] + ct[t] for t in range(3)]
    ga = [pt[(t - 1) % 3] + ct[t] for t in range(3)]
    sg = [kap * (al[t] + ga[t]) for t in range(3)]
    A1 = N.X.A1.astype(float)
    lam = np.linalg.eigvals(A1)
    spec = []
    zero = np.abs(lam) < 1e-8
    third = N.n // 3
    for t in range(3):                  # ker A_1 on class t = kernel of the block t-1 <- t
        rows, cols = N.tau == (t - 1) % 3, N.tau == t
        spec += [-sg[t]] * (third - np.linalg.matrix_rank(A1[np.ix_(rows, cols)]))
    for l in lam[~zero]:                                       # each Z/3 orbit counted once
        if not (-math.pi / 3 + 1e-6 <= np.angle(l) < math.pi / 3 + 1e-6):
            continue
        M = np.diag([-s for s in sg]).astype(complex)
        for t in range(3):
            M[t, (t + 1) % 3] += al[t] * l
            M[t, (t - 1) % 3] += ga[t] * np.conj(l)
        spec += list(np.linalg.eigvals(M))
    return np.array(spec), sg


def same_multiset(u, v, tol):
    u, v = list(u), list(v)
    if len(u) != len(v):
        return False
    for x in u:
        k = int(np.argmin([abs(x - y) for y in v]))
        if abs(x - v[k]) > tol:
            return False
        v.pop(k)
    return True


nB, cplx_trans = 0, 0
for name in ("Y_21", "Y_168"):
    N = NETS[name]
    for _ in range(40 if name == "Y_21" else 6):
        p = (1.0, rng.uniform(-2, 2), rng.uniform(-math.pi, math.pi),
             rng.uniform(-math.pi, math.pi))
        eqs = [s for s in cluster_equilibria(p, 8, rng) if not is_twisted(s)]
        if not eqs:
            continue
        psi = eqs[0]
        th = psi[N.tau]
        num = np.linalg.eigvals(N.jac(th, p))
        blk, sg = block_spectrum(N, psi, p)
        assert same_multiset(num, blk, 1e-5), name
        nB += 1
        if name == "Y_21":
            assert np.abs(np.imag(blk)).max() < 1e-9           # no transverse Hopf on Y_21
        else:
            cplx_trans += int(np.abs(np.imag(blk)).max() > 1e-6)
print(f"(B) Jacobian spectrum = U_lambda spec M(lambda) u {{-sigma_t}} at {nB} non-twisted "
      f"cluster equilibria; real on Y_21; complex transverse pairs on Y_168 in "
      f"{cplx_trans} cases")

# ---------------------------------------------------------------- (O) Ott-Antonsen
def oa_field(z, K1, a, om0=0.0, Dl=0.0):
    H = 0.5 * K1 * (np.exp(1j * a) * np.roll(z, -1) + np.exp(-1j * a) * np.roll(z, 1))
    return (1j * om0 - Dl) * z + 0.5 * (H - np.conj(H) * z ** 2), H


for _ in range(2000):                                         # Lyapunov inequality
    z = np.sqrt(rng.uniform(0, 1, 3)) * np.exp(1j * rng.uniform(0, TWO_PI, 3))
    K1, a, om0 = rng.normal(), rng.uniform(-math.pi, math.pi), rng.normal()
    zd, H = oa_field(z, K1, a, om0)
    Vdot = -np.real(np.sum(np.conj(zd) * H))
    assert Vdot <= -0.5 * np.sum(np.abs(H) ** 2 * (1 - np.abs(z) ** 2)) + 1e-12
chim = 0
for _ in range(60):                                           # no chimera limit, Delta = 0
    z = 0.9 * np.sqrt(rng.uniform(0, 1, 3)) * np.exp(1j * rng.uniform(0, TWO_PI, 3))
    K1, a = rng.choice([1.0, -1.0]), rng.uniform(-math.pi, math.pi)
    for _ in range(10000):
        k1 = oa_field(z, K1, a)[0]
        k2 = oa_field(z + 0.025 * k1, K1, a)[0]
        k3 = oa_field(z + 0.025 * k2, K1, a)[0]
        k4 = oa_field(z + 0.05 * k3, K1, a)[0]
        z = z + 0.05 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    r = np.sort(np.abs(z))
    chim += int(r[2] > 0.999 and r[1] < 0.99)
assert chim == 0


def population_defect(K2, Npop=3000, steps=1500, dt=0.01):
    """Three populations, Lorentzian frequencies (width 0.1), Poisson initial data;
    returns max_t max_class |Z_2 - Z_1^2|."""
    u = (np.arange(Npop) + 0.5) / Npop
    om = 0.1 * np.tan(math.pi * (u - 0.5))
    phi = TWO_PI * np.mod(np.arange(Npop) * 0.6180339887498949, 1.0)
    z0 = np.array([0.6, 0.5j, -0.4])
    th = np.stack([np.angle((np.exp(1j * phi) + z) / (1 + np.conj(z) * np.exp(1j * phi)))
                   for z in z0])
    K1, a, Phi = 1.5, 0.8, 0.9

    def f(th):
        Z = np.exp(1j * th).mean(axis=1)
        H = 0.5 * K1 * (np.exp(1j * a) * np.roll(Z, -1) + np.exp(-1j * a) * np.roll(Z, 1))
        G = K2 * np.exp(1j * Phi) * np.roll(Z, -1) * np.roll(Z, 1)
        return om + np.imag(H[:, None] * np.exp(-1j * th)) + np.imag(G[:, None] * np.exp(-2j * th))

    worst = 0.0
    for s in range(steps):
        k1 = f(th)
        k2 = f(th + dt / 2 * k1)
        k3 = f(th + dt / 2 * k2)
        k4 = f(th + dt * k3)
        th = th + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        if s % 50 == 0:
            Z1, Z2 = np.exp(1j * th).mean(axis=1), np.exp(2j * th).mean(axis=1)
            worst = max(worst, float(np.abs(Z2 - Z1 ** 2).max()))
    return worst


dO0, dO1 = population_defect(0.0), population_defect(1.0)
assert dO0 < 0.02 and dO1 > 5 * dO0
print(f"(O) dV/dt <= -1/2 sum |H_t|^2 (1 - r_t^2) (2000 points); 60 OA runs (Delta = 0) "
      f"without chimera limit; |Z_2 - Z_1^2| stays {dO0:.4f} for K2 = 0, reaches {dO1:.4f} "
      f"for K2 = 1")

# ---------------------------------------------------------------- (N) numerics
C = Counter()
for _ in range(1500):
    p = (rng.choice([1.0, -1.0]), rng.uniform(-2, 2), rng.uniform(-math.pi, math.pi),
         rng.uniform(-math.pi, math.pi))
    for psi in cluster_equilibria(p, 10, rng):
        R = P @ cluster_jac(psi, p) @ P
        ev = np.sort(np.linalg.eigvalsh(R))[:2]               # drop the rotation zero
        if ev.max() >= -1e-7:
            continue
        C[(is_twisted(psi), int((sigmas(psi, *p) > 1e-7).sum()))] += 1
assert all(k[1] != 1 for k in C)
print(f"(N) stable cluster equilibria by (twisted, #transversally stable classes): "
      f"{dict(sorted(C.items()))}")

p_seed = (1.0, -1.9932218532425816, -1.013885695717902, -0.0678623140433463)
psi_seed = np.array([0.0, 1.66481304, 2.08093921])
sg = sigmas(psi_seed, *p_seed)
assert sg[0] > 0 and sg[2] > 0 and sg[1] < 0
for name in ("Y_21", "Y_168"):
    N = NETS[name]
    th = psi_seed[N.tau] + 1e-3 * np.random.default_rng(0).standard_normal(N.n)
    th = N.rk4(th, 0.02, 20000, p_seed)
    v = N.field(th, *p_seed)
    r = N.orders(th)
    assert np.ptp(v) < 1e-9 and r[1] < 0.95 and min(r[0], r[2]) > 0.98
    print(f"    {name}: one-class instability ends frozen, r = {np.round(r, 4)}, "
          f"common frequency {v.mean():.5f}")

N = NETS["Y_168"]
DRIFT = {}
for label, p_drift, seed in [("solitary", (1.0, 1.769, 2.371, -0.565), 1),
                             ("disordered", (-1.0, 1.861, 2.754, 0.976), 0)]:
    th = np.random.default_rng(seed).uniform(0, TWO_PI, N.n)
    th = N.rk4(th, 0.05, 10000, p_drift)                       # transient t = 500
    th0 = th.copy()
    th = N.rk4(th, 0.05, 16000, p_drift)                       # window t = 800
    freq = (th - th0) / 800.0
    med = np.median(freq)
    drift = [int((np.abs(freq[N.tau == t] - med) > 2e-3).sum()) for t in range(3)]
    DRIFT[label] = drift
    print(f"    Y_168, {label}, (K1,K2,a,Phi) = {p_drift}, seed {seed}: drifting oscillators "
          f"per class {drift} of 56, r = {np.round(N.orders(th), 3)}")
assert sum(DRIFT["solitary"]) <= 2
assert min(DRIFT["disordered"]) > 40                           # every class mostly drifting

# ---------------------------------------------------------------- (E) Euler characteristic
for X in (Y21, Y24, Y168, T6):
    chi = X.n - len(X.edges) + len(X.tris)
    assert chi == (X.n if X.q == 2 else 0)
print("(E) chi(Y) = n on the q = 2 covers, chi(T_6) = 0")

print(f"\nALL CHECKS PASSED: (G) (L) (V) (J) (F) (R) (B) (O) (N) (E).   "
      f"[{time.time() - t0:.0f} s]")
