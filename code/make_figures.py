"""Vector figures for Volume III (written to ../figures/*.pdf).

  fig_heawood_fano.pdf            Chapter 1: PG(2,2) with the cyclic labelling of the triangle
                                  presentation, and the vertex link (Heawood graph) with Z/3 colours.
  fig_avalanche_distribution.pdf  Chapter 2: exact mean avalanche size <s> = 1^t G 1/(n-1) on thick
                                  covers and canonical-tower levels (Theta(n)) against thin tori
                                  (Theta(n log n)).
  fig_hopf_orbit.pdf              Chapter 3: the critical eigenvalue of the block M(lambda) crossing
                                  the imaginary axis on Y_168, and the hard-loss-of-stability orbit.
  fig_quantum_chaos_crossover.pdf Chapter 4: <r>(eps) from the first-order plateau to GUE (data of
                                  v3p4_gue_transition.py).
  fig_v2_growth.pdf               Chapter 6: v_2/n_m and (e_m+1)/n_m along the canonical tower, with
                                  three extrapolations (data of v3p6_level5_v2.py).

Run from code/:  python make_figures.py     (numpy, scipy, matplotlib; about two minutes)
"""
import itertools
import json
import math
import os

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt                                      # noqa: E402

from a2_common import ORBITS, GIDX, TowerRing, standard_covers       # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
OUT = os.path.join(HERE, "..", "data")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
                     "axes.titlesize": 9, "legend.fontsize": 7.5, "pdf.fonttype": 42})
C0, C1, C2 = "#1f4e79", "#c0504d", "#4f8f3a"                          # type colours t, t+1, t+2


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ================================================================ Chapter 1: Fano plane and Heawood link
lines = [tuple(sorted(l)) for l in ORBITS]                           # {c, c+1, c+3}
s3 = math.sqrt(3)
pos = {"A": (0, 0), "B": (2, 0), "C": (1, s3), "ab": (1, 0), "bc": (1.5, s3 / 2), "ca": (0.5, s3 / 2),
       "g": (1, s3 / 3)}
draw_lines = [("A", "ab", "B"), ("B", "bc", "C"), ("C", "ca", "A"), ("A", "bc", "g"), ("B", "ca", "g"),
              ("C", "ab", "g"), ("ab", "bc", "ca")]
names = list(pos)
for perm in itertools.permutations(range(7)):
    lab = dict(zip(names, perm))
    if {tuple(sorted(lab[x] for x in l)) for l in draw_lines} == set(lines):
        break
fig, ax = plt.subplots(1, 2, figsize=(6.3, 3.0))
a = ax[0]
for l in draw_lines:
    if l == ("ab", "bc", "ca"):
        a.add_patch(plt.Circle(pos["g"], s3 / 3 - 0.0, fill=False, lw=1.1, color="0.35"))
    else:
        xs = [pos[x][0] for x in l]
        ys = [pos[x][1] for x in l]
        p0, p1 = min(zip(xs, ys)), max(zip(xs, ys))
        ends = sorted(zip(xs, ys))
        a.plot([ends[0][0], ends[-1][0]], [ends[0][1], ends[-1][1]], color="0.35", lw=1.1)
for x, (px, py) in pos.items():
    a.plot(px, py, "o", ms=11, color="white", mec=C1, mew=1.6, zorder=3)
    a.text(px, py, str(lab[x]), ha="center", va="center", fontsize=8.5, zorder=4)
a.set_aspect("equal")
a.axis("off")
a.set_title(r"(a) $\mathrm{PG}(2,2)$: lines $\{c,c+1,c+3\}$, $c\in\mathbb{Z}/7$")
b = ax[1]
R = 1.0
ang = lambda k: math.pi / 2 - 2 * math.pi * k / 14
P = {i: (R * math.cos(ang(2 * i)), R * math.sin(ang(2 * i))) for i in range(7)}
Lc = {}
for i in range(7):
    Lc[i] = (R * math.cos(ang(2 * i + 1)), R * math.sin(ang(2 * i + 1)))
line_of = {i: tuple(sorted({i, (i + 1) % 7, (i + 3) % 7})) for i in range(7)}
assert set(line_of.values()) == set(lines)
for i in range(7):                                                   # faint spokes from v
    for pt in (P[i], Lc[i]):
        b.plot([0, pt[0] * 0.93], [0, pt[1] * 0.93], color="0.85", lw=0.5, zorder=0)
for i in range(7):
    for pnt in line_of[i]:
        b.plot([Lc[i][0], P[pnt][0]], [Lc[i][1], P[pnt][1]], color="0.3", lw=0.9, zorder=1)
for i in range(7):
    b.plot(*P[i], "o", ms=10, color="white", mec=C1, mew=1.6, zorder=3)
    b.text(*P[i], str(i), ha="center", va="center", fontsize=7.5, zorder=4)
    b.plot(*Lc[i], "s", ms=9.5, color="white", mec=C2, mew=1.6, zorder=3)
    b.text(*Lc[i], rf"$\ell_{i}$", ha="center", va="center", fontsize=6.5, zorder=4)
b.plot(0, 0, "o", ms=11, color=C0, zorder=3)
b.text(0, 0, r"$v$", color="white", ha="center", va="center", fontsize=8, zorder=4)
b.set_aspect("equal")
b.axis("off")
b.set_title(r"(b) $\mathrm{Lk}(v)$ = Heawood graph, $\tau(v)=t$")
from matplotlib.lines import Line2D                                   # noqa: E402
b.legend(handles=[Line2D([], [], marker="o", ls="", color=C0, label=r"type $t$ (centre $v$)"),
                  Line2D([], [], marker="o", ls="", mfc="white", mec=C1, label=r"type $t+1$ (points)"),
                  Line2D([], [], marker="s", ls="", mfc="white", mec=C2, label=r"type $t+2$ (lines)")],
         loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, handletextpad=0.2,
         columnspacing=0.8)
save(fig, "fig_heawood_fano.pdf")


# ================================================================ Chapter 2: mean avalanche size
def mean_avalanche(Lap):
    n = Lap.shape[0]
    Lr = Lap[1:, 1:].tocsc()
    g = spsolve(Lr, np.ones(n - 1))
    return float(g.sum()) / (n - 1)


def laplacian_from_A1(A1):
    A1 = csr_matrix(A1)
    S = A1 + A1.T
    deg = np.asarray(S.sum(1)).ravel()
    return (diags(deg) - S).tocsr()


thick = []
for name, X in standard_covers().items():
    thick.append((X.n, mean_avalanche(laplacian_from_A1(X.A1)), name))
for m in (3, 4, 5):                                                    # canonical tower levels
    R_ = TowerRing(m)
    th = R_.voltages()
    one = R_.normalize(R_.one())
    seen, fr = {one}, [one]
    while fr:
        new = []
        for g_ in fr:
            for h in th:
                x = R_.normalize(R_.mul(g_, h))
                if x not in seen:
                    seen.add(x)
                    new.append(x)
        fr = new
    G = sorted(seen)
    gid = {g_: i for i, g_ in enumerate(G)}
    n = 3 * len(G)
    rows, cols = [], []
    for g_ in G:
        for s in range(3):
            for xx in range(7):
                w = R_.normalize(R_.mul(g_, th[GIDX[(s, xx)]]))
                rows.append(3 * gid[g_] + s)
                cols.append(3 * gid[w] + (s + 1) % 3)
    A1 = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    thick.append((n, mean_avalanche(laplacian_from_A1(A1)), f"tower $m={m}$"))
thick.sort()
thin = []
for m in (6, 9, 12, 18, 24, 36, 48, 66, 90):
    n = m * m
    rows, cols = [], []
    for a_ in range(m):
        for b_ in range(m):
            for (da, db) in ((1, 0), (0, 1), (-1, -1)):
                rows.append(a_ * m + b_)
                cols.append(((a_ + da) % m) * m + (b_ + db) % m)
    A1 = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    thin.append((n, mean_avalanche(laplacian_from_A1(A1))))
tn, ts_ = np.array([t[0] for t in thick]), np.array([t[1] for t in thick])
hn, hs = np.array([t[0] for t in thin]), np.array([t[1] for t in thin])
assert abs(dict((t[2], t[1]) for t in thick)["Y_21"] - 29 / 20) < 1e-9
fig, ax = plt.subplots(1, 2, figsize=(6.3, 2.6), gridspec_kw={"wspace": 0.38})
ax[0].loglog(tn, ts_, "o", color=C0, ms=4, label=r"thick, $q=2$ (covers of $X_3$, tower)")
ax[0].loglog(hn, hs, "s", color=C1, ms=3.5, label=r"thin tori $T_m$, $q=1$")
nn = np.logspace(1, 4.3, 50)
ax[0].loglog(nn, nn / 14 * (ts_[-1] / (tn[-1] / 14)), "--", color=C0, lw=0.8, label=r"$\propto n$")
cfit = np.polyfit(np.log(hn), hs / hn, 1)                          # <s>/n = a log n + c
assert abs(cfit[0] - math.sqrt(3) / (12 * math.pi)) < 0.003          # triangular lattice: sqrt3/(12 pi)
ax[0].loglog(nn, nn * np.polyval(cfit, np.log(nn)), ":", color=C1, lw=1.0,
             label=rf"$n\,({cfit[0]:.4f}\log n{cfit[1]:+.3f})$")
ax[0].set_xlabel(r"$n$"), ax[0].set_ylabel(r"$\langle s\rangle$"), ax[0].legend(frameon=False)
ax[1].semilogx(tn, ts_ / tn, "o-", color=C0, ms=4, lw=0.8, label="thick")
ax[1].semilogx(hn, hs / hn, "s-", color=C1, ms=3.5, lw=0.8, label="thin")
ax[1].axhline(1 / 14, color=C0, ls=":", lw=0.7)
ax[1].text(1.2e3, 1 / 14 - 0.012, r"$1/\lambda_{\max}=1/14$", color=C0, fontsize=7, va="top")
ax[1].semilogx(nn, np.polyval(cfit, np.log(nn)), ":", color=C1, lw=0.8)
ax[1].set_xlabel(r"$n$"), ax[1].set_ylabel(r"$\langle s\rangle/n$"), ax[1].legend(frameon=False)
for a_ in ax:
    a_.grid(alpha=0.25, lw=0.5)
save(fig, "fig_avalanche_distribution.pdf")
print("   thick:", [(int(a_), round(b_, 3)) for a_, b_ in zip(tn, ts_)])
print("   thin :", [(int(a_), round(b_, 3)) for a_, b_ in zip(hn, hs)], "slope", cfit[0])

# ================================================================ Chapter 3: Hopf crossing and orbit
H = json.load(open(os.path.join(OUT, "v3p3_hopf_search.json")))
Pt = H["point"]
K2s, alpha, Phi = float(Pt["K2"]), float(Pt["alpha"]), float(Pt["Phi"])
lam = complex(*Pt["lam"])
a0 = 2 * math.pi * alpha / 3
KAP = 7


def red(x, K2):
    d = [x[0], x[1], -x[0] - x[1]]
    f = [0.5 * (math.sin(d[t] + a0) - math.sin(d[t - 1] + a0)) + K2 * math.sin(d[t] - d[t - 1] + Phi)
         for t in range(3)]
    return np.array([f[1] - f[0], f[2] - f[1]])


def newton(x, K2):
    for _ in range(40):
        F = red(x, K2)
        J = np.zeros((2, 2))
        for k in range(2):
            e = np.zeros(2)
            e[k] = 1e-7
            J[:, k] = (red(x + e, K2) - red(x - e, K2)) / 2e-7
        x = x - np.linalg.solve(J, F)
    return x


def block(x, K2, l):
    d = [x[0], x[1], -x[0] - x[1]]
    pt = [math.cos(d[t] + a0) / (2 * KAP) for t in range(3)]
    ct = [K2 * math.cos(d[t] - d[t - 1] + Phi) / KAP for t in range(3)]
    al = [pt[t] + ct[t] for t in range(3)]
    ga = [pt[t - 1] + ct[t] for t in range(3)]
    M = np.diag([-KAP * (al[t] + ga[t]) for t in range(3)]).astype(complex)
    for t in range(3):
        M[t, (t + 1) % 3] += al[t] * l
        M[t, (t - 1) % 3] += ga[t] * np.conj(l)
    return np.linalg.eigvals(M)


x = np.array([float(Pt["d0"]), float(Pt["d1"])])
track, trackc, others = [], [], []
xc = x.copy()
for K2 in np.linspace(K2s - 0.25, K2s + 0.004, 160):
    xc = newton(xc, K2)
    e = block(xc, K2, lam)
    mu = e[np.argmax(e.real)]
    track.append((K2, mu))
    ec = block(xc, K2, np.conj(lam))
    trackc.append(ec[np.argmin(np.abs(ec - np.conj(mu)))])
track = np.array(track)
mus, kk = track[:, 1], track[:, 0].real


class Net:
    def __init__(self, X):
        self.n = X.n
        self.tau = np.array(X.tau)
        self.src = np.array([e_[0] for e_ in X.edges])
        self.dst = np.array([e_[1] for e_ in X.edges])
        self.T = np.array(X.tris)
        self.deg = np.bincount(np.r_[self.src, self.dst], minlength=self.n)
        self.deg2 = np.bincount(self.T.ravel(), minlength=self.n)

    def field(self, th, K2):
        phi = th[self.dst] - th[self.src] + a0
        f = np.bincount(self.src, np.sin(phi), self.n) - np.bincount(self.dst, np.sin(phi), self.n)
        A, B, C = (th[self.T[:, i]] for i in range(3))
        g = (np.bincount(self.T[:, 0], np.sin(B + C - 2 * A + Phi), self.n)
             + np.bincount(self.T[:, 1], np.sin(C + A - 2 * B + Phi), self.n)
             + np.bincount(self.T[:, 2], np.sin(A + B - 2 * C + Phi), self.n))
        return f / self.deg + K2 * g / self.deg2


NET = Net(standard_covers()["Y_168"])
dK = 0.0005
K2 = K2s + dK
xe = newton(x.copy(), K2)
the = np.array([0.0, xe[0], xe[0] + xe[1]])[NET.tau]
Jx = np.zeros((NET.n, NET.n))
for k in range(NET.n):
    e = np.zeros(NET.n)
    e[k] = 1e-6
    Jx[:, k] = (NET.field(the + e, K2) - NET.field(the - e, K2)) / 2e-6
w, V = np.linalg.eig(Jx)
ci = [i for i in np.argsort(-w.real) if abs(w[i]) > 1e-8][:42]
rng = np.random.default_rng(11)
cc = rng.normal(size=42) + 1j * rng.normal(size=42)
pert = np.real(V[:, ci] @ cc)
pert /= np.linalg.norm(pert) / math.sqrt(NET.n)
th = the + 1e-4 * pert
Wl = np.linalg.inv(V)
dt, Z, A_, T_ = 0.25, [], [], []
for s in range(24000):
    k1 = NET.field(th, K2)
    k2 = NET.field(th + dt / 2 * k1, K2)
    k3 = NET.field(th + dt / 2 * k2, K2)
    k4 = NET.field(th + dt * k3, K2)
    th = th + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    if s % 4 == 0:
        dl = np.angle(np.exp(1j * (th - the)))
        dl = np.angle(np.exp(1j * (dl - np.angle(np.exp(1j * dl).mean()))))
        Z.append(dl @ Wl[ci[0]])
        A_.append(math.sqrt((dl ** 2).mean()))
        T_.append((s + 1) * dt)
Z, A_, T_ = np.array(Z), np.array(A_), np.array(T_)
fig = plt.figure(figsize=(6.3, 2.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.04, 1.0], wspace=0.45)
a = fig.add_subplot(gs[0, 0])
cax = fig.add_subplot(gs[0, 1])
b = fig.add_subplot(gs[0, 2], projection="polar")
sc = a.scatter(mus.real, mus.imag, c=kk, s=6, cmap="viridis", zorder=2)
a.scatter(np.real(trackc), np.imag(trackc), c=kk, s=7, cmap="viridis", marker="x", lw=0.6, zorder=2)
a.axvline(0, color="k", lw=0.6)
a.plot([0, 0], [float(Pt["Omega_H"]), -float(Pt["Omega_H"])], "r*", ms=7, zorder=3)
cb = fig.colorbar(sc, cax=cax)
cb.set_label(r"$K_2$", labelpad=1)
cb.ax.tick_params(labelsize=7)
a.set_xlabel(r"$\mathrm{Re}\,\mu$"), a.set_ylabel(r"$\mathrm{Im}\,\mu$")
a.set_title(r"(a) $\mu$ ($\bullet$) and $\bar\mu$ ($\times$) through $K_2^*$")
a.set_xlim(-0.15, 0.03)
a.grid(alpha=0.25, lw=0.5)
lin = A_ < 0.03
zz = Z[lin]
rad = 1 + np.log10(np.abs(zz) / np.abs(zz[0]))
b.plot(np.angle(zz), rad, color=C0, lw=0.8)
b.plot(np.angle(zz[0]), rad[0], "o", color=C2, ms=3)
b.set_rticks([1, 2, 3])
b.set_yticklabels([r"$1$", r"$10$", r"$100$"], fontsize=6.5)
b.set_xticklabels([])
b.set_title(r"(b) mode coordinate $z(t)$, $\delta=+5\cdot10^{-4}$:" "\n"
            r"angle $\arg z$, radius $\log_{10}|z/z(0)|$ ($A<0.03$)", fontsize=8)
save(fig, "fig_hopf_orbit.pdf")

# ================================================================ Chapter 4: <r>(eps)
Q = json.load(open(os.path.join(OUT, "v3p4_gue_transition.json")))
ex = Q["extended"]
eps = np.array(ex["eps"])
fig, ax = plt.subplots(figsize=(6.3, 2.6))
for fam, col, lab_ in (("G", C0, "(G) GUE-type"), ("S", C1, "(S) geometric phase"), ("O", C2, "(O) GOE-type")):
    ax.semilogx(eps, ex[fam]["r"], "o-", color=col, ms=3, lw=0.8, label=lab_)
for val, nm in ((0.3863, "Poisson"), (0.5307, "GOE"), (0.5996, "GUE")):
    ax.axhline(val, color="0.5", ls="--", lw=0.6)
    ax.text(1.15e-4, val + 0.004, nm, fontsize=7, color="0.35")
ax.axhline(Q["window"]["G"]["pred"], color=C0, ls=":", lw=0.8)
ax.text(1.15e-4, Q["window"]["G"]["pred"] - 0.019, r"first-order plateau $\langle r\rangle_1$", fontsize=7, color=C0)
ax.axvline(1 / Q["N"], color="k", lw=0.6)
ax.text(1 / Q["N"] * 1.15, 0.405, r"$1/\dim W$", fontsize=7)
ax.axvline(ex["G"]["eps_one"], color="k", lw=0.6, ls="-.")
ax.text(ex["G"]["eps_one"] * 1.1, 0.405, r"$\varepsilon_1$", fontsize=7)
ax.set_ylim(0.37, 0.63)
ax.set_xlabel(r"$\varepsilon$"), ax.set_ylabel(r"$\langle r\rangle$")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 0.06))
ax.grid(alpha=0.25, lw=0.5)
save(fig, "fig_quantum_chaos_crossover.pdf")

# ================================================================ Chapter 6: v_2 growth
D = json.load(open(os.path.join(OUT, "v3p6_level5_v2.json")))
ms = [m for m in sorted(map(int, D["levels"])) if m >= 2]
n = np.array([D["levels"][str(m)]["n"] for m in ms], float)
x = np.array([D["levels"][str(m)]["v2_over_n"] for m in ms])
me = sorted(map(int, D["e_plus_1"]))
ne = np.array([D["levels"][str(m)]["n"] for m in me], float)
ce = np.array([D["e_plus_1"][str(m)] for m in me]) / ne
i3 = n >= 192
nn = np.logspace(np.log10(20), 9, 200)
g0, lc0 = np.polyfit(np.log(n[i3]), np.log(x[i3]), 1)
L = D["profile"]["L_best"]
g1, lc1 = np.polyfit(np.log(n[i3]), np.log(x[i3] - L), 1)
u3 = 1 / np.log2(n[-3:])
Mx = np.column_stack([np.ones(3), u3, -u3 * x[-3:]])
p0, p1, q1 = np.linalg.solve(Mx, x[-3:])
un = 1 / np.log2(nn)
fig, ax = plt.subplots(figsize=(6.3, 2.9))
ax.semilogx(n, x, "o", color=C0, ms=4.5, zorder=3, label=r"$v_2|\mathrm{Jac}^{H}(Y_m)|/n_m$, $m=2,\dots,8$")
ax.semilogx(ne, ce, "s", color=C1, ms=4, zorder=3, label=r"$(e_m+1)/n_m\ \geq c_e$, $m=2,\dots,6$")
ax.semilogx(nn, np.exp(lc0) * nn ** g0, "--", color=C0, lw=0.8, label=rf"power law, limit $0$ ($n^{{{g0:.3f}}}$)")
ax.semilogx(nn, L + np.exp(lc1) * nn ** g1, "-.", color=C2, lw=0.8, label=rf"$L+an^{{-g}}$, $L={L:.3f}$")
far = nn >= n[-3]                                                    # fitted through m = 6, 7, 8
ax.semilogx(nn[far], ((p0 + p1 * un) / (1 + q1 * un))[far], ":", color="0.3", lw=1.2,
            label=rf"Padé [1/1] in $1/\log_2 n$ ($m=6,7,8$), limit ${p0:.3f}$")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel(r"$n_m$"), ax.set_ylim(-0.02, 0.8)
ax.legend(frameon=False, loc="upper right")
ax.grid(alpha=0.25, lw=0.5)
save(fig, "fig_v2_growth.pdf")
print(f"   v2 fits: power exponent {g0:.4f}; L* = {L:.4f}, g = {-g1:.3f}; Pade limit {p0:.4f}")
