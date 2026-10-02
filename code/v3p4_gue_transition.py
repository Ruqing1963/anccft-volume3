"""Volume III, Paper 4: the unitary remainder U of the geodesic edge operator of Y_168 on the
full space W = (im Psi)^perp (dim W = 4n + 6 = 678), perturbed by U(eps) = U_0 exp(i eps V), and
the window 0 <= eps <= 1/dim W (chapters/ch04-s06-gue-crossover.tex, subsection "The window eps <= 1/dim W").

  (A) U_0 = W^t L_E W / sqrt(q): real orthogonal, Z/3-graded (J U_0 J^-1 = w U_0, w = e^{2 pi i/3},
      J = w^type on edges); 48 distinct eigenphases, the cube roots of the 16 of U_3; atom 630/678
      = 0.929 at s = 0 of the unfolded spacing measure.
  (B) perturbations, Tr V^2 = dim W:  (G) GUE-type;  (S) geometric phase: a random U(1) potential
      d_e in (-pi, pi) on the edges, compressed, V = W^t diag(d) W (real symmetric; it commutes
      with J, so U(eps) stays Z/3-covariant);  (O) GOE-type (real, not graded).  Certificate
      kappa_{1,2}(V) != 0: no antiunitary time reversal of the family (Prop. V3P4:prop:noT).
  (W) the requested window: 21 values eps = 0, ..., 1/dim W, 16 samples each: <r>, fraction of
      spacings below 0.1 mean, and the first-order prediction <r>_1 of the plateau computed from
      the eigenvalues of P_k V P_k on the 48 eigenspaces (limit eps -> 0+, proved).
  (X) the extended scan eps = 1e-4 .. 10 (26 values, 8 samples): where the crossover actually is,
      eps_half, eps_sat, <r>_sat, against eps_1 = (2 pi / N) / mean first-order gap.
  (P) spacing histograms P(s) and maximum-likelihood Brody fits at eps = 1/dim W, eps_1 and
      eps = 3 (saturation), with KS distances to the Poisson / GOE / GUE surmises.
  output: data/v3p4_gue_transition.json, v3p4_r_vs_eps.png, v3p4_Ps.png.

Run from code/:  python v3p4_gue_transition.py      (numpy, scipy, matplotlib; seeded; ~15 min)
"""
import json
import math
import os
import time

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import erf

from a2_common import standard_covers

t0 = time.time()
q = 2
TWO_PI = 2 * math.pi
R_POISSON, R_GOE, R_GUE = 0.3863, 0.5307, 0.5996
rng = np.random.default_rng(2027)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT, exist_ok=True)
ZERO = 1e-10                                     # spacings below this are exact coincidences


def elapsed():
    return f"[{time.time() - t0:6.0f} s]"


# ================================================================ (A) U_0 on W
def ops(X):
    E, idx, n = X.edges, X.eidx, X.n
    m = len(E)
    chambers = set(frozenset(t) for t in X.tris)
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
    for (a, b, c) in X.tris:
        Q[idx[(b, c)], a] += 1
        Q[idx[(c, a)], b] += 1
        Q[idx[(a, b)], c] += 1
    return L, S, T, Q


Y = standard_covers()["Y_168"]
L, S, T, Q = ops(Y)
u_, s_, vt_ = np.linalg.svd(np.hstack([S.T, T.T, Q]).T.astype(float))
W = vt_[int((s_ > 1e-9 * s_.max()).sum()):].T
N = W.shape[1]
assert N == 4 * Y.n + 6 == 678
U0 = W.T @ L.astype(float) @ W / math.sqrt(q)
assert np.abs(L.astype(float) @ W - W @ (W.T @ L @ W)).max() < 1e-9     # W is L_E-invariant
assert np.abs(U0.T @ U0 - np.eye(N)).max() < 1e-10                     # real orthogonal
tail = np.array([Y.tau[e[0]] for e in Y.edges])
w3 = np.exp(TWO_PI * 1j / 3)
J = W.T @ (w3 ** tail[:, None] * W)                                      # compressed grading
assert np.abs(J.conj().T @ J - np.eye(N)).max() < 1e-10
sgn = min((1, -1), key=lambda e: np.abs(J @ U0 @ J.conj().T - w3 ** e * U0).max())
assert np.abs(J @ U0 @ J.conj().T - w3 ** sgn * U0).max() < 1e-10
MEAN_SP = TWO_PI / N


def clusters(ph, tol=1e-6):
    ph = np.sort(np.mod(ph, TWO_PI))
    reps, mult = [ph[0]], [1]
    for x in ph[1:]:
        if x - reps[-1] < tol:
            mult[-1] += 1
        else:
            reps.append(x)
            mult.append(1)
    if len(reps) > 1 and reps[0] + TWO_PI - reps[-1] < tol:
        mult[0] += mult.pop()
        reps.pop()
    return np.array(reps), np.array(mult)


def spacings(ph):
    ph = np.sort(np.mod(ph, TWO_PI))
    return np.diff(np.r_[ph, ph[0] + TWO_PI]) * N / TWO_PI            # unfolded, mean exactly 1


def rstat(s):
    """<r> over consecutive pairs; exact zeros (< ZERO) are zero, pairs 0/0 are excluded.
    Returns (<r>, fraction of excluded pairs)."""
    s = np.where(s < ZERO, 0.0, s)
    s1, s2 = s, np.roll(s, -1)
    mx = np.maximum(s1, s2)
    ok = mx > 0
    r = np.minimum(s1, s2)[ok] / mx[ok]
    return (float(r.mean()) if r.size else float("nan")), float(1 - ok.mean())


ph0 = np.angle(np.linalg.eigvals(U0))
reps0, mult0 = clusters(ph0)
s0 = spacings(ph0)
atom = float(np.mean(s0 < 1e-6))
assert len(reps0) == 48 and abs(atom - 630 / 678) < 1e-12
assert sorted(mult0.tolist()) == sorted([21] * 24 + [9] * 6 + [7] * 12 + [6] * 6)
r00, excl0 = rstat(s0)
print(f"(A) Y_168: dim W = {N}; U_0 = W^t L_E W / sqrt 2 real orthogonal, J U_0 J^-1 = w^{sgn} U_0 "
      f"(Z/3 grading); {len(reps0)} distinct eigenphases (cube roots of the 16 of U_3), "
      f"multiplicities 21^24 9^6 7^12 6^6; atom at s = 0: {atom:.4f} = 630/678; "
      f"<r>(0) undefined: {excl0:.3f} of the pairs are 0/0   {elapsed()}")
EIG = []
for th in reps0:
    _, sv, vh = np.linalg.svd(U0 - np.exp(1j * th) * np.eye(N))
    EIG.append(vh[sv < 1e-8].conj().T)
assert [B.shape[1] for B in EIG] == mult0.tolist()


# ================================================================ (B) perturbations
def normalize(V):
    return V * math.sqrt(N / float(np.real(np.trace(V @ V))))


def V_gue(rng):
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    return normalize((A + A.conj().T) / 2)


def V_goe(rng):
    A = rng.standard_normal((N, N))
    return normalize((A + A.T) / 2)


def V_phase(rng):
    d = rng.uniform(-math.pi, math.pi, size=len(Y.edges))
    return normalize(W.T @ (d[:, None] * W))


FAMS = {"G": V_gue, "S": V_phase, "O": V_goe}


def kappa(U, V):
    U2 = U @ U
    return np.trace(V @ U @ V @ U2 @ V) - np.trace(V @ U2 @ V @ U @ V)


for fam, gen in FAMS.items():
    V = gen(rng)
    assert abs(kappa(U0, V)) > 1e-3
    if fam == "S":
        assert np.abs(J @ V - V @ J).max() < 1e-10                   # geometric V keeps the grading
print("(B) kappa_{1,2}(V) != 0 for one sample of each family: U(eps) has no antiunitary time "
      "reversal (Prop. V3P4:prop:noT); the geometric phase family (S) commutes with J")


def perturbed(w, v, eps):
    return np.angle(np.linalg.eigvals(U0 @ ((v * np.exp(1j * eps * w)) @ v.conj().T)))


def first_order(V):
    """Prediction of lim_{eps->0+} <r>: inside the cluster of e^{i theta_k} the phases are
    theta_k + eps mu_j + o(eps), mu_j the eigenvalues of P_k V P_k (Prop. V3P4:prop:firstorder);
    r is scale invariant, so the mu_j-gaps give the internal ratios and the two pairs that
    straddle a cluster boundary give r -> 0.  Also the mean first-order gap."""
    tot, gaps = 0.0, []
    for B in EIG:
        mu = np.sort(np.linalg.eigvalsh(B.conj().T @ V @ B))
        g = np.diff(mu)
        gaps += list(g)
        tot += float(np.sum(np.minimum(g[:-1], g[1:]) / np.maximum(g[:-1], g[1:])))
    return tot / N, np.array(gaps)


# ================================================================ (W) window 0 <= eps <= 1/N
EPSW = np.linspace(0, 1 / N, 21)
NSW = 16
WIN = {}
for fam, gen in FAMS.items():
    R = np.full((NSW, len(EPSW)), np.nan)
    F = np.zeros((NSW, len(EPSW)))
    pred, gaps = [], []
    for k_ in range(NSW):
        V = gen(rng)
        w, v = np.linalg.eigh(V)
        p, g = first_order(V)
        pred.append(p)
        gaps.append(g)
        for i, eps in enumerate(EPSW[1:], 1):
            s = spacings(perturbed(w, v, eps))
            R[k_, i], _ = rstat(s)
            F[k_, i] = np.mean(s < 0.1)
    F[:, 0] = atom
    WIN[fam] = dict(r=np.r_[np.nan, R[:, 1:].mean(0)], r_sd=np.nanstd(R[:, 1:]) / math.sqrt(NSW), f=F.mean(0),
                    pred=float(np.mean(pred)), pred_sd=float(np.std(pred) / math.sqrt(NSW)),
                    gap=float(np.concatenate(gaps).mean()))
    print(f"(W) family {fam}: done   {elapsed()}")
print(f"(W) window eps in [0, 1/N] = [0, {1 / N:.3e}], {len(EPSW)} values, {NSW} samples:")
print("    fam  <r> at eps = 1/(20N), 1/(4N), 1/(2N), 1/N | <r>_1 (1st order, eps->0+) | frac(s<0.1) at 1/N"
      " | max first-order splitting at 1/N, in mean spacings")
for fam, d in WIN.items():
    sel = [1, 5, 10, 20]
    spl = (1 / N) * 4 * math.sqrt(21 / N) / MEAN_SP        # semicircle diameter of a 21-cluster
    print(f"    {fam}    " + " ".join(f"{d['r'][i]:.4f}" for i in sel) +
          f" | {d['pred']:.4f} +- {d['pred_sd']:.4f} | {d['f'][-1]:.4f} | {spl:.4f}")
    # the window is the first-order plateau: <r> is flat and equals the first-order prediction
    assert np.all(np.abs(d["r"][1:] - d["pred"]) < 0.02)
    assert d["f"][-1] > 0.9
    assert d["r"][1:].max() < R_GUE - 0.05
print("    checks: on the whole window <r>(eps) is constant up to sampling error and equals the "
      "first-order plateau <r>_1 (within 0.02); > 90% of the spacings are still below 0.1; "
      "<r> stays more than 0.05 below the GUE value")

# ================================================================ (X) extended scan
EPSX = np.logspace(-4, 1, 26)
NSX = 8
EXT = {}
for fam, gen in FAMS.items():
    R = np.zeros((NSX, len(EPSX)))
    F = np.zeros((NSX, len(EPSX)))
    gaps = []
    for k_ in range(NSX):
        V = gen(rng)
        w, v = np.linalg.eigh(V)
        gaps.append(first_order(V)[1])
        for i, eps in enumerate(EPSX):
            s = spacings(perturbed(w, v, eps))
            R[k_, i], _ = rstat(s)
            F[k_, i] = np.mean(s < 0.1)
    r, f = R.mean(0), F.mean(0)
    k = int(np.argmax(f < 0.5))
    t = (f[k - 1] - 0.5) / (f[k - 1] - f[k])
    eps_half = float(np.exp(np.log(EPSX[k - 1]) + t * (np.log(EPSX[k]) - np.log(EPSX[k - 1]))))
    r_sat = float(r[EPSX >= 2].mean())
    eps_sat = float(EPSX[int(np.argmax(r >= r_sat - 0.01))])
    eps_one = MEAN_SP / float(np.concatenate(gaps).mean())
    EXT[fam] = dict(r=r, f=f, eps_half=eps_half, eps_sat=eps_sat, eps_one=eps_one, r_sat=r_sat,
                    r_min=float(r.min()), eps_rmin=float(EPSX[int(np.argmin(r))]))
    print(f"(X) family {fam}: done   {elapsed()}")
print("(X) extended scan eps = 1e-4..10 (26 values, 8 samples):")
print("    fam  eps_half  eps_1   eps_sat  eps_sat*N  <r>_sat | min <r> (at eps)")
for fam, d in EXT.items():
    print(f"    {fam}    {d['eps_half']:.4f}   {d['eps_one']:.4f}  {d['eps_sat']:.4f}  {d['eps_sat'] * N:7.1f}"
          f"    {d['r_sat']:.4f} | {d['r_min']:.4f} ({d['eps_rmin']:.4f})")
    assert abs(d["r_sat"] - R_GUE) < 0.015
    assert d["eps_sat"] > 50 / N
print(f"    checks: <r>_sat within 0.015 of {R_GUE} for every family; saturation needs eps > 50/N")

# ================================================================ (P) spacing distributions
def brody_alpha(b):
    return math.gamma((b + 2) / (b + 1)) ** (b + 1)


def brody_mle(s):
    s = s[s > 1e-12]
    nll = lambda b: -np.sum(np.log(b + 1) + np.log(brody_alpha(b)) + b * np.log(s) - brody_alpha(b) * s ** (b + 1))
    return float(minimize_scalar(nll, bounds=(0.0, 6.0), method="bounded").x)


CDF = {"Poisson": lambda s: 1 - np.exp(-s), "GOE": lambda s: 1 - np.exp(-math.pi * s ** 2 / 4),
       "GUE": lambda s: erf(2 * s / math.sqrt(math.pi)) - (4 * s / math.pi) * np.exp(-4 * s ** 2 / math.pi)}
PDF = {"Poisson": lambda s: np.exp(-s), "GOE": lambda s: math.pi * s / 2 * np.exp(-math.pi * s ** 2 / 4),
       "GUE": lambda s: 32 / math.pi ** 2 * s ** 2 * np.exp(-4 * s ** 2 / math.pi)}


def ks(s, F):
    s = np.sort(s)
    k = len(s)
    Fs = F(s)
    return float(max(np.abs(np.arange(1, k + 1) / k - Fs).max(), np.abs(np.arange(k) / k - Fs).max()))


NSP = 24
PS = {}
for fam in ("G", "S"):
    for lab, eps in (("1/N", 1 / N), ("eps_1", EXT[fam]["eps_one"]), ("3", 3.0)):
        sp, rs = [], []
        for k_ in range(NSP):
            V = FAMS[fam](rng)
            w, v = np.linalg.eigh(V)
            s = spacings(perturbed(w, v, eps))
            sp.append(s)
            rs.append(rstat(s)[0])
        s = np.concatenate(sp)
        b = brody_mle(s)
        PS[(fam, lab)] = dict(eps=eps, s=s, r=float(np.mean(rs)), b=b,
                              ks={k: ks(s, F) for k, F in CDF.items()})
print(f"(P) spacing distributions ({NSP} samples, {NSP * N} unfolded spacings each):")
print("    fam  eps            <r>     Brody b   KS: Poisson  GOE     GUE")
for (fam, lab), d in PS.items():
    print(f"    {fam}    {lab:6s}={d['eps']:.4f}  {d['r']:.4f}  {d['b']:6.3f}     "
          f"{d['ks']['Poisson']:.4f}      {d['ks']['GOE']:.4f}  {d['ks']['GUE']:.4f}")
for fam in ("G", "S"):
    d1, d3 = PS[(fam, "1/N")], PS[(fam, "3")]
    assert d1["b"] < 0.05                                 # degenerate: Brody pinned at its lower end
    assert d3["ks"]["GUE"] < 0.02 and d3["ks"]["GUE"] < d3["ks"]["GOE"] / 3
    assert 1.3 < d3["b"] < 1.7                            # MLE Brody of a GUE spectrum (not 2)
print("    checks: at eps = 1/N the Brody MLE is pinned at b ~ 0 (the atom at s = 0 dominates); at "
      "eps = 3 the KS distance to the GUE surmise is < 0.02 and < 1/3 of the distance to GOE, and the "
      "Brody MLE is 1.3-1.7 (the GUE value of the estimator, Remark V3P4:rem:brodyfit)")

# ================================================================ output
res = dict(N=N, atom=atom, window=dict(eps=EPSW.tolist(), **{f: dict(r=d["r"].tolist(), f=d["f"].tolist(),
                                                                       pred=d["pred"], pred_sd=d["pred_sd"],
                                                                       mean_gap=d["gap"]) for f, d in WIN.items()}),
           extended=dict(eps=EPSX.tolist(), **{f: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                    for k, v in d.items()} for f, d in EXT.items()}),
           spacing={f"{fam}@{lab}": dict(eps=d["eps"], r=d["r"], brody=d["b"], ks=d["ks"])
                    for (fam, lab), d in PS.items()})
with open(os.path.join(OUT, "v3p4_gue_transition.json"), "w") as fh:
    json.dump(res, fh, indent=1)
np.savez_compressed(os.path.join(OUT, "v3p4_spacings.npz"),
                    **{f"{fam}_{lab.replace('/', '')}": d["s"] for (fam, lab), d in PS.items()})

import matplotlib                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                         # noqa: E402

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for fam, c in zip(FAMS, ("C0", "C1", "C2")):
    ax[0].plot(EPSW[1:] * N, WIN[fam]["r"][1:], "o-", ms=3, color=c, label=f"({fam})")
    ax[0].axhline(WIN[fam]["pred"], color=c, ls=":", lw=0.8)
    ax[1].semilogx(EPSX, EXT[fam]["r"], "o-", ms=3, color=c, label=f"({fam})")
for a_ in ax:
    for val, nm in ((R_POISSON, "Poisson"), (R_GOE, "GOE"), (R_GUE, "GUE")):
        a_.axhline(val, color="gray", lw=0.6, ls="--")
        a_.text(a_.get_xlim()[0], val, nm, fontsize=7, va="bottom", color="gray")
    a_.set_ylabel(r"$\langle r\rangle$"), a_.grid(alpha=0.3), a_.legend(fontsize=8)
ax[0].set_xlabel(r"$\varepsilon \cdot$ dim W   (window 0 < $\varepsilon$ <= 1/dim W; dotted: first-order plateau)")
ax[1].axvline(1 / N, color="k", lw=0.8)
ax[1].set_xlabel(r"$\varepsilon$   (vertical line: 1/dim W)")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v3p4_r_vs_eps.png"), dpi=150)

fig, ax = plt.subplots(2, 3, figsize=(12, 6.5))
xs = np.linspace(0, 4, 400)
for i, fam in enumerate(("G", "S")):
    for j, lab in enumerate(("1/N", "eps_1", "3")):
        d = PS[(fam, lab)]
        a_ = ax[i, j]
        a_.hist(d["s"], bins=np.arange(0, 4.01, 0.1), density=True, color="0.75")
        for nm, st in (("Poisson", ":"), ("GOE", "--"), ("GUE", "-")):
            a_.plot(xs, PDF[nm](xs), st, lw=1, label=nm)
        b, al = d["b"], brody_alpha(d["b"])
        a_.plot(xs[1:], (b + 1) * al * xs[1:] ** b * np.exp(-al * xs[1:] ** (b + 1)), "r-", lw=0.8,
                label=f"Brody b={b:.2f}")
        a_.set_ylim(0, 1.3), a_.set_title(f"({fam}) eps={d['eps']:.4g}, <r>={d['r']:.3f}", fontsize=9)
        a_.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v3p4_Ps.png"), dpi=150)
print(f"\nALL CHECKS PASSED: (A) (B) (W) (X) (P).   {elapsed()}")
