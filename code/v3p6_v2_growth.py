"""Volume III, Paper 6, open problem (b): the 2-part of the Hecke Jacobian along the canonical
2-tower Y_m over X_3 (chapters/ch06-s08-two-adic-tower.tex).

  (LV) levels m = 2..5 of the tower, n = 24, 192, 768, 6144, D^H = 7I + A1 - 2 A1^T;
  (BL) block decomposition over the characters of the central elementary abelian layer
       N_m = P_m/P_{m+1}: the trivial block is D^H(Y_m), the twisted blocks D^H_chi have entries
       with signs chi(voltage), and Tr A1(Y_{m+1})^k = sum_chi Tr A_chi^k exactly (k = 3, 6);
  (V2) v_2|Jac^H(Y_{m+1})| = v_2|Jac^H(Y_m)| - d_m + sum_{chi != 1} v_2 det D^H_chi, validated
       at m+1 = 3 (18 -> 99) and m+1 = 4 (99 -> 289) against the direct elimination, then
       applied at m+1 = 5 (n = 6144), where it gives the exact value;
  (SM) invariant-factor valuation profiles (how many Z/2^k) by 2-adic elimination mod 2^30, with
       a cap check; agreement with the exact Smith form at m = 2, 3;
  (E)  e_m = number of even invariant factors = dim_F2 ker(I + A1) - 1 = dim F_2[G_m]/(F_2[G_m] F_m) - 1
       (Schur complement over F_2[G_m]); each twisted block has exactly e_m + 1 even factors;
       (e_m + 1)/n_m is non-increasing (Nakayama);
  (AU) depth of F_m in the augmentation filtration of F_2[G_m], and the Loewy length;
  (SY) the automorphisms a_x -> a_{2x}, a_x -> a_{x+1} of the tower permute the nontrivial
       characters of each layer transitively, so all twisted blocks of a layer have the same
       Smith form over Z_2;
  (TO) total order: (1/n) log|Jac^H(Y_5)| = 2.062520 from the blocks, and v_2 <= log_2|Jac^H|;
  (FD) direct 2-adic elimination of the full 6144 x 6144 matrix (about 3-4 minutes; skipped
       with --quick): same v_2 and e_5, and the exact invariant-factor profile at level 5;
  (TB),(GR) the table, the sequence v_2/n_m, index ratios, fitted exponent, and the verdict
       that the data support.

Run:  python v3p6_v2_growth.py [--quick]      (numpy, scipy.sparse; about 5 minutes, 1 with --quick)
"""
import math
import sys
import time
from collections import Counter, defaultdict

import numpy as np

from a2_common import GIDX, T, TowerRing, standard_covers, torsion

t0 = time.time()


def elapsed():
    return f"[{time.time() - t0:7.1f} s]"


# ---------------------------------------------------------------- tower levels
def closure(R, gens):
    one = R.normalize(R.one())
    seen, frontier = {one}, [one]
    while frontier:
        new = []
        for g in frontier:
            for h in gens:
                k = R.normalize(R.mul(g, h))
                if k not in seen:
                    seen.add(k)
                    new.append(k)
        frontier = new
    return seen


class Level:
    """Level m of the canonical 2-tower: vertices (g, s), g in G_m = P/P_m, s in Z/3, and the
    type-increasing arcs (g, s) -> (g theta_{s,x}, s+1)."""

    def __init__(self, m):
        self.m = m
        self.R = TowerRing(m)
        self.th = self.R.voltages()
        self.G = sorted(closure(self.R, self.th))
        self.gidx = {g: i for i, g in enumerate(self.G)}
        self.verts = [(g, s) for g in self.G for s in range(3)]
        self.vid = {v: i for i, v in enumerate(self.verts)}
        self.n = len(self.verts)
        assert self.n == 3 * 2 ** (3 * (m - 1) - (m - 1) // 3)
        src, dst = [], []
        for (g, s) in self.verts:
            for x in range(7):
                w = self.step((g, s), x)
                src.append(self.vid[(g, s)])
                dst.append(self.vid[w])
            for (x, y, z) in T:
                assert self.step(self.step(self.step((g, s), x), y), z) == (g, s)
        self.src, self.dst = np.array(src), np.array(dst)
        assert len(set(zip(src, dst))) == 7 * self.n

    def mul(self, a, b):
        return self.R.normalize(self.R.mul(a, b))

    def inv(self, a):
        return self.R.normalize(self.R.inv(a))

    def step(self, v, x):
        return (self.mul(v[0], self.th[GIDX[(v[1], x)]]), (v[1] + 1) % 3)

    def A1(self):
        A = np.zeros((self.n, self.n), dtype=np.int64)
        A[self.src, self.dst] = 1
        return A

    def DH(self):
        A = self.A1()
        return 7 * np.eye(self.n, dtype=np.int64) + A - 2 * A.T


# ---------------------------------------------------------------- 2-adic elimination
def _odd_round(A, k):
    """In-place elimination mod 2^k with odd pivots (any odd entry is a Smith pivot over Z_2).
    Rows/columns are permuted so that the pivots occupy A[:r, :r] and the remaining block
    A[r:, r:] has only even entries.  Returns r."""
    mask = (1 << k) - 1
    nr, nc = A.shape
    r, j, last = 0, 0, nc - 1
    B = 512
    while j <= last and r < nr:
        odd = np.flatnonzero(A[r:, j] & 1)
        if odd.size == 0:
            if j != last:
                A[:, [j, last]] = A[:, [last, j]]
            last -= 1
            continue
        i = r + int(odd[0])
        if i != r:
            A[[r, i]] = A[[i, r]]
        piv = int(A[r, j])
        uinv = pow(piv, -1, 1 << k)
        if r + 1 < nr and j + 1 < nc:
            f = (A[r + 1:, j] * uinv) & mask
            nz = np.flatnonzero(f)
            if nz.size:
                rowv = A[r, j + 1:]
                if 4 * nz.size < nr - r - 1:
                    rows = nz + r + 1
                    sub = A[rows, j + 1:]
                    sub -= f[nz, None] * rowv[None, :]
                    sub &= mask
                    A[rows, j + 1:] = sub
                else:
                    for a in range(r + 1, nr, B):
                        b = min(a + B, nr)
                        blk = A[a:b, j + 1:]
                        blk -= f[a - r - 1:b - r - 1, None] * rowv[None, :]
                        blk &= mask
        r += 1
        j += 1
    return r


def smith2_profile(M, K=30):
    """Valuations of the invariant factors of the integer matrix M over Z_2, by elimination
    modulo 2^K.  Returns (Counter {valuation: multiplicity}, number of factors that are
    0 mod 2^K, cap_hit).  int64 is safe: residues < 2^30, products < 2^60."""
    A = np.array(M, dtype=np.int64)
    counts, shift, nzero, cap = Counter(), 0, 0, False
    while A.size:
        k = K - shift
        if k <= 0:
            cap = True
            break
        A &= (1 << k) - 1
        if not A.any():
            nzero = min(A.shape)
            break
        r = _odd_round(A, k)
        if r:
            counts[shift] = r
        A = np.ascontiguousarray(A[r:, r:])
        if A.size:
            assert not (A & 1).any()
            A >>= 1
        shift += 1
    return counts, nzero, cap


def profile_v2(counts):
    return sum(k * c for k, c in counts.items())


def profile_str(counts):
    return " ".join(f"2^{k}:{counts[k]}" for k in sorted(counts) if k > 0) or "-"


# ---------------------------------------------------------------- F_2 linear algebra (packed)
def pack_rows(Mbool):
    """Rows of a 0/1 matrix as little-endian uint64 words."""
    R, C = Mbool.shape
    W = -(-C // 64)
    buf = np.zeros((R, 8 * W), dtype=np.uint8)
    buf[:, :-(-C // 8)] = np.packbits(Mbool.astype(np.uint8), axis=1, bitorder="little")
    return np.ascontiguousarray(buf.view(np.uint64)), C


def unpack_rows(P, C):
    return np.unpackbits(P.view(np.uint8), axis=1, bitorder="little")[:, :C]


def gf2_echelon(P, C):
    """Row echelon form over F_2 of packed rows.  Returns (rows, pivot columns)."""
    M = P.copy()
    R = M.shape[0]
    r, piv = 0, []
    one = np.uint64(1)
    for c in range(C):
        if r == R:
            break
        w, b = divmod(c, 64)
        col = (M[r:, w] >> np.uint64(b)) & one
        nz = np.flatnonzero(col)
        if nz.size == 0:
            continue
        i = r + int(nz[0])
        if i != r:
            M[[r, i]] = M[[i, r]]
        if nz.size > 1:
            M[nz[1:] + r] ^= M[r]
        piv.append(c)
        r += 1
    return M[:r].copy(), piv


def gf2_reduce(v, E, piv):
    """Reduce the packed vector v against the echelon rows E (pivots piv)."""
    v = v.copy()
    one = np.uint64(1)
    for row, c in zip(E, piv):
        w, b = divmod(c, 64)
        if (v[w] >> np.uint64(b)) & one:
            v ^= row
    return v


def gf2_rank(Mbool):
    P, C = pack_rows(Mbool)
    return len(gf2_echelon(P, C)[1])


# ---------------------------------------------------------------- block decomposition
def decompose(Lhi, Llo):
    """Y_{m+1} -> Y_m with group N = P_m/P_{m+1}.  Returns the integer blocks D^H_lambda
    indexed by the characters lambda of N, the fibre dimension d, and the voltage list."""
    m = Llo.m
    assert Lhi.m == m + 1
    proj = {g: Llo.R.normalize(g[:m]) for g in Lhi.G}
    fib = defaultdict(list)
    for g in Lhi.G:
        fib[proj[g]].append(g)
    N = len(Lhi.G) // len(Llo.G)
    assert len(fib) == len(Llo.G) and all(len(f) == N for f in fib.values())
    d = N.bit_length() - 1
    assert d == (2 if m % 3 == 0 else 3)
    rep = {gb: min(f) for gb, f in fib.items()}

    def unit(c):                      # 1 + c phi^m in G_{m+1}
        e = [0] * (m + 1)
        e[0], e[m] = 1, c
        return Lhi.R.normalize(tuple(e))

    for gb in Llo.G:                  # the fibre is the N-orbit of the representative
        assert {Lhi.mul(rep[gb], unit(c)) for c in range(8)} == set(fib[gb])
        assert {Lhi.mul(unit(c), rep[gb]) for c in range(8)} == set(fib[gb])   # central
    edges = []
    for gb in Llo.G:
        gt = rep[gb]
        for s in range(3):
            for x in range(7):
                w = Lhi.mul(gt, Lhi.th[GIDX[(s, x)]])
                wb = proj[w]
                assert wb == Llo.mul(gb, Llo.th[GIDX[(s, x)]])
                h = Lhi.mul(w, Lhi.inv(rep[wb]))
                assert h[0] == 1 and not any(h[1:m])
                c = h[m]
                if m % 3 == 0:
                    assert c & 1 == 0
                edges.append((Llo.vid[(gb, s)], Llo.vid[(wb, (s + 1) % 3)], c))
    chars = [lam for lam in range(8) if (m % 3 or lam % 2 == 0)]
    blocks, adj = {}, {}
    for lam in chars:
        A = np.zeros((Llo.n, Llo.n), dtype=np.int64)
        for v, w, c in edges:
            A[v, w] += 1 - 2 * (bin(lam & c).count("1") & 1)
        adj[lam] = A
        blocks[lam] = 7 * np.eye(Llo.n, dtype=np.int64) + A - 2 * A.T
    assert (blocks[0] == Llo.DH()).all() and (adj[0] == Llo.A1()).all()
    for lam in chars:
        assert (np.abs(adj[lam]) == adj[0]).all()        # entries +-1 on the arcs of Y_m
    return blocks, adj, d, edges


def log_detprime(A1f):
    """log of the product of the nonzero eigenvalues of 7I + A1 - 2 A1^T (float)."""
    a = np.linalg.eigvals(A1f)
    mu = 7 + a - 2 * np.conj(a)
    order = np.argsort(np.abs(mu))
    assert abs(mu[order[0]]) < 1e-6 and abs(mu[order[1]]) > 1e-3
    return float(np.sum(np.log(np.abs(mu[order[1:]]))))


def log_det(A1f):
    a = np.linalg.eigvals(A1f)
    mu = 7 + a - 2 * np.conj(a)
    assert np.abs(mu).min() > 1e-3
    return float(np.sum(np.log(np.abs(mu))))


def sparse_traces(L):
    """Tr A1^3 and Tr A1^6 from the arc lists (exact integers)."""
    from scipy.sparse import csr_matrix
    A = csr_matrix((np.ones(len(L.src)), (L.src, L.dst)), shape=(L.n, L.n))
    A3 = (A @ A @ A).tocsr()
    tr3 = int(round(A3.diagonal().sum()))
    tr6 = int(round(A3.multiply(A3.T).sum()))
    return tr3, tr6


# ================================================================ (LV) levels
levels = {}
for m in (2, 3, 4, 5):
    levels[m] = Level(m)
    print(f"(LV) level {m}: n = {levels[m].n}, |G_m| = {len(levels[m].G)}   {elapsed()}")
n_of = {m: levels[m].n for m in levels}
n_of[1] = 3
assert [n_of[m] for m in (2, 3, 4, 5)] == [24, 192, 768, 6144]

# ================================================================ (SM) direct profiles, m <= 4
known_v2 = {1: 0, 2: 18, 3: 99, 4: 289}
prof, v2, nz = {}, {}, {}
for m in (2, 3, 4):
    DH = levels[m].DH()
    prof[m], nz[m], cap = smith2_profile(DH)
    assert not cap and nz[m] == 1
    v2[m] = profile_v2(prof[m])
    assert v2[m] == known_v2[m]
    print(f"(SM) level {m}: v_2|Jac^H| = {v2[m]} (known {known_v2[m]}), profile {profile_str(prof[m])}, "
          f"zero factors = {nz[m]}   {elapsed()}")
    if m in (2, 3):
        J = torsion(DH.tolist())
        exact = Counter(((d & -d).bit_length() - 1) for d in J)
        exact.pop(0, None)
        assert {k: c for k, c in prof[m].items() if k > 0} == dict(exact)
        print(f"(SM) level {m}: profile agrees with the exact Smith form ({len(J)} torsion factors)")
Y24 = standard_covers()["Y_24"]
assert sorted(torsion(levels[2].DH().tolist())) == sorted(torsion(Y24.DH.tolist()))
v2[1] = 0
prof[1] = Counter()

# ================================================================ (BL),(V2) decomposition
blocks, adjs, dims, blockprof = {}, {}, {}, {}
for m in (2, 3, 4):
    blocks[m], adjs[m], dims[m], _ = decompose(levels[m + 1], levels[m])
    Lhi = levels[m + 1]
    if m + 1 <= 4:
        A = Lhi.A1().astype(float)
        A3f = np.linalg.matrix_power(A, 3)
        tr = [int(round(np.trace(A3f))), int(round(np.sum(A3f * A3f.T)))]
    else:
        tr = list(sparse_traces(Lhi))
    trb = [0, 0]
    for lam, Al in adjs[m].items():
        A3f = np.linalg.matrix_power(Al.astype(float), 3)
        trb[0] += int(round(np.trace(A3f)))
        trb[1] += int(round(np.sum(A3f * A3f.T)))
    assert tr == trb
    print(f"(BL) Y_{m + 1} -> Y_{m}: group (Z/2)^{dims[m]}, {len(blocks[m]) - 1} twisted blocks of size "
          f"{levels[m].n}; Tr A1^3 = {tr[0]}, Tr A1^6 = {tr[1]} equal the block sums   {elapsed()}")
    blockprof[m] = {}
    tot = 0
    for lam, Bk in blocks[m].items():
        if lam == 0:
            continue
        c, z, cap = smith2_profile(Bk)
        assert not cap and z == 0
        blockprof[m][lam] = c
        tot += profile_v2(c)
    s = " ".join(f"chi{lam}:{profile_v2(blockprof[m][lam])}" for lam in sorted(blockprof[m]))
    pred = v2[m] - dims[m] + tot
    print(f"(V2) level {m + 1}: v_2 det D^H_chi = {s}; sum = {tot}; "
          f"{v2[m]} - {dims[m]} + {tot} = {pred}   {elapsed()}")
    if m + 1 <= 4:
        assert pred == known_v2[m + 1]
        print(f"(V2) level {m + 1}: recovers the direct value {known_v2[m + 1]}")
    else:
        v2[5] = pred
        print(f"(V2) level 5: v_2|Jac^H(Y_5)| = {v2[5]} EXACTLY (n = 6144)")
V2_5 = v2[5]

# ================================================================ (E) even invariant factors
e = {}
for m in (2, 3, 4, 5):
    L = levels[m]
    Mb = np.zeros((L.n, L.n), dtype=np.uint8)
    Mb[L.src, L.dst] = 1
    Mb[np.arange(L.n), np.arange(L.n)] ^= 1
    corank = L.n - gf2_rank(Mb)
    e[m] = corank - 1
    if m <= 4:
        assert corank == (L.n - prof[m][0])          # odd pivots = F_2-rank
        assert e[m] == sum(c for k, c in prof[m].items() if k > 0)
    print(f"(E) level {m}: dim_F2 ker(I + A1) = {corank}, e_m = {e[m]}, e_m/n = {e[m] / L.n:.4f}, "
          f"(e_m+1)/n = {(e[m] + 1) / L.n:.4f}   {elapsed()}")
e[1] = 0
for m in (2, 3, 4):                          # twisted blocks: exactly e_m + 1 even factors each
    for lam, c in blockprof[m].items():
        assert levels[m].n - c[0] == e[m] + 1
    print(f"(E) level {m}: every twisted block D^H_chi == D^H(Y_{m}) mod 2 has e_m + 1 = {e[m] + 1} even factors")
for m in (2, 3, 4):
    assert (e[m + 1] + 1) * n_of[m] <= (e[m] + 1) * n_of[m + 1]      # Nakayama bound
    assert e[m + 1] >= e[m]
print("(E) (e_m+1)/n_m is non-increasing and e_m is non-decreasing, m = 2..5")

# Schur complement: ker(I + A1) = left annihilator of F_m = 1 + fbar_1 fbar_0 fbar_2 in F_2[G_m]
depth, loewy, dimquot = {}, {}, {}
for m in (2, 3, 4, 5):
    L = levels[m]
    G, gidx = L.G, L.gidx
    one = L.R.normalize(L.R.one())
    fbar = []
    for s in range(3):
        fbar.append(Counter(L.inv(L.th[GIDX[(s, x)]]) for x in range(7)))

    def galg_mul(P, Q):
        out = Counter()
        for a, ca in P.items():
            for b, cb in Q.items():
                out[L.mul(a, b)] += ca * cb
        return Counter({g: c % 2 for g, c in out.items() if c % 2})

    F = galg_mul(galg_mul(fbar[1], fbar[0]), fbar[2])
    F[one] ^= 1
    F = Counter({g: c for g, c in F.items() if c})
    supp = list(F)
    assert sum(F.values()) % 2 == 0                      # augmentation 0: F in I
    # right multiplication by F on F_2[G]
    RF = np.zeros((len(G), len(G)), dtype=np.uint8)
    for i, g in enumerate(G):
        for k in supp:
            RF[i, gidx[L.mul(g, k)]] ^= 1
    cork = len(G) - gf2_rank(RF)
    assert cork == e[m] + 1
    print(f"(E) level {m}: |supp F_m| = {len(supp)}, dim F_2[G_m]/(F_2[G_m] F_m) = {cork} = e_m + 1   {elapsed()}")

    # (AU) augmentation filtration: I^k = sum_g I^{k-1} (g - 1) over 3 generators of G_m
    gens = [L.th[GIDX[(2, x)]] for x in (0, 1, 2)]       # phi-coefficients 1, alpha, alpha^2
    assert len(closure(L.R, gens)) == len(G)
    perms = []
    for g in gens:
        perms.append(np.array([gidx[L.mul(h, g)] for h in G]))
    nG = len(G)
    Fv = np.zeros(nG, dtype=np.uint8)
    for k in supp:
        Fv[gidx[k]] = 1
    Fp = pack_rows(Fv[None, :])[0][0]
    basis = np.zeros((nG - 1, nG), dtype=np.uint8)       # I^1: g - 1
    for row, g in enumerate(h for h in G if h != one):
        basis[row, [gidx[one], gidx[g]]] = 1
    dims_I = [nG]
    kk, depth[m] = 0, None
    E, piv = gf2_echelon(*pack_rows(basis))
    while True:
        kk += 1
        dims_I.append(len(piv))
        inside = not gf2_reduce(Fp, E, piv).any()
        if inside:
            depth[m] = kk
        if len(piv) == 0:
            break
        B = unpack_rows(E, nG)
        new = []
        for p in perms:
            Bg = np.zeros_like(B)
            Bg[:, p] = B
            new.append(Bg ^ B)
        E, piv = gf2_echelon(*pack_rows(np.vstack(new)))
    loewy[m] = kk                                        # I^kk = 0, I^(kk-1) != 0
    dimquot[m] = nG - dims_I[depth[m]]
    assert depth[m] >= 1 and e[m] + 1 >= dimquot[m]
    print(f"(AU) level {m}: F_m in I^{depth[m]} \\ I^{depth[m] + 1}, Loewy length (I^k = 0) k = {loewy[m]}, "
          f"dim F_2[G]/I^{depth[m]} = {dimquot[m]} <= e_m + 1 = {e[m] + 1}; "
          f"dim I^k = {dims_I[:min(len(dims_I), 12)]}{'...' if len(dims_I) > 12 else ''}   {elapsed()}")

# ================================================================ (SY) symmetries of the tower
# a_x -> a_{2x} (Frobenius, coefficientwise on F_8[[phi; sigma]]) and a_x -> a_{x+1}
# (conjugation by alpha^{-1}) preserve T, hence Gamma, Gamma^0, rho and every U_m.  They act
# on N_m = {1 + c phi^m} by c -> sigma(c) and c -> alpha^(2^m - 1) c, and permute the
# nontrivial characters of N_m transitively: all twisted blocks have the same Smith form.
from a2_common import _ALPHA, _FROB

perm_T = {"x -> 2x": lambda x: (2 * x) % 7, "x -> x + 1": lambda x: (x + 1) % 7}
for name, pi in perm_T.items():
    assert {tuple(pi(x) for x in t) for t in T} == set(T)
print("(SY) T is invariant under x -> 2x and x -> x + 1")


def unit_of(m, c):
    e = [0] * m
    e[0], e[m - 1] = 1, c
    return tuple(e)


def automorphisms(L):
    m, R = L.m, L.R
    const = lambda a: tuple([a] + [0] * (m - 1))
    alpha, alpha_inv = const(_ALPHA[1]), const(_ALPHA[6])
    assert L.mul(alpha, alpha_inv) == R.normalize(R.one())
    def u(x):
        e = [0] * m
        e[0], e[1] = 1, _ALPHA[x]
        return tuple(e)
    u0, u1 = u(0), u(1)
    corr = {0: R.normalize(R.one())}
    for s in (1, 2):                                        # u_1^s u_0^{-s}
        a = R.normalize(R.one())
        for _ in range(s):
            a = L.mul(a, u1)
        for _ in range(s):
            a = L.mul(a, L.inv(u0))
        corr[s] = a

    def psi_frob(v):
        return (R.normalize(tuple(_FROB[c] for c in v[0])), v[1])

    def psi_conj(v):
        g, s = v
        return (L.mul(L.mul(L.mul(alpha_inv, g), alpha), corr[s]), s)
    return {"x -> 2x": psi_frob, "x -> x + 1": psi_conj}


for m in (2, 3, 4, 5):
    L = levels[m]
    psis = automorphisms(L)
    for name, psi in psis.items():
        assert {psi(v) for v in L.verts} == set(L.verts)
        pi = perm_T[name]
        for v in L.verts:
            for x in range(7):
                assert psi(L.step(v, x)) == L.step(psi(v), pi(x))
    print(f"(SY) level {m}: both maps are type-preserving automorphisms of Y_{m}   {elapsed()}")
for m in (2, 3, 4):                                         # action on the layer N_m
    Lhi, Llo = levels[m + 1], levels[m]
    psis_hi, psis_lo = automorphisms(Lhi), automorphisms(Llo)
    cmul = lambda a, b: __import__("a2_common")._F8[a][b]
    expect = {"x -> 2x": lambda c: _FROB[c],
              "x -> x + 1": lambda c: cmul(_ALPHA[(2 ** m - 1) % 7], c)}
    chars = [lam for lam in range(1, 8) if (m % 3 or lam % 2 == 0)]
    for name in psis_hi:
        psi, psi_lo, f = psis_hi[name], psis_lo[name], expect[name]
        for v in Lhi.verts:
            assert Llo.R.normalize(psi(v)[0][:m]) == psi_lo((Llo.R.normalize(v[0][:m]), v[1]))[0]
            for c in range(8):
                w = (Lhi.mul(unit_of(m + 1, c), v[0]), v[1])
                assert psi(w) == (Lhi.mul(unit_of(m + 1, f(c)), psi(v)[0]), v[1])
        # induced permutation of the characters: chi_lam(f(c)) = chi_{lam'}(c)
        def dual(lam):
            vals = [(-1) ** bin(lam & f(c)).count("1") for c in range(8)]
            cands = [l2 for l2 in range(8) if all((-1) ** bin(l2 & c).count("1") == vals[c] for c in range(8))]
            assert len(cands) == 1
            return cands[0]
        perm = {lam: dual(lam) for lam in range(8)}
        orbit, frontier = {chars[0]}, [chars[0]]
        while frontier:
            lam = perm[frontier.pop()]
            if lam not in orbit:
                orbit.add(lam)
                frontier.append(lam)
        if name == "x -> x + 1" and m % 3:
            assert orbit == set(chars)
        if name == "x -> 2x" and m % 3 == 0:
            assert orbit == set(chars)
    profs = [tuple(sorted(blockprof[m][lam].items())) for lam in chars]
    assert len(set(profs)) == 1
    print(f"(SY) layer Y_{m + 1} -> Y_{m}: the automorphisms commute with the projection, act on N_m "
          f"by c -> sigma(c) and c -> alpha^{(2 ** m - 1) % 7} c, permute the {len(chars)} nontrivial "
          f"characters transitively; all twisted blocks have the same valuation profile")

# ================================================================ (TO) total order
hH = 2.063693
logJ = {}
for m in (2, 3, 4):
    logJ[m] = log_detprime(levels[m].A1().astype(float)) - math.log(n_of[m])
assert abs(logJ[2] / 24 - 1.862909) < 1e-5 and abs(logJ[3] / 192 - 2.041580) < 1e-5
assert abs(logJ[4] / 768 - 2.054414) < 1e-5
for m in (2, 3):                         # block identity for the total order
    s = sum(log_det(Al.astype(float)) for lam, Al in adjs[m].items() if lam)
    assert abs(logJ[m] + s - dims[m] * math.log(2) - logJ[m + 1]) < 1e-6
logdet5 = sum(log_det(Al.astype(float)) for lam, Al in adjs[4].items() if lam)
logJ[5] = logJ[4] + logdet5 - 3 * math.log(2)
assert abs(logJ[5] / 6144 - 2.062520) < 1e-5
print(f"(TO) (1/n) log|Jac^H| = " + ", ".join(f"{logJ[m] / n_of[m]:.6f}" for m in (2, 3, 4, 5)) +
      f" (level 5 from the blocks; chapter value 2.062520; h_H = {hH})   {elapsed()}")
for m in (2, 3, 4, 5):
    assert v2[m] <= logJ[m] / math.log(2)
print("(TO) v_2|Jac^H| <= log_2|Jac^H| at every level: " +
      ", ".join(f"{v2[m]} <= {logJ[m] / math.log(2):.1f}" for m in (2, 3, 4, 5)))

# ================================================================ (FD) direct elimination, n = 6144
if "--quick" not in sys.argv:
    print(f"\n(FD) direct 2-adic elimination of the full 6144 x 6144 matrix (about 3-4 minutes) ...   {elapsed()}")
    t1 = time.time()
    c5, z5, cap5 = smith2_profile(levels[5].DH())
    assert not cap5 and z5 == 1
    print(f"(FD) full profile {profile_str(c5)}; v_2 = {profile_v2(c5)}; even factors = "
          f"{sum(c for k, c in c5.items() if k > 0)}; max exponent = {max(c5)}; "
          f"elapsed {time.time() - t1:.0f} s")
    assert profile_v2(c5) == V2_5 and sum(c for k, c in c5.items() if k > 0) == e[5]
    print(f"(FD) agrees with the block value v_2 = {V2_5} and with e_5 = {e[5]}; the block profiles "
          f"(max exponent {max(max(c) for c in blockprof[4].values())}) are not the global one")
    prof[5] = c5
else:
    prof[5] = None
    print("\n(FD) skipped (--quick): level-5 profile only per block")

# ================================================================ table and growth
print("\n(TB) m    n     v_2    e_m   v_2/n    e_m/n   (e_m+1)/n  max exponent  profile")
maxexp = {}
for m in (1, 2, 3, 4, 5):
    if prof[m] is not None:
        maxexp[m] = max([k for k in prof[m]] + [0])
        ps = profile_str(prof[m])
        me = str(maxexp[m])
    else:
        bl = blockprof[4]
        maxexp[m] = None
        ps = "blocks: " + "; ".join(f"chi{lam} [{profile_str(bl[lam])}]" for lam in sorted(bl))
        me = "(block max %d)" % max(max(c) for c in bl.values())
    print(f"(TB) {m}  {n_of[m]:5d}  {v2[m]:5d}  {e[m]:5d}  {v2[m] / n_of[m]:.4f}  {e[m] / n_of[m]:.4f}   "
          f"{(e[m] + 1) / n_of[m]:.4f}    {me:>14s}  {ps}")
for m in (2, 3, 4, 5):                       # observed: largest exponent = log_2|G_m|, multiplicity 6
    if prof[m] is not None:
        assert maxexp[m] == 3 * (m - 1) - (m - 1) // 3 and prof[m][maxexp[m]] == 6
        assert v2[m] <= e[m] * maxexp[m]
print("(TB) observed at every computed level: the largest 2-exponent is log_2|G_m| = log_2(n_m/3) "
      "(3, 6, 8, 11), with multiplicity 6, and v_2 <= e_m * (largest exponent)")

print("\n(GR) growth of v_2 along the tower")
ms = [2, 3, 4, 5]
print("(GR) v_2/n_m        : " + ", ".join(f"{v2[m] / n_of[m]:.4f}" for m in ms))
print("(GR) v_2(m)/v_2(m-1): " + ", ".join(f"{v2[m] / v2[m - 1]:.3f}" for m in (3, 4, 5)) +
      "   (index n_m/n_{m-1} = 8, 4, 8)")
loc = {m: math.log(v2[m] / v2[m - 1]) / math.log(n_of[m] / n_of[m - 1]) for m in (3, 4, 5)}
print("(GR) local exponent log(v_2 ratio)/log(index): " + ", ".join(f"{loc[m]:.3f}" for m in (3, 4, 5)))
x = np.log([n_of[m] for m in ms])
y = np.log([v2[m] for m in ms])
beta, logc = np.polyfit(x, y, 1)
beta3 = np.polyfit(x[1:], y[1:], 1)[0]
print(f"(GR) least-squares fit v_2 ~ c n^beta over m = 2..5: beta = {beta:.3f} (m = 3..5: {beta3:.3f})")
print("(GR) per unit of index v_2/(n/3): " + ", ".join(f"{3 * v2[m] / n_of[m]:.3f}" for m in ms))
print("(GR) e_m/n_m: " + ", ".join(f"{e[m] / n_of[m]:.4f}" for m in ms) +
      ";  (e_m+1)/n_m: " + ", ".join(f"{(e[m] + 1) / n_of[m]:.4f}" for m in ms) +
      ";  successive ratios: " +
      ", ".join(f"{((e[m] + 1) / n_of[m]) / ((e[m - 1] + 1) / n_of[m - 1]):.3f}" for m in (3, 4, 5)))
laysum = {m: sum(profile_v2(c) for c in blockprof[m].values()) for m in (2, 3, 4)}
laybound = {m: (2 ** dims[m] - 1) * (e[m] + 1) for m in (2, 3, 4)}
print("(GR) layer sums sum_chi v_2 det D_chi = " + ", ".join(str(laysum[m]) for m in (2, 3, 4)) +
      " vs bounds (2^d - 1)(e_m + 1) = " + ", ".join(str(laybound[m]) for m in (2, 3, 4)) +
      "; factors " + ", ".join(f"{laysum[m] / laybound[m]:.2f}" for m in (2, 3, 4)))
print(f"(GR) h_H / log 2 = {hH / math.log(2):.3f}")
ratios = [v2[m] / n_of[m] for m in ms]
if all(loc[m] < 1 for m in (3, 4, 5)) and all(ratios[i] > ratios[i + 1] for i in range(3)):
    verdict = ("SUBLINEAR: v_2/n decreases at every level and every local exponent is < 1; "
               f"the data fit v_2 ~ n^{beta:.2f}.  Five levels do not decide the limit.")
elif all(loc[m] >= 1 for m in (3, 4, 5)):
    verdict = "LINEAR: local exponents >= 1."
else:
    verdict = "OSCILLATING / undecided: local exponents are not on one side of 1."
print(f"(GR) verdict supported by the data: {verdict}")
lower = [(1 - n_of[m - 1] / n_of[m]) * (e[m - 1] + 1) / n_of[m - 1] for m in (3, 4, 5)]
print("(GR) proved lower bound v_2(m)/n_m >= (1 - n_{m-1}/n_m)(e_{m-1}+1)/n_{m-1}: " +
      ", ".join(f"{b:.4f}" for b in lower) + " vs actual " +
      ", ".join(f"{v2[m] / n_of[m]:.4f}" for m in (3, 4, 5)))
for m, b in zip((3, 4, 5), lower):
    assert v2[m] / n_of[m] >= b

print(f"\nALL CHECKS PASSED: (LV) (SM) (BL) (V2) (E) (AU) (SY) (TO)"
      f"{'' if '--quick' in sys.argv else ' (FD)'} (TB) (GR).   [{time.time() - t0:.0f} s]")
