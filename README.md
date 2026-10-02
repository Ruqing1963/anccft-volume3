# Algorithmic Non-commutative Class Field Theory, Volume III
## Higher-Rank Buildings, Non-Equilibrium Topology and Higher-Order Games

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23109523.svg)](https://doi.org/10.5281/zenodo.23109523)

**Ruqing Chen**, GUT Geoservice Inc., Rivière-Beaudette, Québec, Canada.

This repository contains the complete monograph (LaTeX sources and the compiled book,
[`ANCCFT_Volume_III.pdf`](ANCCFT_Volume_III.pdf), 104 pages), every verification and computation
script used in it, the numerical data those scripts produce, and the figures. Every numbered
statement of the book is either proved or explicitly labelled as numerical; the numerical ones are
reproduced by the scripts in `code/`.

### Chapters

| Ch. | Title | Source | Verification |
|---|---|---|---|
| 1 | Higher-Rank Buildings: The Cubic Trace Identity, Garland Rigidity, and the Hecke Pencil | `chapters/ch01-*.tex` | `v3p1_verify.py`, `v3p1_ad_trace.py` |
| 2 | Simplicial Sandpiles on $\widetilde A_2$ Complexes: Sign Obstructions, Hodge Factorization, and Avalanche Statistics | `chapters/ch02-*.tex` | `v3p2_verify.py` |
| 3 | Phase Oscillators on Buildings: Chiral Connections, Frustrated Ground States, and Transverse Bifurcations | `chapters/ch03-*.tex` | `v3p3_verify.py`, `v3p3_hopf_search.py` |
| 4 | The Geodesic Edge Operator: Unitary Remainders, Ramanujan Equivalence, and Quantum Spectral Chaos | `chapters/ch04-*.tex` | `v3p4_verify.py`, `v3p4_quantum_chaos.py`, `v3p4_gue_transition.py` |
| 5 | Chamber Games and Higher-Order Interactions: Pairwise Reductions, Interfaces, and Entrainment | `chapters/ch05-*.tex` | `v3p5_verify.py`, `v3p5_nonlinear_general_q.py` |
| 6 | Hecke Algebras on Chambers, Iwasawa Modules, and the 2-adic Valuation Tower | `chapters/ch06-*.tex` | `v3p6_verify.py`, `v3p6_deep_verify.py`, `v3p6_v2_growth.py`, `v3p6_level5_v2.py`, `v3p6_pro2_localization.py` |

### Repository layout

```
ANCCFT_Volume_III.pdf      the compiled monograph
ANCCFT_Volume_III.tex      book driver (preamble, title page, chapter order)
macros.tex                 notation shared with Volumes I-II
bibliography.tex           46 references, checked against Crossref and zbMATH
chapters/                  one file per chapter (chNN-*.tex) and per added section (chNN-sMM-*.tex)
frontmatter/               "State of Volume III" (closing summary of results and open problems)
references/                anccft-vol1-2.aux: labels of Volumes I-II, read by xr-hyper
figures/                   the five vector figures of the book (PDF)
code/                      verification and computation scripts (Python 3, numpy/scipy/sympy/mpmath)
data/                      JSON and NPZ output of the scripts; data/logs: their full run logs
```

### Building the book

```
pdflatex ANCCFT_Volume_III.tex
pdflatex ANCCFT_Volume_III.tex
```

Cross-references to Volumes I–II (printed as "Vol. I–II, Theorem 16.3.2" etc.) are resolved from
`references/anccft-vol1-2.aux` with the label prefix `M-`, so the book builds from this repository
alone. Standard TeX Live or MiKTeX packages suffice.

### Reproducing the computations

Run the scripts from `code/` (Python 3.10+, numpy, scipy; `v3p1_ad_trace.py` also needs sympy,
`v3p3_hopf_search.py` mpmath, `make_figures.py` matplotlib). Each prints its checks and ends with
`ALL CHECKS PASSED`. Running times on a laptop: the six `v3pK_verify.py` batteries from under a
minute to about ten minutes; `v3p6_level5_v2.py` about three minutes (plus about 70 with
`--check7`); `v3p4_gue_transition.py` about 30; `v3p3_hopf_search.py` about 25 (a few with
`--quick`). Scripts that produce data write it to `data/`; `make_figures.py` reads `data/` and
writes `figures/`.

Note on `data/logs/`: `v3p3_hopf_search_full-run_interrupted.log` is the last full run of `v3p3_hopf_search.py`, stopped by request after its scan (S) and locus (L) sections, whose numbers the book cites; `v3p3_hopf_search_quick.log` is the complete `--quick` run that passes all checks (A)-(C). `data/v3p3_hopf_search.json` comes from that quick run.

### Citation

R. Chen, *Algorithmic Non-commutative Class Field Theory, Volume III: Higher-Rank Buildings,
Non-Equilibrium Topology and Higher-Order Games*, 2026. doi:[10.5281/zenodo.23109523](https://doi.org/10.5281/zenodo.23109523).
Machine-readable metadata: [`CITATION.cff`](CITATION.cff).

### Conventions

- Theorems are numbered chapter.section.item. In the summaries below, numbers are section.item
  within the chapter: in Chapter 1, "Thm. 2.3" is Theorem 1.2.3.
- Labels carry the chapter in their prefix (`V3P1:`, …, `V3P6:`) and script names follow the same
  scheme (`v3p1_verify.py`, …), so they cannot collide with the labels of Volumes I–II.
- References to Volumes I–II always name the volume: "[Vol. I–II, Ch. 16]" is chapter 16 of the
  Volumes I–II book.

---

The rest of this file summarizes, chapter by chapter, what is proved and what is checked.

## Chapter 1 — verified statements

Numbers are section.item. In the book they appear as 1.section.item: for example,
"Thm. 2.3" is Theorem 1.2.3.

1. **Grading.** $D\,C^{(3)}D^{-1}=\omega C^{(3)}$, so $\mathrm{Tr}(C^{(3)})^k=0$ for $3\nmid k$ and $\det(I-uC^{(3)})\in\mathbb Z[u^3]$ (Prop. 2.1).
2. **Cubic trace identity.** $\mathrm{Tr}(C^{(3)})^3=\mathrm{Tr}A_1^3-3q|E|+3nq^3$ holds for every $\widetilde A_2$ complex. Under condition (L) it equals $n(q-1)^2(q+1)+3s$, where $s$ is the number of non-simplex type-increasing 3-cycles (Thm. 2.3).
3. **Local Garland rigidity.** Suppose every vertex link is the incidence graph of $\mathrm{PG}(2,q)$ with $q\ge2$ (the Heawood graph when $q=2$). Then $H^1(Y;\mathcal E)=0$ for every unitary local system $\mathcal E$. The proof uses only the link gap $1-\sqrt q/(q+1)>1/2$; no building, group action or flag condition is needed (Thm. 4.3). Consequences:
   - $b_1=0$, and $H_1$, $K(Y)$ and $\mathrm{coker}\,\Delta_1=\mathrm{Jac}_1$ are finite;
   - the character group is finite and unitary representations are infinitesimally rigid, so there is no Brillouin zone (Prop. 5.1);
   - there are no $\mathbb Z_\ell$ towers, and every $\ell$-adic analytic tower has a perfect Lie algebra of dimension $\ge3$ (Prop. 5.3).
4. **Hecke pencil.**
   - $\det(I-uC^{(3)})=\det P(u)$ exactly (Thm. 6.1).
   - Sector lemma: the nontrivial eigenvalues of $\Delta^{\mathrm H}$ have $\mathrm{Re}>0$ (Lem. 6.2).
   - $\det(I-uC^{(3)})=(1-u^3)(1-q^3u^3)(1-q^6u^3)\,Q_0(u^3)$ with $Q_0(1)\neq0$ (Thm. 6.3).
   - $\frac{d}{du}\det(I-uC^{(3)})|_{u=1}=(-1)^n n(q-1)^2(q+1)\,|\mathrm{Jac}^{\mathrm H}(Y)|$ (Thm. 6.4).

5. **First nonzero trace in every rank (Sec. 3, Thm. 3.3, `chapters/ch01-s03-first-nonzero-trace.tex`, `code/v3p1_ad_trace.py`).** For an $\widetilde A_d$ complex whose links are flag complexes of $\mathrm{PG}(d,q)$ and which satisfies the rank-$d$ flag condition, $\mathrm{Tr}(C^{(d+1)})^{d+1}=n\prod_{i=1}^{d}(1-q^i)=(-1)^d n(q-1)^d[d]_q!$; $d=3$: $-n(q-1)^3(q+1)(q^2+q+1)$ (the conjectured $n(q-1)^3(q+1)(q^2+1)$ is false). Proof: excursion expansion of the companion, closed walks = partial flags ($q$-multinomials), Euler's $q$-exponential. Sign $(-1)^d$: the rank-one non-realizability recurs in every odd rank.

## Key clarifications (binding for later chapters)

- **The flag condition is necessary** for $\mathrm{Tr}(C^{(3)})^3=n(q-1)^2(q+1)$. On $Y_{21}$, which fails the condition, $\mathrm{Tr}A_1^3=1029=441+588$, so the true trace is **651**; **63** is only the flag part. Likewise, on $Y_{42}$ the true trace is **630** and the flag part is **126**. The formula is exact on $Y_{168}$, where the value is 504.
- **The multiplicity of $1-u^3$ in $\det(I-uC^{(3)})$ is exactly $b_0=1$.** The Betti numbers $b_1,b_2$ do not enter the vertex companion. They enter only on the edge/branch side, through the factor $(1-u^3)^{\chi}$ in the Kang–Li zeta function, where $\chi=1+b_2$. The proposed $(1-u^3)^{b_2-b_1}P(u)$ is false, because its degree is $3(b_2-b_1)+3n>3n$.
- **Local Garland rigidity** depends only on the spectral gap of the Heawood link, $1-\sqrt2/3\approx0.528595>1/2$. It does not require an ambient Lie-group or building action. This corrects Vol. I–II, Ch. Phys1, Rem. "The obstruction is global": $b_1=0$ is detected locally.
- **"$\mathrm{Jac}_0$ pure torsion" is not the content of $b_1=0$.** $\mathrm{Jac}_0$ is always finite. The content is that $H_1(Y;\mathbb Z)$ and $\mathrm{Jac}_1$ are finite.

## Parameter table (checked by `code/v3p1_verify.py`, q = 2)

| $Y$ | $n$ | $\lvert E\rvert=\lvert F\rvert$ | flag | $\mathrm{Tr}A_1^3$ | $3\lvert F\rvert$ | $3s$ | flag part | $\mathrm{Tr}(C^{(3)})^3$ | $\mathrm{Tr}(C^{(3)})^6$ | $(b_0,b_1,b_2)$ | $H_1(Y;\mathbb Z)$ | $\mathrm{Jac}^{\mathrm H}$ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| $Y_{21}$ | 21 | 147 | no | 1029 | 441 | 588 | 63 | 651 | 15939 | (1,0,20) | $(\mathbb Z/2)^6$ | $(\mathbb Z/7)^{17}\oplus\mathbb Z/147$ |
| $Y_{24}$ | 24 | 168 | no | 1008 | 504 | 504 | 72 | 576 | 12840 | (1,0,23) | $(\mathbb Z/2)^4\oplus\mathbb Z/14$ | $(\mathbb Z/7)^2\oplus\mathbb Z/49\oplus(\mathbb Z/392)^5\oplus\mathbb Z/1176$ |
| $Y_{42}$ | 42 | 294 | no | 1386 | 882 | 504 | 126 | 630 | 15414 | (1,0,41) | $(\mathbb Z/2)^3\oplus(\mathbb Z/4)^2$ | $(\mathbb Z/7)^{18}\oplus\mathbb Z/299593\oplus\mathbb Z/2097151\oplus\mathbb Z/176160684$ |
| $Y_{168}$ | 168 | 1176 | yes | 3528 | 3528 | 0 | 504 | 504 | — | (1,0,167) | — | quoted from Vol. I–II, Ch. X |

The link gap is $1-\sqrt2/3$ at every vertex of every complex. The derivative identity holds
exactly: it equals $-2154391190035524027=-63\cdot3\cdot7^{19}$ on $Y_{21}$ (where $Q_0(1)=7^{18}$)
and $126\cdot|\mathrm{Jac}^{\mathrm H}(Y_{42})|$ on $Y_{42}$. The seven $\mathbb Z/14$ covers
$Y_{42}$ all have identical invariants. The row $Y_{168}$ is quoted from Vol. I–II, Ch. X and was not
recomputed.

## Chapter 2 — verified statements

Theorem numbers are section.item; in the book they are prefixed with 2 (e.g. Thm. 1.5 = Theorem 2.1.5).

1. **Sign obstruction (Thm. 1.5).** A threshold rule is an abelian sandpile when its matrix is a Z-matrix. On every thick complex ($q\ge2$) none of $\Delta_1$, $\partial_2\partial_2^t$, $\Delta_2$, $\Delta^{\mathrm H}$ can be made a Z-matrix by any choice of orientations. The obstructions are local:
   - any three points of $\mathrm{PG}(2,q)$, which are pairwise non-adjacent in the link (for $\Delta_1$);
   - one chamber (for $\partial_2\partial_2^t$);
   - three chambers on one edge (for $\Delta_2$);
   - $\Delta^{\mathrm H}_{xy}\Delta^{\mathrm H}_{yx}=-q$ (for $\Delta^{\mathrm H}$).

   On thin tori $\Delta_2$ *is* coherent: it is the dual honeycomb Laplacian. Consequently edge, chamber and Hecke toppling are not abelian, and have no recurrent group (Cor. 1.6).
2. **Hodge factorization (Thm. 2.2).** $|\mathrm{Jac}_1|\,\mathrm{disc}\ker_{\mathbb Z}\Delta_1=n\,|\mathrm{Jac}_0|\,|\mathrm{Jac}_2|\,\mathrm{disc}\,H_2(Y;\mathbb Z)$. On thick complexes ($b_1=0$) this reads $|\mathrm{Jac}_1|=\det\Delta_1$.
3. **Vertex sandpile (Thm. 3.2, Cor. 3.3).**
   - The mean avalanche size is $\langle s\rangle=n^2L^+_{ss}/(n-1)\in[n/\lambda_{\max},\,n/\lambda_2]$.
   - Hence $\langle s\rangle=\Theta(n)$ on expanding families, against $\Theta(n\log n)$ on 2D tori.
   - The mean-field exponents $\tau=3/2$, $\tau_T=2$, $s\sim T^2$ are only a conjecture (Conj. 4.2).
4. **$Y_{21}$ (Prop. 5.1).**
   - Its 1-skeleton is $K_{7,7,7}$.
   - Thresholds: vertex 14, edge 5, chamber 3, Hecke 7.
   - $\langle s\rangle=29/20$, and $\mathrm{disc}\,H_2=3\cdot7^{29}$.

## Chapter 2 — key clarifications

- **There is no recurrent-configuration group $\mathrm{Pic}_k$ for $k=1,2$.** Only the algebraic group $\mathrm{coker}\,\Delta_k$ exists.
- **$\mathbb Z^{|S_k|-1}/\mathrm{im}$ is the vertex ($k=0$) statement.** For $k=1$ with $b_1=0$, "$|\mathrm{Pic}_1|=\det'\Delta_1$" is a tautology. The content is the factorization above.
- **Exponents are not provable on finite complexes.** The dynamic exponent $z$ is undefined on expanders, where the diameter is $\sim\log n$; use $s\sim T^\gamma$ instead.
- **$\tau\approx1.22$ is an early numerical estimate for the 2D (not 1D) BTW model.** The 1D model has no power law.
- **The divergence from Euclidean lattices is spectral (expansion), not topological.** $b_1(Y)=0$ plays no role in it.

## Chapter 2 — vertex sandpile data (sink = vertex 0, q = 2)

| $Y$ | $n$ | $\lambda_2(\Delta_0)$ | $\langle s\rangle$ | $\frac1n\log\lvert\mathrm{Jac}_0\rvert$ |
|---|---|---|---|---|
| $Y_{21}$ | 21 | 14 | 29/20 | 2.407026 |
| $Y_{24}$ | 24 | 13 | 7031/4186 | 2.426183 |
| $Y_{42}$ | 42 | 8.417424 | 489107/157768 | 2.485136 |
| $Y_{84}$ | 84 | 8.417424 | 6.3933 | 2.532442 |
| $Y_{168}$ | 168 | 8.417424 | 12.9760 | 2.560221 |
## Chapter 3 — verified statements

Theorem numbers are section.item; in the book they are prefixed with 3. The model is
$\dot\theta_j=\omega+\frac{K_1}{2\kappa}\sum_{k\sim j}\sin(\theta_k-\theta_j+A_{jk})+\frac{K_2}{D_2}\sum_{\{j,k,l\}}\sin(\theta_k+\theta_l-2\theta_j+\Phi)$,
with $A=\frac{2\pi\alpha}{3}$ on every type-increasing edge, $\kappa=q^2+q+1$ and $D_2=\kappa(q+1)$.

1. **Chiral connections (Prop. 1.2).** The uniform connection has flux $2\pi\alpha$ on every chamber. The type gauge $\frac{2\pi}{3}\tau$ shifts $\alpha$ by 1. Gauge classes form a torsor under $\mathrm{Hom}(H_1,U(1))$, which has 64 elements on $Y_{21}$.
2. **Gradient flux sector (Thm. 2.3).** For $K_2=0$ the model is a gradient flow for every flux, on any graph: there are no limit cycles and no Hopf bifurcation.
3. **Ground states (Thm. 2.5).** The ground states are the twisted states $\Theta_m=\frac{2\pi m}{3}\tau$ with $m\equiv-\mathrm{round}(\alpha)$, of energy $-K_1|E|\cos\frac{2\pi\beta}{3}$. They carry static currents. Two ground states exist exactly when $\alpha\in\frac12+\mathbb Z$.
4. **Twisted states for all parameters (Thm. 3.1).** They rotate at $\omega+K_2\sin\Phi$, with Jacobian $-c_m\Delta_0$ and $c_m=(K_1\cos\frac{2\pi(m+\alpha)}{3}+2K_2\cos\Phi)/(2\kappa)$. Two twisted states are stable iff $\{\alpha\}\in(\frac14,\frac34)$ when $K_2=0$ (Cor. 3.2).
5. **Cluster subspace (Thm. 3.3).** It is a gradient flow modulo rotation. On the full network, the non-gradient part is the chamber drive $\sum_\Delta m_\Delta\mathbf 1_\Delta$.
6. **Block decomposition (Thm. 3.4).** The Jacobian at a cluster state is $\bigoplus M(\lambda)\otimes I\oplus\{-\sigma_t\}$ over the $\mathbb Z/3$-orbits of eigenvalues of $A_1$. It is real on $Y_{21}$ (Cor. 3.5); on $Y_{168}$ complex blocks occur.
7. **Ott–Antonsen (Prop. 4.1, Thm. 4.2).** The reduction closes iff $K_2=0$. For $K_2=0$ and $\Delta=0$, $V=-\frac12Z^*WZ$ satisfies $\dot V\le-\frac12\sum|H_t|^2(1-r_t^2)$, so there is no chimera and no Hopf bifurcation.
8. **Frequencies and phases (Prop. 2.2, Thm. 5.1, Props. 5.3–5.4).** The frequency is $\frac1n\sum\dot\theta_j=\bar\omega+K_2\langle m_\Delta\rangle_F$, and $\Delta\Omega=K_2\sin\Phi$ on twisted states. The geometric phase of a rigid rotation is 0. Flux-threading monodromy is $\Theta_m\mapsto\Theta_{m+1}$ (order 3).

## Chapter 3 — key clarifications

- **The flux does not force rotation.** The $K_1$ sector is a gradient flow for every flux, and flux breaks the conjugation $\theta\mapsto-\theta$ without breaking the gradient structure.
- **$\Phi_\Delta$ is a Sakaguchi lag, not a gauge field.** The gauge-covariant triadic term is blind to the flux.
- **No Hopf bifurcation** occurs at twisted states, in the cluster subspace, for $K_2=0$, or at any cluster state of $Y_{21}$. Transverse Hopf bifurcations **do occur on $Y_{168}$** (Sec. 3.8, `chapters/ch03-s08-transverse-hopf.tex`, `code/v3p3_hopf_search.py`): e.g. at $(K_1,\alpha,\Phi)=(1,0.55,-1.40)$, $K_2^*=-0.544073288416193055$, $\Omega_H=0.00741439514889$, $d\,\mathrm{Re}\,\mu/dK_2=5.0116$; 52 of 274 losses of stability in a scan. The pair $\pm i\Omega_H$ has multiplicity 21 each (forced by the deck group; $\lambda^3=4(5\pm\sqrt{-7})$), so the classical Hopf theorem does not apply. A real eigenvalue of the $e^{i\pi/3}$ block crosses shortly after ($\delta_{\rm next}=0.0005$–$0.0035$), so these are near Hopf/steady-state interactions. Numerically the loss of stability is hard: no stable limit cycle, the network freezes into a frequency-locked state at every Hopf point tested. Quick check: `python code/v3p3_hopf_search.py --quick` (~5 min).
- **The chimera "$r_0\approx1$, $r_1,r_2$ incoherent" is unproved.** It is excluded for $K_2=0$ with identical oscillators. Numerically it is never the primary instability: among 2196 stable cluster equilibria, none had exactly one transversally stable class. On $Y_{168}$ only solitary and disordered drifting states were found.
- **$\Delta\Omega$ is local.** $D_2=3|F|/\chi(Y)$ holds only for $q=2$, and on $T_6$, $\chi=0$.
- **Trees.** Every connection on a tree is pure gauge, and graphs have no 2-cells. What the mechanism needs is 2-cells, not thickness: $T_6$ carries it too.


## Chapter 4 — verified statements

Theorem numbers are section.item; in the book they are prefixed with 4. $L_E$ acts on type-increasing edges and continues $x\to y$ to $y\to z$ when $\{x,y,z\}$ is not a chamber.

1. **Structure (Lem. 1.2, Props. 1.3–1.4).** Locally $L_E=\bigoplus_y(J-\mathrm{Inc}_y)$, with row and column sums $q^2$. It is block-cyclic in the tail type (not tridiagonal). It satisfies $TL_E=q^2S$, $L_ES^t=q^2T^t$, $L_ET^t=T^tA_1-Q$ and $L_EQ=qT^tA_2-qS^t$.
2. **Vertex block (Thm. 1.5).** $L_E\Psi=\Psi\widehat N$ with $\det(I-u\widehat N)=\det P(u)$.
3. **Unitary remainder (Thm. 2.1).** On $W=\ker S\cap\ker T\cap\ker Q^t$, $L_E^t$ is exactly $\sqrt q$ times an isometry, unconditionally. The $\widetilde A_d$ version with $q^{(d-1)/2}$ is Thm. 2.4.
4. **Lower bound (Thm. 2.2).** Every eigenvalue has $|\lambda|\ge\sqrt q$, so no Betti-number factor appears in $\det(I-uL_E)$.
5. **Kernel and factorization (Lem. 3.1, Thm. 3.2).** $\ker\Psi$ is 6-dimensional unless a nontrivial $a$ lies on $\Gamma_q=\{(q+1)\lambda+q^2\lambda^{-2}:|\lambda|=\sqrt q\}$. Off $\Gamma_q$, $\det(I-uL_E)=\det P(u)\det(I-u\bar L)/((1-u^3)(1-q^3u^3))$ and $\dim W=(q-1)(q+2)n+6$.
6. **Ramanujan equivalence (Thm. 3.3).** The nontrivial spectrum lies on $|\lambda|\in\{q,\sqrt q\}$ iff $Y$ is Ramanujan. All five covers are Ramanujan.
7. **Y_21.** $\det(I-uL_E)=(1-64u^3)(1-8u^3)^{18}(1-2u^3+8u^6)^9(1+5u^3+8u^6)^6$. This is checked numerically in full and exactly at 8 points.
8. **Statistics (Props. 4.1–4.3).** Degenerate spectra have an atom at 0 ($210/226$ on $Y_{168}$). Small $\varepsilon$ splits the spectrum but does not give Wigner–Dyson statistics. Brody $b=2$ is not the GUE surmise.

## Chapter 4 — key clarifications

- **The unconditional statement concerns the remainder $W$ only.** $L_E$ minus its Perron orbit has spectral radius $q$, not $\sqrt q$: $Y_{21}$ has 54 eigenvalues of modulus 2. The full (two-circle) RH is equivalent to Ramanujan.
- **The remainder spectrum is degenerate, not Poisson.** On $Y_{168}$ there are 16 distinct values among 226 phases. Deck groups are abelian and force no degeneracy.
- **Perturbation numerics (seed 1).** For $\varepsilon\le10^{-2}$, 93% of spacings stay below 0.1 of the mean. Near saturation $\langle r\rangle\approx0.595$ (GUE-like) for both real and complex $V$, because $U_3e^{i\varepsilon V}$ has no antiunitary symmetry.
- **What unifies Ramanujan and chaos is structural:** $\mathbb C^E=\mathrm{im}\Psi\oplus W$. The Ramanujan condition lives on $\mathrm{im}\Psi$, and the eigenphases of the unitary quantum map live on $W$. Neither implies the other.
- **Betti numbers (Rem. 2.3).** $\det(I-uL_E)$ has no factor $1-u^3$, because every eigenvalue satisfies $|\lambda|\ge\sqrt q$. The Kang–Li factor $(1-u^3)^{\chi}$ recorded under Chapter 1 therefore belongs to the branch operator, not to $L_E$.



## Chapter 5 — verified statements

Theorem numbers are section.item; in the book they are prefixed with 5.

1. **Pairwise reduction (Thm. 1.2).** On any 2-complex, $\pi_i=(r/3-1)D_2(i)s_i+(r/3)\sum_j w_{ij}s_j$, where $w_{ij}$ counts the chambers on edge $ij$. Under (L), $w=q+1$, and the normalized payoff is $(r/3-1)s_i+\frac{r}{3\kappa}N_C(i)$.
2. **Mean field (Prop. 1.4).** $D$ is dominant for $r<3$ in every configuration. The Fermi mean field is $\dot x=x(1-x)\tanh((r/3-1)/2K)$, so $r_c=3$ for every $q$.
3. **Zero noise (Lem. 2.1, Thm. 2.2, Prop. 2.3).** A cluster $A$ is stable iff $r>3\kappa/(\kappa+\delta(A))$ with $\delta\in\mathbb Z$, so $5/3$ never occurs for $q=2$. The single-mutant threshold is $3\kappa/(\kappa-1)=3.5$. $Y_{21}$ has $\delta\le0$ for every cluster. The star of $Y_{168}$ has $\delta=2$ and threshold $7/3$.
4. **Monte Carlo (Rem. 3.1, $K=0.1$).** At $r=2.7$: $Y_{168}$ 0.00, random 14-regular graph 0.00, thin torus $T_{12}$ 0.40.
5. **Rock–paper–scissors (Thms. 4.1–4.2).** $\frac{d}{dt}\log(x_1x_2x_3)=(a-b)(1-3\sigma)$, so there are no limit cycles and no torus. With mutation, the Hopf point is $\mu_H=(b-a)/18$ with frequency $\sqrt3(a+b)/6$.

## Chapter 5 — key clarifications

- **No "topological shielding cage".** The public-goods game is linear, so the chamber game is a pairwise game, and the flags enter only through $q+1$.
- **$r_c=1+2/(q+1)$ is not derived.** The mean-field threshold is 3, and the zero-noise thresholds are $3\kappa/(\kappa+\delta)$.
- **Catalysis is reversed.** Thick complexes behave like random regular graphs (mean field), while the thin lattice favours cooperation.
- **The replicator on the 2-simplex cannot have a torus or a limit cycle.** Oscillations require mutation (Hopf limit cycle).


## Chapter 6 — verified statements

Theorem numbers are section.item; in the book they are prefixed with 6.

1. **Degenerate affine Hecke algebra (Def. 1.1, Prop. 1.2).** The polynomial representation is checked exactly. For any $L$, the assignment $s_i\mapsto1$, $x_i\mapsto L+(i-1)c$ is an $\mathbb H_d$-module, so "realized in $\mathbb H_d$" says nothing about $L$.
2. **Iwahori–Hecke action (Thm. 1.3).** Under (L), $T_i^2=(q-1)T_i+q$ and the braid relations hold on $\mathbb C^F$, including on $Y_{21}$, which fails the flag condition.
3. **Operators (Thm. 1.4).** $\Delta_2=3+\sum T_i$, $A_1=(q+1)^{-1}\sum_t\pi_t\pi_{t+1}^t$, and $L_E=p\,T_tT_{t+2}T_t\,p^t/(q(q+1))$ blockwise. With Satake parameters, $A_1=qe_1$ and $A_2=qe_2$.
4. **No $\mathbb Z_p$-direction (Thm. 2.2).** Every open subgroup of a tower group has finite abelianization, so the CFKSV Ore set does not exist.
5. **Main theorem (Thm. 3.1, under the tower hypothesis (T)).** $\Delta^{\mathrm H}_\Lambda$ is injective and $\partial[\Delta^{\mathrm H}_\Lambda]=[X]$. Descent gives $X_U\cong\mathbb Z_p\oplus\mathrm{Jac}^{\mathrm H}(Y_U)_p$. The class number formula is $\prod_{\rho\ne1}\det(\Delta^{\mathrm H}_\rho)^{\dim\rho}=[G:U]\,|\mathrm{Jac}^{\mathrm H}(Y_U)|/|\mathrm{Jac}^{\mathrm H}(Y)|$.
6. **Check (Rem. 3.2).** $|\mathrm{Jac}^{\mathrm H}(X_3)|=147$. On $Y_{21}$ the product is $7^{18}=Q_0(1)$; $Y_{24}$ and $Y_{42}$ agree as well.

## Chapter 6 — key clarifications

- **$\mathrm{Sel}(Y/G)$ is undefined.** With the Iwasawa module $X$ of Hecke Jacobians in its place, the "main conjecture" is a theorem by construction. Its non-formal content is descent plus the class number formula.
- **The five-step scheme.** Step 1 is used in a different role (it removes $\mathbb Z_p$), step 2 is not needed, and steps 3–5 do not apply as stated.
- **Existence of $p$-adic Lie towers over the Vol. I–II complexes is open** (Chapter 1, open problem (c)). Theorem 3.1 is conditional on it.
- **Nothing in Volume III supports a cosmological reading.** The natural continuation for Volume IV is Kac–Moody buildings and their lattices.

## Chapter 6 — deepening (canonical tower; `code/v3p6_deep_verify.py`)

1. **Cartwright–Steger model (Thm. 4.1).** The triples satisfy $x+2y+4z\equiv0\pmod 7$, so $a_x\mapsto1+\alpha^x\varphi$ in $D=(\mathbb F_8(t)/\mathbb F_2(t),\mathrm{Frob},t)$ gives $u_xu_yu_z=1+t$. The map is injective, and the building is the Bruhat–Tits building of $\mathrm{PGL}_3(\mathbb F_2((s)))$, $s=1+t$ (not exotic).
2. **Canonical 2-tower (Thm. 4.2).** The closure of $\pi_1(X_3)$ in $(1+\varphi\mathcal O)/(1+t\mathbb F_2[[t]])$ is all of $P$. $P$ has $d(P)=3$ and $|P/P_m|=2^{3(m-1)-\lfloor(m-1)/3\rfloor}$. It has infinite rank, so it is **not** $p$-adic analytic. The tower is exhausting, and its level 2 is $Y_{24}$.
3. **Status (Cor. 4.3).** Hypothesis (T$_2$) holds unconditionally, so Thm. 3.1(i),(iii),(iv),(v) hold unconditionally. Hypothesis (T) fails for this tower. Parts (i′),(ii) remain open because $\mathbb Z_2[[P]]$ is not known to be Noetherian. Margulis' theorem, quoted in Rem. 4.4, suggests that no $p$-adic analytic tower exists at all.
4. **Growth (Thm. 5.1).** $\frac1n\log|\mathrm{Jac}^{\mathrm H}|\to h_H(2)=\int\log|7+a-2\bar a|\,d\nu_{\mathrm{Pl}}=2.063693$. The tower gives 1.862909, 2.041580, 2.054414, and 2.062520 at $n=6144$ (exploratory).
5. **2-part (Rem. 5.2).** $v_2|\mathrm{Jac}^{\mathrm H}|=0,18,99,289$ for $n=3,24,192,768$. No law is proved; open.
6. **2-part, levels 5–8 (Sec. 6.8; `code/v3p6_v2_growth.py`, `code/v3p6_level5_v2.py`).** Induced characters of the elementary abelian kernel $P_k/P_m$ ($2k\ge m$) give $v_2|\mathrm{Jac}^{\mathrm H}(Y_m)|=1742,11595,36937,254662$ for $n=6144,49152,196608,1572864$ (level 6 by two routes, level 5 also by the central layers and the direct elimination), and $\frac1n\log|\mathrm{Jac}^{\mathrm H}|=2.063684$ at level 8. Proved: $\liminf v_2/n\ge c_e$ (constant 1, previously 3/4), and $c_e\le(e_6+1)/n_6=0.0645$. $v_2/n$ still decreases (0.162 at level 8), but more slowly at each level; whether the limit is positive is **not decided** (it is equivalent to $c_e>0$ under a bounded block-to-kernel ratio, observed ≈1.25).
7. **Localization over $\mathbb Z_2[[P]]$ (Sec. 6.9, `chapters/ch06-s09-pro2-localization.tex`; `code/v3p6_pro2_localization.py`).** $P$ is torsion-free. Unconditionally, $\mathcal L_m=[\alpha_m]\in K_1(\mathbb Q_2A_m)$ with $A_m=\mathbb Z_2[G_m]/\mathbb Z_2N_m$ satisfies $\partial\mathcal L_m=[X_m^{\mathrm{nt}}]$, compatibly along the tower; $\Lambda=\varprojlim A_m$, $X(Y/P)=\varprojlim X_m^{\mathrm{nt}}$, and $\log_2|X_m^{\mathrm{nt}}|=\log_2|G_m|+v_2|\mathrm{Jac}^{\mathrm H}(Y_m)|$. **Not proved:** that $\mathbb Z_2[[P]]$ is a domain (Neumann's theorem needs a $p$-adic analytic group) or an Ore domain, so $\partial[\Delta^{\mathrm H}_\Lambda]=[X]$ in $K_1(Q(\Lambda))$ remains conditional.
8. **GUE window (Sec. 4.6, `chapters/ch04-s06-gue-crossover.tex`; `code/v3p4_gue_transition.py`).** On the full $W$ ($\dim W=678$), for $0<\varepsilon\le1/\dim W$, $\langle r\rangle$ is pinned at the first-order plateau $\approx0.508$ (proved limit, Prop. "first-order plateau"). It reaches the GUE value 0.600 only for $\varepsilon\gtrsim0.4$, i.e. $\approx270$–$430/\dim W$.


## Chapter 5 — nonlinear chamber games (Section 5; checks (NL1)–(NL5) in `code/v3p5_verify.py`)

1. **Irreducibility (Thm. 5.1).** $f(n)=c_1n+c_2e_2+c_3e_3$ on $\{0,1\}^3$. The chamber game is pairwise iff $c_2=c_3=0$, i.e. iff $f$ is affine. The threshold game has $(c_1,c_2,c_3)=(0,2r,-3r)$.
2. **Exact potential (Thm. 5.2).** $\Phi=\sum_\Delta(f(n_\Delta)/3-n_\Delta)$ is an exact potential for every $f$. Under logit dynamics, all-C is stochastically stable iff $f(3)>9$ (threshold game: $r>3$), independent of the geometry.
3. **Type-stratified equilibria (Prop. 5.3).** On partite complexes, "D exactly on one type class" is a Nash equilibrium iff $f(2)-f(1)\ge3\ge f(3)-f(2)$, i.e. $\frac32\le r\le3$ for the threshold game. It is metastable, and unstable under imitation.
4. **Mean-field imitation.** $\Delta(x)=\frac r3(4x-3x^2)-1$, so there is a mixed state for $r>9/4$, and all-C is stable for $r>3$.
5. **Star cages (Prop. 5.4, $q\in\{1,2\}$).** $r_{\mathrm{star}}=3(q+1)\kappa/[(q+1)(\varphi_3+q\varphi_2+q^2\varphi_1)-\varphi_2-2q\varphi_1]$. For $q=2$ this is 7/3, 63/19, 63/55 (linear, threshold, quadratic); for $q=1$ it is 9/4, 9/4, 9/11. It is verified in the building ball, on $Y_{168}$ and on $T_{12}$.
6. **Exhaustive $2^{21}$ comparison (Rem. 5.5).** Best cages on $Y_{21}$ are 3, 9/4, 0.677. Random tripartite and random 21-regular hypergraphs are equal or better.
7. **Monte Carlo (Rem. 5.6, threshold game, $K=0.1$).** At $r=2.5$: $Y_{168}$ 0.59, random ($D_2=21$) 0.60, $T_{12}$ 0.87, $Y_{21}$ 0.00.
8. **Conclusion.** Nonlinearity lowers thresholds (mean field $3\to9/4$), but no topological catalysis by thickness is found.


## Chapter 5 — chiral games, interfaces, economics (Section 6; checks (NC1)–(NC6))

1. **Potential criterion (Thm. 6.1).** The network game is an exact potential game iff the per-chamber mixed differences are symmetric, $D_{\rho\sigma}u_\rho\equiv D_{\rho\sigma}u_\sigma$. The supply-chain game (technology → manufacturing → channel, with hold-up) fails this. Its circulation is ±2/7 on $Y_{21}$, and it has a strict improvement 6-cycle, so there is no ordinal potential either.
2. **Potential mean fields converge (Prop. 6.2).** $L=\Phi/K+\sum H(x_\rho)$ is a Lyapunov function.
3. **Cycles (Thm. 6.3).** For cyclic feedback $G_\rho(x_{\rho-1})$ with negative loop sign, the equilibrium is unique and stable iff $|P|<8$. At $|P|=8$ there is a Hopf bifurcation at $\pm i\sqrt3$, and for $|P|>8$ a periodic orbit exists (Mallet-Paret–Smith, quoted). For the supply chain, $K_H=1/4$ and the period is $2\pi/\sqrt3$. The three-body variant has $K_H=0.170582$ (numerical cycle only).
4. **Networks (Rem. 6.5).** The class-density oscillation std decays with $n$: 0.262 → 0.091 → 0.023 ($Y_{24}$, $Y_{768}$, $Y_{6144}$), and 0.094 → 0.025 ($T_{12}$ → $T_{48}$). The cycle is a mean-field phenomenon.
5. **Flat interfaces (Prop. 6.6).** On thin tori the threshold is $18/(3\varphi_3+\varphi_2-\varphi_1)$: 9/5, 18/11, 3/5. Boundary cooperators have 1 of 6 "lonely" chambers on a wall versus 12 of 21 on a thick star. Balls of radius 2 and 3 in the thick building give the star values again.
6. **No λ₂ law (Rem. 6.7).** The transitions are ordered roughly, but not as a function of the gap.
7. **Economics (Prop. 6.8).** Contributing is dominant iff $\min_k(f(k+1)-f(k))\ge3$, which is never the case for a threshold technology. Logit selection depends only on $f(3)$, not on the network. So the requested "iff threshold nonlinearity + locality ⇒ globally dominant cooperation" is false.


## Chapter 5 — clocks, entrainment, anchors (Section 7; checks (NA1)–(NA5))

1. **Clock (Prop. 7.1).** The Chapter 3 flux sector locks onto the twisted ground state $m^*=-\mathrm{round}(\alpha)$. This was verified on $Y_{24}$, $Y_{168}$ ($\alpha=0.3,1.2,-0.8$ gives $m=0,2,1$) and $Y_{768}$. On the thin $T_{48}$ it gets trapped (order 0.41).
2. **Entrainment (Thm. 7.2).** The linear response is $X_m(\omega)=(i\omega+I-C)^{-1}f_m/(4K)$, and it resonates near $\sqrt3/(4K)$. Per unit ε at $K=0.4$ the amplitudes are 0.94, 0.78, 1.20 for $m=0,1,2$. The response is proportional to the clock order parameter.
3. **Coherence (Rem. 7.3).** With a synchronized clock, the density std stays 0.110 → 0.103 ($Y_{768}$ → $Y_{6144}$, $K=0.4$). With random offsets it decays, 0.037 → 0.013. The cycle is entrained, not endogenous.
4. **Frontier principle (Prop. 7.4).** Spreading from anchored balls in the building: $R=1$ gives 7/3, 63/19, 63/55; $R=2$ gives 21/10, 3, 21/20. But the unanchored radius-3 ball needs 7/3, 63/19, 63/55 again. No finite anchor removes the thick frontier threshold. An anchored wall in the thin lattice spreads at 9/5, 18/11, 3/5.
5. **Finite-noise anchors (Rem. 7.5).** With 9% committed, cooperation near the transition rises. The leader star is no better than random commitment. $Y_{168}$ still has 0 at $r=2.25$, whereas $T_{12}$ with a committed row has 0.64.


