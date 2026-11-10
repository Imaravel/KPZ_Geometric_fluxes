# KPZ_Geometric_fluxes

Code and data for the article

> G. E. Ardila Carrillo and J. C. Cortissoz Iriarte,
> *Why is the Kardar–Parisi–Zhang equation ubiquitous?*

The scripts integrate the exact graph equation of a moving interface,

    ∂_t h = Q·v(θ) + Q·F[κ] + Q·η ,    Q = sqrt(1 + h_x²),  θ = arctan h_x,

directly, without a small-slope expansion, for a range of geometric fluxes `F`
(curvature flow, tension + bending, p-flows, elastica, polyharmonic flows). Only
the flux and the growth law `v(θ) = v₀(1 + A cos 2θ)` change between runs. The
grid, time step and noise realizations are shared across all configurations, so
any difference between two runs comes from the flux alone.

## Contents

| Path | Description |
|---|---|
| `scripts/article_fluxes.jl` | Experiments 1–4: growth exponent β with and without growth, β against the kinetic anisotropy A, the tilt–velocity relation, and graph survival under strong noise. Produces `figures/fig_fluxes.png` and `data/article_exp{1,2,3,4}.csv`. |
| `scripts/article_polyflows.jl` | Higher-order flows (elastica, surface diffusion, sixth-order polyharmonic and energy-ladder flows). Produces `data/article_poly{A,B,C,D}.csv` and the raw tilt–velocity points `data/article_polyC_tilt.csv`. |
| `data/` | The data shown in the figure and in the table of growth exponents. |
| `figures/fig_fluxes.png` | The figure of the article. |

### Mapping to the article

- **Fig. 2 (panels a–d)**: `article_exp1.csv` (a), `article_exp2.csv` (b),
  `article_exp3.csv` and `article_polyC_tilt.csv` (c), `article_exp4.csv` (d).
- **λ from the tilt–velocity relation** (quoted in the text): `article_exp3.csv`
  (curvature, tension + bending) and `article_polyC.csv` (elastica, sixth order).
- **Table of effective exponents β**: second-order rows (curvature,
  tension + bending, p-flows) from `article_exp1.csv`; higher-order rows
  (elastica, sixth order, surface diffusion) from `article_polyA.csv` and
  `article_polyB.csv`.

## Parameters

Periodic lattice of N = 512 sites, Δx = 1, Δt = 5×10⁻³, ν = ϰ = 1, v₀ = 3,
noise strength D = 0.08 unless stated otherwise, 12 realizations, β fitted to
W ∼ t^β over 15 ≤ t ≤ 300. The linear part of each flux is solved implicitly in
Fourier space with the multiplicative (Itô) noise inside the implicit step;
fourth-order operators are discretized in divergence form on a staggered grid.
See the header comments of each script for details.

## Running

Requires Julia ≥ 1.10.

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. scripts/article_polyflows.jl
julia --project=. scripts/article_fluxes.jl
```

Run `article_polyflows.jl` first: panel (c) of the figure reads the
fourth- and sixth-order points from `data/article_polyC_tilt.csv`.

Each script overwrites the corresponding files in `data/` (and `figures/`).
Random seeds are fixed, so the runs are reproducible.

## License

MIT, see `LICENSE`.
