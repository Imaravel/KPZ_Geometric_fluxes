# Numerical support for the article: which geometric fluxes produce KPZ, and why.
#
# Every run integrates the EXACT graph equation of the master theorem,
#
#     ∂_t h = Q·v(θ) + Q·F[κ] + Q·η ,    Q = sqrt(1+h_x²),  θ = arctan h_x,
#
# changing ONLY the flux F and the growth law v(θ). The kinematic factor Q, the
# grid, the time step and the noise draws are identical across configurations
# (common random numbers), so a difference in the measured exponents is
# attributable to the flux, and an ABSENCE of difference is evidence for the
# factorization theorem.
#
# Physical parameters follow Chapter 5 of the thesis (ν=1, λ=v₀=3, D₀=0.08) so
# that ℓ_c = ν³/(λ²D₀) ≈ 1.39 and t_c = ν⁵/(λ⁴D₀²) ≈ 1.93 sit far below the fit
# window.
#
# NUMERICS. Two choices matter and both were forced by observed blow-ups.
#
#  (1) Fourth-order fluxes need ∂_s²κ in DIVERGENCE form on a staggered grid.
#      Chaining two centred differences, d1(d1(·)/Q)/Q, decouples the even and
#      odd sublattices and leaves the Nyquist (checkerboard) mode completely
#      undamped; white noise then pumps it without bound and every fourth-order
#      run diverges within a few hundred steps at any noise level. `divgrad`
#      below uses midpoint fluxes, reduces to the compact 5-point stencil when
#      Q≡1, damps k=π correctly, and is exactly conservative (the divergence
#      telescopes), which is what makes Corollary 3.4 testable on the lattice.
#
#  (2) The linear part of each flux is inverted exactly in Fourier space and the
#      multiplicative noise is placed INSIDE the implicit solve, as in the
#      spectral scheme of Chapter 6. Without this the strong-noise threshold of
#      Experiment 4 measures the integrator rather than the equation; with it,
#      the threshold is stable under halving Δt (verified in Experiment 4).
#
# Run:  julia --project=. scripts/article_fluxes.jl

using Random, Statistics, Printf, DelimitedFiles, FFTW, Plots, LaTeXStrings
gr(); default(fontfamily="Computer Modern", framestyle=:box, grid=false, dpi=150)
const C_KPZ, C_EXA, C_EW = "#0072B2", "#D55E00", "#666666"
const CURVES = Dict{String,Tuple{Vector{Float64},Vector{Float64}}}()

const N     = 512
const DX    = 1.0
const DT    = 0.005
const D0    = 0.08
const NU    = 1.0        # ν = σ/β, the tension-to-friction ratio
const VARK  = 1.0        # bending rigidity ϰ
const V0    = 3.0
const TMAX  = 300.0
const TFIT  = 15.0
const NREAL = 12

d1(u) = (circshift(u, -1) .- circshift(u, 1)) ./ (2DX)
d2(u) = (circshift(u, -1) .- 2u .+ circshift(u, 1)) ./ DX^2

"∂ₓ(a·∂ₓu) with `a` averaged onto the half-points i+1/2. See note (1) above."
function divgrad(u, a)
    ah = (a .+ circshift(a, -1)) ./ 2
    fl = ah .* (circshift(u, -1) .- u) ./ DX
    return (fl .- circshift(fl, 1)) ./ DX
end
d4(u) = divgrad(d2(u), ones(length(u)))          # compact ∂ₓ⁴, symbol S₂²

# Lattice symbol of -∂ₓ² for the centred stencil; ∂ₓ⁴ then has symbol S2.^2.
const KS  = 2π .* (0:N÷2) ./ (N*DX)
const S2  = 2 .* (1 .- cos.(KS .* DX)) ./ DX^2

const PFLOW_P = Dict(:pflow_15 => 1.5, :pflow_075 => 0.75)

"Normal velocity F[κ] of the geometric flux, in graph variables (exact)."
function flux(kind, h, Q)
    κ = d2(h) ./ Q.^3
    if kind === :curvature
        return NU .* κ
    elseif haskey(PFLOW_P, kind)
        p = PFLOW_P[kind]
        return NU .* κ .* (κ.^2 .+ 1e-10).^((p - 1)/2)
    elseif kind === :elastica
        return -VARK .* (divgrad(κ, 1 ./ Q) ./ Q .+ 0.5 .* κ.^3)
    elseif kind === :helfrich
        return NU .* κ .- VARK .* (divgrad(κ, 1 ./ Q) ./ Q .+ 0.5 .* κ.^3)
    elseif kind === :surfdiff
        return -NU .* divgrad(κ, 1 ./ Q) ./ Q
    else
        error("unknown flux $kind")
    end
end

"Linear part of Q·F, in real space (finite differences) and as a Fourier symbol."
function linear_parts(kind)
    if kind === :curvature
        return (h -> NU .* d2(h)), (-NU .* S2)
    elseif kind === :elastica
        return (h -> -VARK .* d4(h)), (-VARK .* S2.^2)
    elseif kind === :helfrich
        return (h -> NU .* d2(h) .- VARK .* d4(h)), (-NU .* S2 .- VARK .* S2.^2)
    elseif kind === :surfdiff
        return (h -> -NU .* d4(h)), (-NU .* S2.^2)
    else                                   # p-flows have no linear part
        return (h -> zero(h)), zeros(length(S2))
    end
end

"Growth law v(θ) = v₀(1 + A cos2θ), using cos2θ = (1-h_x²)/(1+h_x²)."
vgrowth(v0, A, hx) = v0 .* (1 .+ A .* (1 .- hx.^2) ./ (1 .+ hx.^2))
lambda_pred(v0, A) = v0 * (1 - 3A)

"""
    run_exact(kind; ...) -> (ts, W, maxslope, blewup, meanvelocity)

Semi-implicit spectral integration of the exact equation. `m` imposes a mean
tilt through helical boundary conditions h = m·x + φ with φ periodic; φ is the
integrated field, so h_x = m + φ_x and h_xx = φ_xx.
"""
function run_exact(kind; v0=V0, A=0.0, D=D0, Tmax=TMAX, m=0.0, seed=1,
                   dt=DT, vel=false, cnoise=false)
    Random.seed!(seed)
    Lreal, Lsym = linear_parts(kind)
    denom = 1 .- dt .* Lsym
    nst = round(Int, Tmax/dt)
    s   = sqrt(2D*dt/DX)
    φ   = zeros(N)
    ts = Float64[]; W = Float64[]; mx = Float64[]
    vsum = 0.0; vcnt = 0; tvel0 = 0.5*Tmax
    for k in 1:nst
        hx = m .+ d1(φ)
        Q  = sqrt.(1 .+ hx.^2)
        rhs = Q .* (vgrowth(v0, A, hx) .+ flux(kind, φ, Q))     # full exact RHS
        if !all(isfinite, rhs) || maximum(abs, hx) > 1e3
            return ts, W, mx, true, NaN
        end
        vel && k*dt >= tvel0 && (vsum += mean(rhs); vcnt += 1)
        nonlin = rhs .- Lreal(φ)                                 # explicit remainder
        # A conserved flux must be driven by a conserved noise, η = -∂ₓξ. By
        # Corollary 3.4 it then enters the height equation with NO factor Q; the
        # midpoint difference below sums to zero exactly, so ∫h dx is conserved.
        ξ = (s/DX) .* randn(N)
        incr = dt .* nonlin .+ (cnoise ? -(ξ .- circshift(ξ, 1)) : (s .* Q) .* randn(N))
        φ      = irfft(rfft(φ .+ incr) ./ denom, N)
        φ    .-= mean(φ)
        t = k*dt
        push!(ts, t); push!(W, std(φ)); push!(mx, maximum(abs, m .+ d1(φ)))
    end
    return ts, W, mx, false, (vcnt > 0 ? vsum/vcnt : NaN)
end

"Ensemble width, pooling W² over realizations under common random numbers."
function ensemble_W(kind; v0=V0, A=0.0, D=D0, Tmax=TMAX, nreal=NREAL, base=4000, cnoise=false)
    W2 = Float64[]; tg = Float64[]; mxmax = 0.0; nbad = 0
    for r in 1:nreal
        ts, W, mx, bad, _ = run_exact(kind; v0=v0, A=A, D=D, Tmax=Tmax, seed=base+r, cnoise=cnoise)
        if bad || isempty(W); nbad += 1; continue; end
        isempty(W2) && (W2 = zeros(length(W)); tg = ts)
        W2 .+= W.^2
        mxmax = max(mxmax, maximum(mx))
    end
    nok = nreal - nbad
    nok == 0 && return Float64[], Float64[], NaN, nbad
    return sqrt.(W2 ./ nok), tg, mxmax, nbad
end

"Least-squares slope of log W vs log t on [tlo,thi], log-spaced subsampling."
function beta_fit(ts, W; tlo=TFIT, thi=TMAX, nlog=60)
    (isempty(ts) || last(ts) < tlo) && return NaN
    thi = min(thi, last(ts))
    grid = 10 .^ range(log10(tlo), log10(thi), length=nlog)
    idx  = unique(argmin(abs.(ts .- g)) for g in grid)
    length(idx) < 5 && return NaN
    lt, lW = log.(ts[idx]), log.(W[idx])
    sum((lt .- mean(lt)) .* (lW .- mean(lW))) / sum((lt .- mean(lt)).^2)
end

pbeta(p) = (2p - 1)/(5p - 1)

# ─────────────────────────────────────────────────────────────────────────────
println("="^86)
println("EXPERIMENT 1 — β for several geometric fluxes, with and without a growth term")
println("  v₀ = 3 turns the kinematic nonlinearity on;  v₀ = 0 leaves the pure gradient flow")
println("="^86)
@printf("%-14s %-5s %-6s %-8s %-8s %-8s %s\n",
        "flux", "v₀", "D", "β", "β_pred", "max|hx|", "note")

exp1 = [
 (:curvature,  V0,  D0,   1/3,         "KPZ"),
 (:curvature,  0.0, D0,   1/4,         "Edwards-Wilkinson"),
 (:helfrich,   V0,  D0,   1/3,         "KPZ (tension+bending)"),
 (:helfrich,   0.0, D0,   1/4,         "EW (tension wins at large l)"),
 (:elastica,   V0,  0.02, 1/3,         "KPZ (pure bending)"),
 (:elastica,   0.0, 0.02, 3/8,         "linear MBE (superrough)"),
 (:pflow_15,   V0,  D0,   1/3,         "KPZ (p=1.5 > p_c)"),
 (:pflow_15,   0.0, D0,   pbeta(1.5),  "p-linear, p=1.5"),
 (:pflow_075,  V0,  D0,   1/3,         "p=0.75 < p_c=4/5"),
 (:pflow_075,  0.0, D0,   pbeta(0.75), "p-linear, p=0.75"),
 (:surfdiff,   0.0, D0,   1/8,         "surface diffusion + conserved noise"),
]
const CNOISE = Set(["surfdiff"])          # conserved flux ⇒ conserved noise

rows = Any[]
for (kind, v0, D, bpred, note) in exp1
    W, ts, mx, nbad = ensemble_W(kind; v0=v0, D=D, cnoise=String(kind) in CNOISE)
    b = beta_fit(ts, W)
    CURVES["$(kind)_v$(v0)"] = (ts, W)
    @printf("%-14s %-5.1f %-6.3f %-8.3f %-8.3f %-8.2f %s%s\n",
            String(kind), v0, D, b, bpred, mx, note, nbad > 0 ? "  [$nbad diverged]" : "")
    push!(rows, [String(kind), v0, D, b, bpred, mx, nbad, note])
end

# ─────────────────────────────────────────────────────────────────────────────
println()
println("="^86)
println("EXPERIMENT 2 — anisotropy switches KPZ off:  v(θ)=v₀(1+A cos2θ), λ = v₀(1-3A)")
println("="^86)
@printf("%-12s %-7s %-9s %-8s %-8s %s\n", "flux", "A", "λ_pred", "β", "β_pred", "class")
rows2 = Any[]
for kind in (:curvature, :helfrich)
    for A in (0.0, 1/6, 1/3, 1/2)
        λp = lambda_pred(V0, A)
        W, ts, mx, nbad = ensemble_W(kind; v0=V0, A=A)
        b = beta_fit(ts, W)
        CURVES["$(kind)_A$(round(A,digits=3))"] = (ts, W)
        isEW = abs(λp) < 1e-9
        @printf("%-12s %-7.3f %-9.3f %-8.3f %-8.3f %s\n",
                String(kind), A, λp, b, isEW ? 0.25 : 1/3, isEW ? "EW" : "KPZ")
        push!(rows2, [String(kind), A, λp, b, isEW ? 0.25 : 1/3, mx, nbad])
    end
end

# ─────────────────────────────────────────────────────────────────────────────
println()
println("="^86)
println("EXPERIMENT 3 — tilt-velocity V(m); fit V(m)-V(0) = a₂m² + a₄m⁴,  λ = 2a₂")
println("  Prediction: λ = v(0)+v''(0) and a₄ = -v₀/8 for A=0, both flux-independent")
println("="^86)
ms = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
rows3 = Any[]
for (kind, A) in ((:curvature, 0.0), (:helfrich, 0.0), (:pflow_15, 0.0), (:curvature, 1/3))
    Vs = Float64[]
    for m in ms
        _, _, _, bad, v = run_exact(kind; v0=V0, A=A, m=m, Tmax=60.0, seed=77, vel=true)
        push!(Vs, bad ? NaN : v)
    end
    dV = Vs .- Vs[1]
    CURVES["tilt_$(kind)_A$(round(A,digits=3))"] = (copy(ms), dV)
    c  = hcat(ms.^2, ms.^4) \ dV
    @printf("%-12s A=%-6.3f a₂=%-8.4f a₄=%-9.4f  λ_meas=%-7.3f λ_pred=%-7.3f\n",
            String(kind), A, c[1], c[2], 2c[1], lambda_pred(V0, A))
    push!(rows3, [String(kind), A, c[1], c[2], 2c[1], lambda_pred(V0, A)])
end

# ─────────────────────────────────────────────────────────────────────────────
println()
println("="^86)
println("EXPERIMENT 4 — strong-noise threshold of the graph description")
println("  Δt and Δt/4 are both reported: agreement means the threshold is physical,")
println("  not a property of the integrator.")
println("="^86)
@printf("%-12s %-7s %-11s %-11s %-9s\n", "flux", "D", "survived Δt", "survived Δt/4", "max|hx|")
rows4 = Any[]
for kind in (:curvature, :helfrich)
    for D in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)
        function sweep(dt)
            nok = 0; mxm = 0.0; tdiv = Float64[]
            for r in 1:4
                ts, _, mx, bad, _ = run_exact(kind; v0=V0, D=D, Tmax=25.0, seed=500+r, dt=dt)
                isempty(mx) || (mxm = max(mxm, maximum(mx)))
                bad ? push!(tdiv, isempty(ts) ? 0.0 : last(ts)) : (nok += 1)
            end
            nok, mxm, isempty(tdiv) ? NaN : mean(tdiv)
        end
        n1, mx1, td1 = sweep(DT)
        n2, _,   _   = sweep(DT/4)
        @printf("%-12s %-7.2f %-11s %-11s %-9.2f %s\n",
                String(kind), D, "$n1/4", "$n2/4", mx1,
                isnan(td1) ? "" : @sprintf("t_div≈%.1f", td1))
        push!(rows4, [String(kind), D, n1, n2, mx1, td1])
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Figure: four panels, one per experiment.
# ─────────────────────────────────────────────────────────────────────────────
const FLOWNAME = Dict(:curvature => "curvature", :helfrich => "tension + bending",
                      :pflow_15 => L"$p$-flow, $p=1.5$")
gref(ts, W, t0, b) = (W[argmin(abs.(ts .- t0))] .* (ts ./ t0).^b)

pa = plot(xscale=:log10, yscale=:log10, xlabel=L"t", ylabel=L"W(t)",
          title="(a)", titleloc=:left, titlefontsize=10, legend=:bottomright, legendfontsize=6)
for (key, lab, c, ls) in (("curvature_v3.0", "curvature, "*L"v_0=3", C_KPZ, :solid),
                          ("curvature_v0.0", "curvature, "*L"v_0=0", C_KPZ, :dash),
                          ("helfrich_v3.0",  "tension + bending, "*L"v_0=3", C_EXA, :solid),
                          ("helfrich_v0.0",  "tension + bending, "*L"v_0=0", C_EXA, :dash))
    haskey(CURVES, key) || continue
    ts, W = CURVES[key]; isempty(ts) && continue
    m = ts .>= 1.0
    plot!(pa, ts[m], W[m], c=c, ls=ls, lw=1.7, label=lab)
end
let (ts, W) = CURVES["curvature_v3.0"]
    m = (ts .>= TFIT) .& (ts .<= TMAX)
    plot!(pa, ts[m], gref(ts, W, TFIT, 1/3)[m], c=:black, ls=:dot, lw=1.2, label=L"t^{1/3}")
    plot!(pa, ts[m], 0.72 .* gref(ts, W, TFIT, 1/4)[m], c=:gray, ls=:dot, lw=1.2, label=L"t^{1/4}")
end

As = [0.0, 1/6, 1/3, 1/2]
pb = plot(xlabel=L"anisotropy $A$", ylabel=L"\beta",
          title="(b)", titleloc=:left, titlefontsize=10, legend=:topright, legendfontsize=6)
for (kind, c) in ((:curvature, C_KPZ), (:helfrich, C_EXA))
    bs = [beta_fit(CURVES["$(kind)_A$(round(A,digits=3))"]...) for A in As]
    plot!(pb, As, bs, c=c, lw=1.8, m=:circle, ms=4, label=FLOWNAME[kind])
    ew = beta_fit(CURVES["$(kind)_v0.0"]...)
    hline!(pb, [ew], c=c, ls=:dash, lw=1.1, label=FLOWNAME[kind]*", "*L"v_0=0")
end
vline!(pb, [1/3], c=:black, ls=:dot, lw=1.4, label=L"A=1/3\ (\lambda=0)")

pc = plot(xlabel=L"tilt $m$", ylabel=L"V(m)-V(0)",
          title="(c)", titleloc=:left, titlefontsize=10, legend=:topleft, legendfontsize=6)
for (kind, c, mk) in ((:curvature, C_KPZ, :circle), (:helfrich, C_EXA, :diamond),
                      (:pflow_15, "#009E73", :utriangle))
    k = "tilt_$(kind)_A0.0"; haskey(CURVES, k) || continue
    ms_, dV = CURVES[k]
    scatter!(pc, ms_, dV, c=c, m=mk, ms=5, label=FLOWNAME[kind])
end
let mm = range(0, 0.9, length=100)
    plot!(pc, mm, V0 .* (sqrt.(1 .+ mm.^2) .- 1), c=:black, ls=:dot, lw=1.5,
          label=L"v_0(\sqrt{1+m^2}-1)")
    plot!(pc, mm, 0.5V0 .* mm.^2, c=:gray, ls=:dash, lw=1.2, label=L"\frac{\lambda}{2}m^2")
end

Ds = unique(Float64[r[2] for r in rows4])
pd = plot(xlabel=L"noise strength $D$", ylabel="fraction of runs surviving",
          title="(d)", titleloc=:left, titlefontsize=10, legend=:bottomleft, legendfontsize=6, ylims=(-0.05,1.15))
for (kind, c, mk) in (("curvature", C_KPZ, :circle), ("helfrich", C_EXA, :diamond))
    fr = [Float64(r[3])/4 for r in rows4 if r[1] == kind]
    plot!(pd, Ds, fr, c=c, lw=1.8, m=mk, ms=5, label=FLOWNAME[Symbol(kind)])
end

fig = plot(pa, pb, pc, pd, layout=(2,2), size=(980,760), left_margin=4Plots.mm,
           bottom_margin=4Plots.mm, dpi=300)
figdir = joinpath(@__DIR__, "..", "figures"); isdir(figdir) || mkpath(figdir)
savefig(fig, joinpath(figdir, "fig_fluxes.png"))
println("wrote figures/fig_fluxes.png")

out = joinpath(@__DIR__, "..", "data"); isdir(out) || mkpath(out)
dumpcsv(f, hdr, rs) = writedlm(joinpath(out, f),
                               vcat(permutedims(hdr), reduce(vcat, permutedims.(rs))), ',')
dumpcsv("article_exp1.csv", ["flux","v0","D","beta","beta_pred","maxslope","ndiverged","note"], rows)
dumpcsv("article_exp2.csv", ["flux","A","lambda_pred","beta","beta_pred","maxslope","ndiverged"], rows2)
dumpcsv("article_exp3.csv", ["flux","A","a2","a4","lambda_meas","lambda_pred"], rows3)
dumpcsv("article_exp4.csv", ["flux","D","nsurv_dt","nsurv_dt4","maxslope","t_div"], rows4)
println("\nwrote data/article_exp{1,2,3,4}.csv")
