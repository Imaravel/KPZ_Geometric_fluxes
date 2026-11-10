# The Cortissoz-Reyes / Cortissoz-Galindo-Murcia flow families, tested for KPZ.
#
# Two ladders through the (E, mu) plane of the master equation, both reaching
# arbitrary even order:
#
#   METRIC LADDER (fixed E = sigma*L, varying mobility):  mu = (-d_s^2)^{-m}
#       V_n = (-1)^m * sigma * d_s^{2m} kappa           <-- the POLYHARMONIC flows
#       m=0 curve shortening (z=2), m=1 surface diffusion (z=4), m=2 (z=6), ...
#       All are surface divergences for m>=1 => exactly conservative in the graph
#       (Cor. 3.4), so they must be driven by a conserved noise.
#
#   ENERGY LADDER (fixed mu = 1, varying energy):  E_j = int (d_s^j kappa)^2 ds
#       j=0 elastica (z=4);  j=1 gives Cortissoz-Reyes' Family II member
#       V = kappa_ssss + kappa^2 kappa_ss - (1/2) kappa kappa_s^2   (z=6),
#       the L^2 gradient flow of int (kappa_s)^2 ds.  NOT conserved.
#
# Same order, different physics: the two order-6 flows differ in conservation,
# in noise structure and in class. That is the (E,mu) point made twice over.
#
# Run:  julia --project=. scripts/article_polyflows.jl

using Random, Statistics, Printf, DelimitedFiles, FFTW

const N     = 512
const DX    = 1.0
const DT    = 0.005
const D0    = 0.08
const AMP   = 1.0        # sigma (metric ladder) or the energy prefactor
const V0    = 3.0
const TMAX  = 300.0
const TFIT  = 15.0
const NREAL = 12

d1(u) = (circshift(u, -1) .- circshift(u, 1)) ./ (2DX)
d2(u) = (circshift(u, -1) .- 2u .+ circshift(u, 1)) ./ DX^2

"dx( a * dx u ) with a on half-points: compact, damps Nyquist, telescopes."
function divgrad(u, a)
    ah = (a .+ circshift(a, -1)) ./ 2
    fl = ah .* (circshift(u, -1) .- u) ./ DX
    return (fl .- circshift(fl, 1)) ./ DX
end

"Intrinsic second arc-length derivative, in divergence form."
ds2(u, Q) = divgrad(u, 1 ./ Q) ./ Q

"Apply ds2 n times."
function ds2n(u, Q, n)
    v = u
    for _ in 1:n; v = ds2(v, Q); end
    return v
end

const KS = 2π .* (0:N÷2) ./ (N*DX)
const S2 = 2 .* (1 .- cos.(KS .* DX)) ./ DX^2      # symbol of -d_x^2

# ---- flux definitions -------------------------------------------------------
# :polyN with N = m  =>  V_n = (-1)^m * AMP * d_s^{2m} kappa
# :fam2         =>  V_n = AMP * (kappa_ssss + kappa^2 kappa_ss - 1/2 kappa kappa_s^2)
# :curvature    =>  V_n = AMP * kappa        (m = 0 of the metric ladder)
# :elastica     =>  V_n = -AMP (d_s^2 kappa + 1/2 kappa^3)   (j = 0 of the energy ladder)

const POLY_M = Dict(:poly1 => 1, :poly2 => 2, :poly3 => 3)

function flux(kind, h, Q)
    κ = d2(h) ./ Q.^3
    if kind === :curvature
        return AMP .* κ
    elseif haskey(POLY_M, kind)
        m = POLY_M[kind]
        return ((-1)^m * AMP) .* ds2n(κ, Q, m)
    elseif kind === :elastica
        return -AMP .* (ds2(κ, Q) .+ 0.5 .* κ.^3)
    elseif kind === :fam2
        κs  = d1(κ) ./ Q
        κss = ds2(κ, Q)
        return AMP .* (ds2n(κ, Q, 2) .+ κ.^2 .* κss .- 0.5 .* κ .* κs.^2)
    else
        error("unknown flux $kind")
    end
end

"Linear part of Q*F in real space, and its lattice Fourier symbol."
function linear_parts(kind)
    if kind === :curvature
        return (h -> AMP .* d2(h)), (-AMP .* S2)
    elseif haskey(POLY_M, kind)
        m = POLY_M[kind]
        f = function (h)
            v = d2(h); for _ in 1:m; v = d2(v); end
            return ((-1)^m * AMP) .* v
        end
        return f, (-AMP .* S2 .^ (m+1))
    elseif kind === :elastica
        return (h -> -AMP .* d2(d2(h))), (-AMP .* S2.^2)
    elseif kind === :fam2
        return (h -> AMP .* d2(d2(d2(h)))), (-AMP .* S2.^3)
    end
end

"Conserved fluxes (surface divergences) must carry a conserved noise."
isconserved(kind) = haskey(POLY_M, kind)

vgrowth(v0, A, hx) = v0 .* (1 .+ A .* (1 .- hx.^2) ./ (1 .+ hx.^2))

function run_exact(kind; v0=V0, A=0.0, D=D0, Tmax=TMAX, m=0.0, seed=1, dt=DT, vel=false)
    Random.seed!(seed)
    Lreal, Lsym = linear_parts(kind)
    denom = 1 .- dt .* Lsym
    cn    = isconserved(kind)
    nst = round(Int, Tmax/dt); s = sqrt(2D*dt/DX)
    φ = zeros(N)
    ts = Float64[]; W = Float64[]; mx = Float64[]
    vsum = 0.0; vcnt = 0; tvel0 = 0.5*Tmax
    for k in 1:nst
        hx = m .+ d1(φ); Q = sqrt.(1 .+ hx.^2)
        rhs = Q .* (vgrowth(v0, A, hx) .+ flux(kind, φ, Q))
        if !all(isfinite, rhs) || maximum(abs, hx) > 1e3
            return ts, W, mx, true, NaN
        end
        vel && k*dt >= tvel0 && (vsum += mean(rhs); vcnt += 1)
        ξ = (s/DX) .* randn(N)
        incr = dt .* (rhs .- Lreal(φ)) .+ (cn ? -(ξ .- circshift(ξ,1)) : (s .* Q) .* randn(N))
        φ = irfft(rfft(φ .+ incr) ./ denom, N)
        φ .-= mean(φ)
        push!(ts, k*dt); push!(W, std(φ)); push!(mx, maximum(abs, m .+ d1(φ)))
    end
    return ts, W, mx, false, (vcnt > 0 ? vsum/vcnt : NaN)
end

function ensemble_W(kind; v0=V0, A=0.0, D=D0, Tmax=TMAX, nreal=NREAL, base=4000)
    W2 = Float64[]; tg = Float64[]; mxm = 0.0; nbad = 0
    for r in 1:nreal
        ts, W, mx, bad, _ = run_exact(kind; v0=v0, A=A, D=D, Tmax=Tmax, seed=base+r)
        if bad || isempty(W); nbad += 1; continue; end
        isempty(W2) && (W2 = zeros(length(W)); tg = ts)
        W2 .+= W.^2; mxm = max(mxm, maximum(mx))
    end
    nok = nreal - nbad
    nok == 0 && return Float64[], Float64[], NaN, nbad
    return sqrt.(W2 ./ nok), tg, mxm, nbad
end

function beta_fit(ts, W; tlo=TFIT, thi=TMAX, nlog=60)
    (isempty(ts) || last(ts) < tlo) && return NaN
    thi = min(thi, last(ts))
    grid = 10 .^ range(log10(tlo), log10(thi), length=nlog)
    idx = unique(argmin(abs.(ts .- g)) for g in grid)
    length(idx) < 5 && return NaN
    lt, lW = log.(ts[idx]), log.(W[idx])
    sum((lt .- mean(lt)) .* (lW .- mean(lW))) / sum((lt .- mean(lt)).^2)
end

# β predictions.  order n = 2m+2.  Conserved noise: 2α+1 = n-2 ; else 2α+1 = n.
beta_cons(m)   = (2m - 1)/(4m + 4)      # H^{-m} flow of length + conserved noise
beta_noncons(n) = (n - 1)/(2n)          # order-n linear operator, non-conserved noise

println("="^92)
println("EXPERIMENT A - the METRIC ladder: H^{-m} gradient flows of length (polyharmonic)")
println("  V_n = (-1)^m d_s^{2m} kappa.  Conserved for m>=1, so driven by conserved noise.")
println("="^92)
@printf("%-12s %-6s %-6s %-8s %-8s %-8s %s\n","flux","order","D","beta","beta_pred","max|hx|","note")
rowsA = Any[]
for (kind, ord, bp, D, note) in (
        (:curvature, 2, beta_noncons(2), D0,  "curve shortening + non-conserved noise (EW)"),
        (:poly1,     4, beta_cons(1),    D0,  "surface diffusion, H^{-1}"),
        (:poly2,     6, beta_cons(2),    D0,  "H^{-2} flow of length"),
        (:poly2,     6, beta_cons(2),    0.02,"H^{-2}, weaker noise"),
        (:poly3,     8, beta_cons(3),    0.02,"H^{-3} flow of length"))
    W, ts, mx, nbad = ensemble_W(kind; v0=0.0, D=D)
    b = beta_fit(ts, W)
    @printf("%-12s %-6d %-6.3f %-8.3f %-8.3f %-8.2f %s%s\n",
            String(kind), ord, D, b, bp, mx, note, nbad>0 ? "  [$nbad diverged]" : "")
    push!(rowsA, [String(kind), ord, D, b, bp, mx, nbad, note])
end

println()
println("="^92)
println("EXPERIMENT B - the ENERGY ladder: L^2 gradient flows of int (d_s^j kappa)^2 ds")
println("  j=0 elastica (order 4);  j=1 Family II, V = k_ssss + k^2 k_ss - k k_s^2/2 (order 6)")
println("="^92)
@printf("%-12s %-6s %-6s %-6s %-8s %-8s %-8s %s\n","flux","order","v0","D","beta","beta_pred","max|hx|","note")
rowsB = Any[]
for (kind, ord, v0, D, bp, note) in (
        (:elastica, 4, 0.0, 0.02, beta_noncons(4), "linear MBE"),
        (:elastica, 4, V0,  0.02, 1/3,             "KPZ"),
        (:fam2,     6, 0.0, 0.02, beta_noncons(6), "linear, order 6"),
        (:fam2,     6, 0.0, 0.005,beta_noncons(6), "linear, order 6, weak noise"),
        (:fam2,     6, V0,  0.02, 1/3,             "KPZ from a 6th-order flux"),
        (:fam2,     6, V0,  0.005,1/3,             "KPZ, weak noise"))
    W, ts, mx, nbad = ensemble_W(kind; v0=v0, D=D)
    b = beta_fit(ts, W)
    @printf("%-12s %-6d %-6.1f %-6.3f %-8.3f %-8.3f %-8.2f %s%s\n",
            String(kind), ord, v0, D, b, bp, mx, note, nbad>0 ? "  [$nbad diverged]" : "")
    push!(rowsB, [String(kind), ord, v0, D, b, bp, mx, nbad, note])
end

println()
println("="^92)
println("EXPERIMENT C - is lambda still kinematic at 6th order?  tilt-velocity, lambda = 2a2")
println("="^92)
ms = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
rowsC = Any[]
rowsCt = Any[]                       # raw V(m) points, plotted in Fig. 2(c)
for (kind, D) in ((:curvature, D0), (:elastica, 0.02), (:fam2, 0.02), (:fam2, 0.005))
    Vs = Float64[]
    for mm in ms
        _, _, _, bad, v = run_exact(kind; v0=V0, D=D, m=mm, Tmax=60.0, seed=77, vel=true)
        push!(Vs, bad ? NaN : v)
    end
    dV = Vs .- Vs[1]
    if any(isnan, dV)
        @printf("%-12s D=%-6.3f  diverged\n", String(kind), D); continue
    end
    c = hcat(ms.^2, ms.^4) \ dV
    @printf("%-12s D=%-6.3f a2=%-8.4f a4=%-9.4f  lambda_meas=%-7.3f lambda_pred=%.3f\n",
            String(kind), D, c[1], c[2], 2c[1], V0)
    push!(rowsC, [String(kind), D, c[1], c[2], 2c[1], V0])
    for (mm, v) in zip(ms, dV); push!(rowsCt, [String(kind), D, mm, v]); end
end

println()
println("="^92)
println("EXPERIMENT D - graph-form ceiling vs order of the flux")
println("  Blatt (2010) and Giga-Ito (1999) prove higher-order flows lose convexity and")
println("  embeddedness.  If that is what breaks the graph, the ceiling should FALL with order.")
println("="^92)
@printf("%-12s %-6s %-8s %-10s %-10s\n","flux","order","D","survived","t_div")
rowsD = Any[]
for (kind, ord) in ((:curvature,2), (:elastica,4), (:fam2,6))
    for D in (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
        nok = 0; td = Float64[]
        for r in 1:4
            ts, _, _, bad, _ = run_exact(kind; v0=V0, D=D, Tmax=25.0, seed=500+r)
            bad ? push!(td, isempty(ts) ? 0.0 : last(ts)) : (nok += 1)
        end
        @printf("%-12s %-6d %-8.2f %-10s %s\n", String(kind), ord, D, "$nok/4",
                isempty(td) ? "" : @sprintf("%.1f", mean(td)))
        push!(rowsD, [String(kind), ord, D, nok, isempty(td) ? NaN : mean(td)])
    end
end

out = joinpath(@__DIR__, "..", "data"); isdir(out) || mkpath(out)
dump(f,hdr,rs) = writedlm(joinpath(out,f), vcat(permutedims(hdr), reduce(vcat, permutedims.(rs))), ',')
dump("article_polyA.csv", ["flux","order","D","beta","beta_pred","maxslope","ndiv","note"], rowsA)
dump("article_polyB.csv", ["flux","order","v0","D","beta","beta_pred","maxslope","ndiv","note"], rowsB)
isempty(rowsC) || dump("article_polyC.csv", ["flux","D","a2","a4","lambda_meas","lambda_pred"], rowsC)
isempty(rowsCt) || dump("article_polyC_tilt.csv", ["flux","D","m","dV"], rowsCt)
dump("article_polyD.csv", ["flux","order","D","nsurv","t_div"], rowsD)
println("\nwrote data/article_poly{A,B,C,D}.csv and data/article_polyC_tilt.csv")
