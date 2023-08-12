export AcousticPropagatorParams, solve
using GLMakie

"""
    Ricker(epp::Union{ElasticPropagatorParams, AcousticPropagatorParams}, 
    a::Union{PyObject, <:Real}, 
    shift::Union{PyObject, <:Real}, 
    amp::Union{PyObject, <:Real}=1.0)

Returns a Ricker wavelet (a tensor). 
- `epp`: a `ElasticPropagatorParams` or an `AcousticPropagatorParams`
- `a`: Width parameter
- `shift`: Center of the Ricker wavelet
- `amp`: Amplitude of the Ricker wavelet

```math
f(x) = \\mathrm{amp}A (1 - x^2/a^2) exp(-x^2/2 a^2)
```
where 
```math
A = 2/sqrt(3a)pi^1/4
```
"""
function Ricker(epp, 
        a, 
        shift, 
        amp=1.0)
    NT, T = epp.NSTEP, epp.NSTEP*epp.DELTAT
    A = @. 2 / (sqrt(3 * a) * (pi^0.25))
    wsq = @. a^2
    vec =  collect(1:NT) .-shift
    xsq = @. vec^2
    mod = @. (1 - xsq / wsq)
    gauss = @. exp(-xsq / (2 * wsq))
    total = @. amp * A * mod * gauss
    return total
end

struct AcousticPropagatorParams{DIM, AT<:AbstractArray{Float64,DIM}}
    # number of grids along x,y axis and time steps
    NX::Int
    NY::Int 
    NSTEP::Int

    # size of grid cell and time step
    DELTAX::Float64
    DELTAY::Float64
    DELTAT::Float64

    # Auxilliary Data
    Σx::AT
    Σy::AT
end

function AcousticPropagatorParams(; nx::Int, ny::Int, nstep::Int,
        dx::Float64, dy::Float64, dt::Float64,
        Rcoef::Float64=0.001, # Relative reflection coefficient
        vp_ref::Float64=1000.0,
        npoints_PML::Int=12,
        USE_PML_XMAX::Bool = true,
        USE_PML_XMIN::Bool = true,
        USE_PML_YMAX::Bool = true,
        USE_PML_YMIN::Bool = true)
    # computing damping coefficient
    Lx = npoints_PML * dx
    Ly = npoints_PML * dy
    damping_x = vp_ref/Lx*log(1/Rcoef)
    damping_y = vp_ref/Ly*log(1/Rcoef)

    Σx, Σy = zeros(nx+2, ny+2), zeros(nx+2, ny+2)
    for i = 1:nx+2
        for j = 1:ny+2
            Σx[i,j] = pml_helper((i-1)*dx, nx, dx,
                damping_x, npoints_PML,
                USE_PML_XMIN, USE_PML_XMAX)
            Σy[i,j] = pml_helper((j-1)*dy, ny, dy,
                damping_y, npoints_PML,
                USE_PML_YMIN, USE_PML_YMAX)
        end
    end
    return AcousticPropagatorParams(nx, ny, nstep, dx, dy, dt, Σx, Σy)
end

function pml_helper(x::Float64, nx::Int, dx::Float64, ξx::Float64, npoints_PML::Int,
        USE_PML_XMIN, USE_PML_XMAX)
    Lx = npoints_PML * dx
    out = 0.0
    if x<Lx && USE_PML_XMIN 
        d = abs(Lx-x)
        out = ξx * (d/Lx - sin(2π*d/Lx)/(2π))
    elseif x>dx*(nx+1)-Lx && USE_PML_XMAX
        d = abs(x-(dx*(nx+1)-Lx))
        out = ξx * (d/Lx - sin(2π*d/Lx)/(2π))
    end
    return out
end

function one_step!(param::AcousticPropagatorParams, u, w, wold, φ, ψ, σ, τ, c)
    Δt = param.DELTAT
    hx, hy = param.DELTAX, param.DELTAY
 
    @inbounds for j=2:param.NY+1, i=2:param.NX+1
        uij = (2 - σ[i,j]*τ[i,j]*Δt^2 - 2*Δt^2/hx^2 * c[i,j] - 2*Δt^2/hy^2 * c[i,j]) * w[i,j] +
            c[i,j] * (Δt/hx)^2  *  (w[i+1,j]+w[i-1,j]) +
            c[i,j] * (Δt/hy)^2  *  (w[i,j+1]+w[i,j-1]) +
            (Δt^2/(2hx))*(φ[i+1,j]-φ[i-1,j]) +
            (Δt^2/(2hy))*(ψ[i,j+1]-ψ[i,j-1]) -
            (1 - (σ[i,j]+τ[i,j])*Δt/2) * wold[i,j] 
        u[i,j] = uij / (1 + (σ[i,j]+τ[i,j])/2*Δt)
    end
    @inbounds for j=2:param.NY+1, i=2:param.NX+1
        φ[i,j] = (1. -Δt*σ[i,j]) * φ[i,j] + Δt * c[i,j] * (τ[i,j] -σ[i,j])/2hx *  
            (u[i+1,j]-u[i-1,j])
        ψ[i,j] = (1-Δt*τ[i,j]) * ψ[i,j] + Δt * c[i,j] * (σ[i,j] -τ[i,j])/2hy * 
            (u[i,j+1]-u[i,j-1])
    end
end

function solve(param::AcousticPropagatorParams, src, 
            srcv::Vector{T}, c::Matrix{T}) where T
    history = Matrix{T}[]
    tupre = zeros(param.NX+2, param.NY+2)
    tu = zeros(param.NX+2, param.NY+2)
    tφ = zeros(param.NX+2, param.NY+2)
    tψ = zeros(param.NX+2, param.NY+2)

    for i = 3:param.NSTEP+1
        tu_ = zeros(param.NX+2, param.NY+2)
        one_step!(param, tu_, tu, tupre, tφ, tψ, param.Σx, param.Σy, c)
        tu, tupre = tu_, tu
        tu[src...] += srcv[i-2]*param.DELTAT^2
        push!(history, copy(tu))
    end
    history
end

nx = ny = 100
nstep = 100

# the landscape
param = AcousticPropagatorParams(; nx, ny,
        Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep)
c = 1000*ones(param.NX+2, param.NY+2)

# the source
src = (param.NX÷2, param.NY÷2)
srcv = Ricker(param, 100.0, 500.0)

u = ones(param.NX+2, param.NY+2)
φ = zeros(param.NX+2, param.NY+2)
ψ = zeros(param.NX+2, param.NY+2)
w, wold = copy(u), copy(u)
one_step!(param, u, w, wold, φ, ψ, param.Σx, param.Σy, c)
gu, gw, gwold, gφ, gψ, gΣx, gΣy, gc = rand(size(u)...), zero(w), zero(wold), zero(φ), zero(ψ), zero(param.Σx), zero(param.Σy), zero(c)
using Enzyme
Enzyme.autodiff(Enzyme.Reverse, one_step!,
    Const(param),
    Duplicated(u, gu),
    Duplicated(w, gw),
    Duplicated(wold, gwold),
    Duplicated(φ, gφ),
    Duplicated(ψ, gψ),
    Duplicated(param.Σx, gΣx),
    Duplicated(param.Σy, gΣy),
    Duplicated(c, gc)
)

function demo_simulate(; nx = 100, ny = 100, nstep=3000)
    param = AcousticPropagatorParams(; nx=nx, ny=ny,
        Rcoef=0.2, dx=20.0, dy=20.0, dt=0.05, nstep)

    # the landscape
    c = 1000*ones(param.NX+2, param.NY+2)
    c[1:param.NX ÷ 3, :] .= 500

    # the source
    src = (param.NX÷2, param.NY÷2)
    srcv = Ricker(param, 100.0, 500.0)

    return solve(param, src, srcv, c)
end

struct SeismicState{MT}
    upre::MT
    u::MT
    φ::MT
    ψ::MT
    step::Base.RefValue{Int}
end

using TreeverseAlgorithm
treeverse_actions(3000, 10)

"""
    treeverse_solve(s0; param, src, srcv, c, δ=20, logger=TreeverseLog())

* `s0` is the initial state,
"""
function treeverse_solve(s0, gnf; param, src, srcv, c, δ=20, logger=TreeverseLog())
    f = x->treeverse_step(x, param, src, srcv, c)
    res = []
    function gf(x, g)
        if g === nothing
            y = f(x)
            push!(res, y)
            g = gnf(y)
        end
        treeverse_grad(x, g[1], param, src, srcv, g[2], c, g[3])
    end
    g = treeverse(f, gf,
        copy(s0); δ=δ, N=param.NSTEP-1, f_inplace=false, logger=logger)
    return res[], g
end

function treeverse_grad(x, g, param, src, srcv, gsrcv, c, gc)
    y = treeverse_step(x, param, src, srcv, c)
    gt = SeismicState([GVar(getfield(y, field), getfield(g, field)) for field in fieldnames(SeismicState)[1:end-1]]..., Ref(y.step[]))
    _, gs, _, _, gv, gc2 = (~bennett_step!)(gt, GVar(x), param, src, GVar(srcv, gsrcv), GVar(c, gc))
    (grad(gs), grad(gv), grad(gc2))
end

nx = ny = 300
result = demo_simulate(; nx, ny)

function animate(result)
    nx, ny = size(result[1])
    x = LinRange(0, 1, nx)
    y = LinRange(0, 1, ny)
    matrix = result[1]
    fig, ax, hm = heatmap(x, y, matrix; colorrange=(-0.003, 0.003))
    Colorbar(fig[:,end+1], hm)
    display(fig)

    for c in result
        hm[3] = c
        yield()
        sleep(0.003)
    end
end
using Enzyme
# one_step!(param::AcousticPropagatorParams, u, w, wold, φ, ψ, σ, τ, c)
Enzyme.autodiff(Reverse, one_step!, )

