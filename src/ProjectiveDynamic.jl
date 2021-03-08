using LinearAlgebra

generate_basis(n) = [[cos(x); sin(x)] for x in range(0, length = n, step = 2*pi/n)]

function discretize(B, A; discretization = :isaia)
    M = zeros(length(B), length(B))
    for (i, v) in enumerate(B)
        w = A*v
        w/=norm(w, 2)
        z = norm.([v-w for v in B], 2)
        p = sortperm(z)
        if discretization == :isaia
            if z[p[1]]!=0
                M[p[1], i] = z[p[2]]/(z[p[1]]+z[p[2]])
                M[p[2], i] = z[p[1]]/(z[p[1]]+z[p[2]])
            else
                M[p[1], i] = 1
            end
        elseif discretization == :alex
            M[p[1], i] = 1
        end
    end
    return M
end

function annealed(DF, dx, dy, N, discretization)
    M = zeros(N, N)
    B = generate_basis(N)
    k = length(dx)
    h = length(dy)
    for x in dx
        for y in dy
            M+=discretize(B, DF(x, y), discretization = discretization)/(k*h)
        end
    end
    return M
end

using ForwardDiff, LinearAlgebra
function numerical_iteration(F, n, ξ)
    orbit = []
    orbit_proj = []
    x = rand(Float64, 2)
    push!(orbit, x)
    Q = I
    J(x) = ForwardDiff.jacobian(F, x)
    Q, R = qr(J(x)*Q)
    for _ in 1:n
        x = F(x)+ξ*(rand(Float64, 2).-0.5)
        push!(orbit, x)
        Q, R = qr(J(x)*Q)
    end
    return Q
end

function circleShape(h, k, r)
    θ = LinRange(0, 2*π, 500)
    h.+r*sin.(θ), k.+r*cos.(θ)
end

using Plots
function visualize(v)
    B = generate_basis(length(v))
    m = maximum(v)
    pl = plot(circleShape(0, 0, m), label = "radius $m")
    for (i, x) in enumerate(v)
        v  = x*B[i]
        pl = plot!([0; v[1]], [0; v[2]], label = "", color = :red, linewidth = 2)
    end
    return pl
end

using Statistics, LinearAlgebra
function W(v, w)
    @assert length(v)==length(w)
    N = length(w)
    V = cumsum(v)
    W = cumsum(w)
    μ = median(V-W)
    return 2*pi*norm(V-W.-μ, 1)/N
end
