
module Fourier2D

include("ProjectiveDynamic.jl")

export ϕ, ψ, grid, purge, H, assemble_matrix

ϕ(k, x)  = (-1)^k*exp(im*pi*x*k/2) #This is on [-2, 2]× [-2, 2]

ψ(k, h, x, y) = ϕ(k, x)*ϕ(h, y)

grid(N, L = 2) = [-2+2*L*i/N for i in 0:N-1]

purge(x, ϵ = 2^-30) = abs(x)<ϵ ? 0 : x
purge(x::Complex, ϵ = 2^-30) = purge(real(x), ϵ)+im*purge(imag(x), ϵ)

# the Henon map
H(x, y, a = 1.4, b = 0.3) = 1-a*x^2+y, b*x

unidimensional_index(i, Nx) = i>=0 ? i+1 : 2*Nx+2+i
inverse_unidim_index(i, Nx) = 1<=i<=Nx+1 ? i-1 : i-2*Nx-2
two_dimensional_index(i, j, Nx, Ny) = (unidimensional_index(j, Ny)-1)*(2*Nx+1)+unidimensional_index(i, Nx)


using FFTW
function assemble_matrix(F, Nx, Ny, FFTNx = 128, FFTNy = 128)
    # when we are rigorously approximating FFT and Fourier Transform
    # are different,
    # but this involves some tricky indexing computations...

    dx = grid(FFTNx); # this Nx-1 is due to the periodic bc
    dy = grid(FFTNy); # probably better to implement this as an iterator
    # please remark that FFTNx is half the dimension of the basis of the FFT and the same for FFTNy

    Fx = [1 for (x, y) in Base.Iterators.product(dx, dy)]
    P = plan_fft(Fx)

    N = (2*Nx+1)*(2*Ny+1) # we are taking Nx positive and negative frequencies and the 0 frequency

    M = zeros(Complex{Float64},(N, N))

    for i in 1:2*Nx+1
        for j in 1:2*Ny+1

            l = inverse_unidim_index(i, Nx) # the index in the form [0, ..., Nx, -Nx, ..., -1]
            m = inverse_unidim_index(j, Ny) # the index in the form [0, ..., Nx, -Nx, ..., -1]

            twodtransform = P*[ψ(l, m, F(x,y)...) for (x, y) in Base.Iterators.product(dx, dy)]
            # we need to take a view of the transform corresponding to the frequencies we are interested in
            #@info twodtransform[1:Nx+1, 1:Ny+1]
            a = twodtransform[1:Nx+1, 1:Ny+1] # [0,...Nx] × [0,..., Ny]
            b = twodtransform[FFTNx-Nx+1:FFTNx, 1:Ny+1] # [-Nx,...,-1] × [0,..., Ny]
            c = twodtransform[1:Nx+1, FFTNy-Ny+1:FFTNy] # [0,...,Nx] × [-Ny,..., -1]
            d = twodtransform[FFTNx-Nx+1:FFTNx, FFTNy-Ny+1:FFTNy] # [-Nx,...,-1] × [-Ny,..., -1]

            v = hcat(vcat(a, b), vcat(c, d))
            #@info two_dimensional_index(l, m, Nx, Ny)
            MM[two_dimensional_index(l, m, Nx, Ny), :] = reshape(v, (2*Nx+1)*(2*Ny+1))

        end
    end
    # we take the adjoint since we are computing the adjoint operator
    return M'/(FFTNx*FFTNy)
end

end # module
