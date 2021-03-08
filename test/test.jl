# a small test
N = 1024
@assert [unidimensional_index(i, N) for i in 0:N] == [j for j in 1:N+1]
@assert [unidimensional_index(i, N) for i in -N:-1] == [j for j in N+2:2*N+1]

for i in 1:2049
    @assert unidimensional_index(inverse_unidim_index(i, N), N) == i
end

two_dimensional_index(i, j, Nx, Ny) = (unidimensional_index(j, Ny)-1)*(2*Nx+1)+unidimensional_index(i, Nx)
@assert two_dimensional_index(1,1, 2, 2) == 7
two_dimensional_index(0, 0, 2, 2)

N = 2

v = [0:N;-N:-1]

[(i, j, two_dimensional_index(i, j, N, N)) for (i, j) in Base.Iterators.product(v, v)]
