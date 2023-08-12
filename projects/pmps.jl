# periodic MPS

using Yao

# MPO for Heisenberg Model: https://itensor.org/docs.cgi?page=tutorials/MPO
J = 1
heisenberg_mpo = zeros(2, 2, 5, 5)
heisenberg_mpo[:, :, 1, 1] = mat(Float64, ConstGate.I2)
heisenberg_mpo[:, :, 2, 1] = mat(Float64, ConstGate.Pu)
heisenberg_mpo[:, :, 3, 1] = mat(Float64, ConstGate.Pd)
heisenberg_mpo[:, :, 4, 1] = mat(Float64, ConstGate.Z)

heisenberg_mpo[:, :, 5, 2] = mat(Float64, 2J * ConstGate.Pd)
heisenberg_mpo[:, :, 5, 3] = mat(Float64, 2J * ConstGate.Pu)
heisenberg_mpo[:, :, 5, 4] = mat(Float64, J * ConstGate.Z)
heisenberg_mpo[:, :, 5, 5] = mat(Float64, ConstGate.I2)

n = 2
h = EasyBuild.heisenberg(n; periodic=true)
matH = mat(h)

using OMEinsum
physical_ups = 1:n
physical_downs = -(1:n)
virtual = 10000 .+ (1:n)
ixs = [[physical_ups[i], physical_downs[i], virtual[i], virtual[mod1(i+1, n)]] for i=1:n]
iy = [physical_ups..., physical_downs...]
mpo_code = DynamicEinCode(ixs, iy)
optcode = optimize_code(mpo_code, uniformsize(mpo_code, 2), TreeSA())
contraction_complexity(optcode, uniformsize(optcode, 2))
mpoH = reshape(optcode([heisenberg_mpo for i=1:10]...), 1<<n, 1<<n);
mpoH - matH

# create an periodic-MPS
D = 10
n = 10
unit = rand(D, 2, D)
physical = 1:n
physical2 = replace(physical, 1=>n+1, 2=>n+2)
virtual = 10000 .+ (1:n)
virtual2 = 20000 .+ (1:n)
twobody = reshape(mat(kron(X, X) + kron(Y, Y) + kron(Z, Z)), 2, 2, 2, 2)
ixs = [
    [[virtual[i], physical[i], virtual[mod1(i+1,n)]] for i=1:n]...,  # ket
    [[virtual2[i], physical2[i], virtual2[mod1(i+1,n)]] for i=1:n]...,     # bra
    [physical[1], physical[2], physical2[1], physical2[2]]
]
energy = n * DynamicEinCode(ixs, iy)([unit for i=1:n]..., [conj.(unit) for i=1:n], twobody)

using Zygote

Zygote.gradient()