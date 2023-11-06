include("wave-equation.jl")

function quadmesh(a, b, Nx, Ny)
    NUM_EDGES = 2(Nx*Ny) + Nx + Ny
    NUM_ELEMS = Nx * Ny
    
    el2edd = repeat([+1 +1 -1 -1], NUM_ELEMS)
    el2ed = zeros(Int64, NUM_ELEMS, 4)
    for jj = 1:Ny
        for ii = 1:Nx
            kk = (jj-1)Nx + ii
            el2ed[kk, :]  .= [ii, ii+Nx+1, ii+Nx+1+Nx, ii+Nx]
            el2ed[kk, :] .+= (jj-1) * (Nx + Nx + 1)
        end
    end

    return el2ed, el2edd, NUM_EDGES
end

function femproblem(n, m)
    @assert n * m < 40*40 "Are you sure? Try smaller problem first!"
    
    # parameters
    Δt = 0.01e-9
    Lx = 2.00
    Ly = 2.00
    lx = Lx / n
    ly = Ly / m
    el2ed, el2edd, nedge = quadmesh(Lx, Ly, n, m);
    
    # degrees of freedom
    DOF_NONE = 0
    DOF_PEC  = 1

    h = [  1+(2n+1)i: n+0+(2n+1)i for i=0:m]
    v = [n+1+(2n+1)i:2n+1+(2n+1)i for i=0:m-1]

    Γ = zeros(Int64, nedge)
    Γ[first(h)] .= DOF_PEC
    Γ[last(h)]  .= DOF_PEC
    for i=1:m
         Γ[first(v[i])] = DOF_PEC
         Γ[last(v[i])] = DOF_PEC
    end

    dof = collect(1:nedge)
    free = Γ .!= DOF_PEC

    # assemble finite element matrices
    S = zeros(nedge, nedge)
    T = zeros(nedge, nedge)
    R = zeros(nedge, nedge)
    assemble!(S, T, R, el2ed, el2edd, dof, lx, ly, n*m, nedge)
    
    # construct the problem left hand side
    A = lhs(S[free, free], T[free, free], R[free, free], Δt);
    
    # calculate eigensolution and use it as a starting point
    k², v = eigen(Array(S[free, free]), Array(T[free, free]))
    
    e = zeros(nedge)
    ep = copy(e)
    epp = copy(e)
    ep[free] .= epp[free] .= v[:, 1+(n-1)*(m-1)]
    
    # construct the problem right hand side
    b = rhs(S[free, free], T[free, free], R[free, free], Δt, ep[free], epp[free])
    
    return A, b
end

A, b = femproblem(10, 10)
e = A \ b