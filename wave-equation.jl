using LinearAlgebra

stiffness_element(lx, ly) =
[lx/ly  1  lx/ly  1
   1  ly/lx  1  ly/lx
 lx/ly  1  lx/ly  1
   1  ly/lx  1  ly/lx]

mass_element(lx, ly) = (lx * ly) / 6.0 *
[ 2. 0 -1  0
  0  2  0 -1
 -1  0  2  0
  0 -1  0  2]

function assemble!(S, R, T, el2ed, el2edd, dof, lx, ly, nelem, ndof; εr=1, μr=1, ε0=8.854e-12, μ0=4e-7π)
  # ASSUMPTION: Waveguide is homogenous
  # Assemble stiffness and mass matrices
  ε = εr*ε0
  for ielem = 1:nelem # Assemble by elements
    Se = stiffness_element(lx, ly)
    Te = mass_element(lx, ly)
    
    for jedge = 1:4
      dj = el2edd[ielem, jedge]
      jj = dof[el2ed[ielem, jedge]]
      if jj == 0
        continue
      end
      
      for kedge = 1:4
        dk = el2edd[ielem, kedge]
        kk = dof[el2ed[ielem, kedge]]
        if kk == 0
          continue
        end
  
        S[jj, kk] = S[jj, kk] + dj * dk * (1/μr) * Se[jedge, kedge]
        T[jj, kk] = T[jj, kk] + dj * dk * (μ0*ε) * Te[jedge, kedge]
      end
    end
  end
  return nothing
end

function lhs(S, T, R, Δt)
  A = (+0.25Δt^2 * S +  T + 0.5Δt * R)
end

function rhs(S, T, R, Δt, ep, epp)
  b = (-0.25Δt^2 * S -  T + 0.5Δt * R) * epp +
      (-0.50Δt^2 * S + 2T) * ep
end

function solve!(x, A, b)
   println("Here goes your implementation...")
   nothing
end