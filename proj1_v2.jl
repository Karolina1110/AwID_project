using LinearAlgebra
include("FEM.jl")
using PyCall
using PyPlot
using BenchmarkTools

matplotlib.use("TkAgg") 

using IterativeSolvers

A, b = femproblem(10, 10)

A = 1e23.*A
b = 1e23.*b

#starting point
x0 = zeros(size(A,1))
x = zeros(size(A,1))

#max ammount of Iterations
max_m = 500
m = 0

@time xg = gmres(A,b)
#xg = gmres(A, b)
#print(xg, '\n')

# r = b - A*x0

# beta = norm(r,2)
# beta2 = norm(r,2)

# q = zeros((size(A,1), max_m))
# q[:,1] = r./beta

# h = zeros((max_m+1, max_m))

# e = zeros(max_m+1)
# e[1] = 1

# err = Float64[]

# while (m != max_m)    
#     global m, q, x, r, beta2, err
#     m = m+1

#     y = A*q[:,m]    

#     for j in 1:m
#         h[j, m] = (q[:,j]' * y)
#         y = y - h[j, m] * q[:,j] 
#     end


#     h[m + 1, m] = norm(y,2)

#     if (h[m + 1, m] != 0) && (m != max_m)
#         q[:,m + 1] = y / h[m + 1, m]
#     end


#     c = h[1:m,1:m]\(beta*e[1:m])

#     x = q[:,1:m]*c + x0
#     r = b - A*x
#     beta2 = norm(r,2)

#     push!(err, beta2)

#     #accuracy
#     beta2 < 1e-32 && break
  
# end

# ------------------------------------------------------------ GAUSS ----------------------------------------------------------
function rspec(T) # spectral radius
    λ = eigvals(T)
    ρ = 0.0
    for i=1:length(λ)
        if abs(λ[i]) > ρ
            ρ = abs(λ[i])
        end
    end
    if(ρ < 1.0)
        println("spectral radius: ", ρ, " < 1.0")
        return true
    else
        return false
    end
end

function gaussseidel!(x, A, b, error, history)

    max_iter = size(A,1)
    sum0 = 0.0
    sum1 = 0.0
    sum2 = 0.0


   for iter=1:max_iter
    # first element
        for p=2:size(A,1)
            sum0 += A[1,p]*x[p]
        end
        x[1] = (b[1] - sum0)/A[1,1]
    # rest of elements
        for i=2:size(A, 1)
            for j=1:i-1
                sum1 += A[i,j]*x[j]
            end
            for k=i+1:size(A, 1)
                sum2 += A[i,k]*x[k]
            end
            x[i] = (b[i] - sum1 - sum2)/A[i,i]
            sum0 = 0.0
            sum1 = 0.0
            sum2 = 0.0
        end
        push!(history, norm(A * x - b))
        if history[iter]<error
            println("iterations = ", iter)
            break
        end
    end
    #display(plot(history, title = "Gauss-Seidel", ylabel = "Error", xlabel = "Iterations", yaxis=:log))
    return x
end



function GMRES!(x0, A, b, error, hist)
    
    x = zeros(size(A,1))

    #max ammount of Iterations
    max_m = size(A, 2)
    m = 0


    r = b - A*x0

    beta = norm(r,2)
    #beta2 = norm(r,2)

    q = [r./beta zeros((size(A,1), max_m-1))]
    #q = zeros((size(A,1), max_m))
    #q[:,1] = r./beta

    h = zeros((max_m+1, max_m))

    e = [1.0; zeros(max_m)]
    #e = zeros(max_m+1)
    #e[1] = 1

   

    while (m != max_m)    
        #global m, q, x, r, beta2, err
        m = m+1
        #y = similar(q[:,m])
        y = A*q[:,m] 
        #mul!(y,A,q[:,m])   

        for j in 1:m
            h[j, m] = (q[:,j]' * y)
            y = @. y - h[j, m] * q[:,j] 
        end


        h[m + 1, m] = norm(y,2)

        if (h[m + 1, m] != 0) && (m != max_m)
            q[:,m + 1] = y / h[m + 1, m]
        end


        c = h[1:m,1:m]\(beta*e[1:m])

        x = q[:,1:m]*c + x0
        r = b - A*x
        #beta2 = norm(r,2)

        push!(hist, norm(r,2))

        #accuracy
        norm(r,2) < error && break
    
    end
    println("iteracje: ", m)
    return x

end

x0 = zeros(size(A,1))
hist = Float64[]
history = Float64[]
@time x = GMRES!(x0, A, b, 1e-6, hist)
#x = GMRES!(x0, A, b, 1e-6, hist)
@time gaussseidel!(x0, A, b, 1e-6, history)

xe = A\b
#print(x, '\n')
#print(xe, '\n')
#print(xg.-x, '\n')

plt.plot(1:size(hist,1), hist)
plt.plot(1:size(history,1), history)
plt.title("% error GMRES method")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.show()


#plt.scatter(1:1:size(A,1), xe, label = "Exact value")
plt.scatter(1:1:size(A,1), xg, label = "GMRES method from Julia")
plt.scatter(1:1:size(A,1), x, label = "implemented GMRES method")
plt.title("Comparison GMRES methods to original value")
plt.xlabel("Vector elements")
plt.ylabel("Value")
plt.legend()
plt.show()




