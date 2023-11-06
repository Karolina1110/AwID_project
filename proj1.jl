using LinearAlgebra
include("FEM.jl")
using PyPlot
using IterativeSolvers


A, b = femproblem(10, 10)
#print(A,'\n')
#print(b,'\n')

#A = [4. -1. -0.2 2.;
#    -1 5. 0. -2.;
#    0.2 1. 10. -1.;
#    0. -2. -1. 4.]

#b = [30., 0., -10., 5.]

#Jacobi

#starting point
xj = zeros(size(A,1))

n = range(1,size(A,1))

g = Float64[]
h = zeros(size(A,1), size(A,2))

for i in n
    push!(g, b[i]/A[i,i])
    for j in n
        if i != j
            h[i,j] = -A[i,j]/A[i,i]
        else
            h[i,j] = 0
        end
    end
end


#if norm(h)>=1
#    print("\nerror: problem with convergence!")
#    exit()
#end

#error
#rw = Float64[]
#for i in n
#    temp = 0
#    for j in n
#        temp += A[i,j]*xj[j]
#    end
#    temp -= b[i] 
#    push!(rw, abs(temp))
#end
#sumaric error
#rs = sum(rw)

r = b .- A*xj

p=0

dj = Float64[]

while (norm(r,2) >1e-16 && p != 1000)
    global p, xj, r, dj
    p = p+1

    xj = g + h * xj

    #for i in n
    #    temp = 0
    #    for j in n
    #        temp += A[i,j]*xj[j]
    #    end
    #    temp -= b[i] 
    #    rw[i] = abs(temp)
    #end
    #rs = sum(rw)
    
    r = b .- A*xj
    push!(dj, norm(r,2))
end

print(A\b,"\n\n")
#print(dj,'\n')
print(xj,"\n\n")

plt.plot(1:1:size(dj,1), dj)
plt.title("% error Jacobi method")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.show()

ej = xj - A\b
#print(ej, '\n')

#Gauss-Seidel
xg = zeros(size(A,1))

#for i in n
#    temp = 0
#    for j in n
#        temp += A[i,j]*xg[j]
#    end
#    temp -= b[i] 
#    rw[i] = abs(temp)
#end
#rs = sum(rw)

r = b .- A*xg


p=0
dg = Float64[]

while (norm(r,2) > 1e-16 && p != 1000)
    global p, xg, r, rw, dg
    p = p+1
 

    for i in n
        temp1 = 0
        temp2 = 0


        if i != 1
            for j in 1:(i-1)
                temp1 += h[i,j]*xg[j]
            end
        end

        for j in i:size(A,1)
            temp2 += h[i,j]*xg[j]
        end
        xg[i] = g[i] + temp1 + temp2
    end

    #for i in n
    #    temp = 0
    #    for j in n
    #        temp += A[i,j]*xg[j]
    #    end
    #    temp -= b[i] 
    #    rw[i] = abs(temp)
    #end
    #rs = sum(rw)
   

    r = b .- A*xg
    push!(dg, norm(r,2))
end

print(xg,"\n\n")
eg = xg - A\b
#print(eg,'\n')

xje = jacobi(A,b)
print(xje, "\n\n")

xge = gauss_seidel(A, b)
print(xge, "\n\n")

plt.plot(1:size(dg,1), dg)
plt.title("% error Gauss method")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.show()

plt.plot(1:1:size(dj,1), dj, label = "Jacobi method")
plt.plot(1:1:size(dg,1), dg, label = "Gauss method")
plt.title("% error Jacobi and Gauss methods")
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.legend()
plt.show()

plt.scatter(1:1:size(A,1), xj, label = "Jacobi method")
plt.scatter(1:1:size(A,1), xg, label = "Gauss method")
plt.scatter(1:1:size(A,1), A\b, label = "Exact value")
plt.title("Comparison Jacobi and Gauss methods to original value")
plt.xlabel("Vector elements")
plt.ylabel("Value")
plt.legend()
plt.show()