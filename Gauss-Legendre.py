# Numerical investigation of properties of quadrature rules
# Example: Gauss-Legendre quadrature
# Example: composite Gauss-Legendre quadrature

import numpy as np
import matplotlib.pyplot as plt

# The nodes are defined as the roots of the n-th degree Legendre polynomial
# Calculate nodes and weights for N nodes
N = 2
xk, wk = np.polynomial.legendre.leggauss(N)

Dop = False
convergence = True

if Dop:

 # Define functions for degree n polynomial and integral
    def p(x,n):
        return x**n

    def p_int(x,n):
        return x**(n+1)/(n+1)
    # Return the approximate of a degree n polynomial using N nodes
    def Legendre(N,n):
        xk, wk = np.polynomial.legendre.leggauss(N)
        # approx = 0
        # for i in range(N):
        #     approx += p(xk[i])*wk[i]
        # approx = p(xk) @ wk
        # approx = np.dot(p(xk), wk)
        approx = np.sum(p(xk,n)*wk)
        return approx

    # Return the error of the Gauss-Legendre quadrature routine
    def Legendre_error(N,n):
        I_exact = p_int(1,n) - p_int(-1,n)
        error = abs(I_exact - Legendre(N,n))
        return error

    # Test degree of precision
    N = 3
    tol = 10e-13
    n = 0
    while n <=1000:
        if Legendre_error(N,n) > tol:
            print(f'The degree of presision for {N} nodes is {n-1}')
            break
        n += 1

if convergence:
    # composite quadrature
    # create random polynomial of degree 2N +2
    p_coefs = 2*np.random.random(2*N +2) -1
    p_coefs_int =  np.polyint(p_coefs)
    M = 10

    def composite(M,p_coefs,p_coefs_int):
        # h = (b-a)/M
        h = 2/M
        bounds = np.linspace(-1,1,M+1)
        I_approx = 0
        for i in range(M):
            # scale x co-ords
            yk = ((bounds[i+1] - bounds[i])/2) * (xk + 1) +bounds[i]
            I_approx += ((bounds[i+1] - bounds[i])/2) * np.sum(np.polyval(p_coefs,yk)*wk)

        I_exact = np.polyval(p_coefs_int,1) - np.polyval(p_coefs_int,-1)
        err = abs(I_exact - I_approx)
        return err

    # Test the rate of convergence: Compute the error for different values of h (which is determined by M)

    M_vals = np.logspace(1,7,7, base =2, dtype = int)
    err = []
    h_vals = 2/M_vals

    for M in M_vals:
        err.append(composite(M,p_coefs,p_coefs_int))
    
    # Plot the results

    fig, ax = plt.subplots(1,2, figsize = (12,4))
    ax[0].plot(h_vals, err, 'rx')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set(title = 'log axes', xlabel = 'h', ylabel = 'Error')
    ax[1].plot(np.log(h_vals), np.log(err), 'rx')
    ax[1].set(title = 'log values', xlabel = 'log(h)', ylabel = 'log(Error)')
    plt.show()

    # Determine the rate of convergence
    line_coefs = np.polyfit(np.log(h_vals), np.log(err),1)
    r = line_coefs[0]
    print(r)


