'''
Exercise: 3.1, but only the Newton's method part and no line search is required.
However, have your code plot in 2D the places where the evaluations occurred so
it will be easy to see the path the search algorithm took.

Program the Newton algorithms using the backtracking line
search, Algorithm 3.1. Use them to minimize the Rosenbrock function (2.22). Set the initial
step length α0 = 1 and print the step length used by each method at each iteration. First try
the initial point x0 = (1.2, 1.2)T and then the more difficult starting point x0 = (−1.2, 1)T
'''

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math


def gradient(x1,x2):
    #[x1,x2]=x0
    partial_x1 = 400*x1**3 - 400*x1*x2 + 2*x1 - 2
    partial_x2 = 200*x2 - 200*x1**2
    return np.array([partial_x1, partial_x2])

def Hessian_inv(x1,x2):
 #   [x1,x2]=x0
    second_deri_x1 = 1200*x1**2 - 400*x2 +2
    second_deri_x1x2 = -400*x1
    second_deri_x2x1 = -400*x1
    second_deri_x2 = 200
    Hession = np.array([[second_deri_x1,second_deri_x1x2],[second_deri_x2x1,second_deri_x2]])
    return np.linalg.inv(Hession)

def optimize_step(x1,x2):
    H_inv = Hessian_inv(x1,x2)
    p = np.dot(H_inv, gradient(x1,x2))
    x0 = [x1,x2]
    return x0 - p
#print (gradient(1.2,1.2))   # where x0=(1.2,1.2)
#print (Hessian_inv(1.2,1.2))
#print (optimize_step(1.2,1.2))

def func_z(x1,x2):
       #[x1,x2]=x0
       return (100*(x2-x1**2)**2)+(1-x1)**2
       
def optimizer(x1,x2):
    #[x1,x2]=x0   
    n_iters=0
    optimized = 1
    x_new = []
    x0 = [x1,x2]
    while (optimized > 1*np.e**(-10)) and (n_iters<100):
       x_new.append(x0)
       optimized = func_z(x1,x2)
       x0 = optimize_step(x1,x2)
       x1 = x0[0]
       x2 = x0[1]
       n_iters = n_iters+1
    x_new.append(x0)
    return np.array(x_new)


#print (optimizer(-1.2,1))




## plot 

def func_z(x1,x2):
       return (100*(x2-x1**2)**2)+(1-x1)**2

x1vals = np.arange(-1,1,0.01)
x2vals = np.arange(-1,1,0.01)
X1,X2 = np.meshgrid(x1vals, x2vals)
Z = func_z(X1,X2)


plt.imshow(Z, extent=(X1.min(), X1.max(), X2.max(), X2.min()),
           interpolation='nearest', cmap=cm.gist_rainbow)

plt.contour(X1, X2, Z, colors = 'k', linestyles = 'solid')

def points_visited(x1,x2):
    OP = optimizer(x1,x2)
    pvT = OP.T
    return pvT

def plot_optimizer(x1,x2):
    x1vals = np.arange(-2,2,0.01)
    x2vals = np.arange(-4,2,0.01)
    X1,X2 = np.meshgrid(x1vals, x2vals)
    Z = func_z(X1,X2)
    plt.imshow(Z, extent=(X1.min(), X1.max(), X2.max(), X2.min()), cmap=cm.gist_rainbow)
    plt.contour(X1, X2, Z, colors = 'k', linestyles = 'solid')
    pvt = points_visited(x1,x2)
    xs = pvt[0]
    ys = pvt[1]
    plt.scatter(xs, ys)
    return plt.show()


plot_optimizer(1.2, 1.2)
#plot_optimizer(-1.2, 1)





