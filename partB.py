'''
Part B: For 2.1, 2.2, and 2.4 above, use Python to plot the functions 
and plot points for where you are evaluating these functions.

'''

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pylab


fig = plt.figure()

ax = plt.axes(projection='3d')

## 2.1

def func_z(x,y):
	return (100*(y-x**2)**2)+(1-x)**2

xvals = np.arange(-1,1,0.01)
yvals = np.arange(-1,1,0.01)
X,Y = np.meshgrid(xvals, yvals)
Z = func_z(X,Y)

ax.scatter(X,Y, Z, c=X+Y)
ax.scatter(1,1, func_z(1,1), c='r')
#ax.plot(X,Y,Z, '-b')
ax.plot_surface(X, Y, Z, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
plt.show()

# plot contour
plt.contour(X, Y, Z, colors = 'k', linestyles = 'solid')
plt.show()



## 2.2
def func_zz(x,y):
    return (8*x+12*y+x**2 - 2*y**2)
plt.figure() 
xlist = np.linspace(-15, 10, 100)
ylist = np.linspace(-6, 12, 100)
X,Y = np.meshgrid(xlist, ylist)
Z = func_zz(X,Y)


ax.scatter(X,Y, Z, c=X+Y)
#ax.plot([-4],[3], func_zz(-4,3), c='r')
ax.scatter(-4,3, func_zz(-4,3), c='b',marker='*')
#ax.plot(X,Y,Z, '-b')
ax.plot_surface(X, Y, Z, cmap=plt.cm.jet, rstride=1, cstride=1, linewidth=0)
plt.show()

plt.contour(X, Y, Z)
plt.show()


## 2.4

def func_1(x):
    return (np.cos(1/x))
def func_t1(x):
        return (np.cos(1/x)+(np.sin(1/x)/x)-((2*np.sin(1/x)*x +np.cos(1/x))/(2*x**2)))
def func_2(x):
    return (np.cos(x))
def func_t2(x):
        return (np.cos(x)-np.sin(x)*x -0.5*np.cos(x)*x**2+(1/6)*np.sin(x)*x**3)

#x = np.linspace(-1.5,1.5,1000)
x = np.linspace(-5,5,1000)
Z1 = func_1(x)
Z2 = func_2(x)
T1 = func_t1(x)
T2 = func_t2(x)

# plot cos(1/x)

pylab.plot(x, Z1, label='cos(1/x)')
pylab.plot(x, T1, label='2nd Taylor')
plt.axvline(x=1)
pylab.legend(loc='upper left')
pylab.ylim((-5,5))
pylab.show()

# plot cos(x)
pylab.plot(x, Z2, label='cos(x)')
pylab.plot(x, T2, label='2nd Taylor')
pylab.legend(loc='upper left')
pylab.ylim((-5,5))
pylab.show()

