
def rosenbrock(x,y):
    return (1-x)**2+100*(y-x**2)**2
def himmelblau (x,y):
    return (x**2+y-11)**2+(x+y**2-7)**2

def evaluation(ind):
    x=ind[0]
    y=ind[1]
    return rosenbrock(x,y), himmelblau(x,y)

#print (evaluation([-2.684,3.272]))

'''import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)
Z = himmelblau(X, Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='plasma')
plt.show()'''

