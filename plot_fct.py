
import matplotlib.pyplot as plt
import numpy as np
def proba_T_consigne(Tint):
    return 1/(1+np.exp(-(-10.86+0.28*Tint)))
def PDD (PMV):
    return 100-95*np.exp(-0.03353*(PMV**4)-0.2179*PMV*PMV)
x = np.arange(18,28,1) # start,stop,step
y = proba_T_consigne(x)
PMV=np.arange(-3,4,1)
f=PDD(PMV)
plt.figure()
plt.plot(PMV, f)
plt.show()