import matplotlib.pyplot as plt
import numpy as np  
import math


"""

P0 = 2
r = 2.5
pt = P0
k = 50000
t = 100
dt = 1


time_steps = int(t / dt)
t_values_1 = np.linspace(0, t, time_steps)
P_values_1 = np.zeros(time_steps)
P_values_1[0] = P0  

ptp1 = r * pt * (1 - (pt / k))

for i in range(1, time_steps):
    pt = P_values_1[i-1]
    P_values_1[i] = r * pt * (1 - pt / k) * dt


plt.plot(t_values_1, P_values_1, label="Population at r = 1.5", color="blue")
plt.show()
"""

def p(t):
    k = 5000
    r = 0.5

    p = math.sqrt((math.exp(t * r) + 1)/r * k)
    return p

t = 10
valuelst = []

for i in range(t):
    population = p(t)
    valuelst.append(population)

x = np.array(valuelst)
y = np.linspace(1, t)

plt.plot(x, y)
plt.show() 