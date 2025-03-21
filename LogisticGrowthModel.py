import matplotlib.pyplot as plt
import numpy as np  



P0 = 300  
K = 500000  
r_1 = 0.2
r_2 = 0.4
r_3 = 0.6
r_4 = 0.8
r_5 = 1
T = 100
dt = 0.1


time_steps = int(T / dt)
t_values_1 = np.linspace(0, T, time_steps)
P_values_1 = np.zeros(time_steps)
P_values_1[0] = P0  

time_steps = int(T / dt)
t_values_2 = np.linspace(0, T, time_steps)
P_values_2 = np.zeros(time_steps)
P_values_2[0] = P0  

time_steps = int(T / dt)
t_values_3 = np.linspace(0, T, time_steps)
P_values_3 = np.zeros(time_steps)
P_values_3[0] = P0  

time_steps = int(T / dt)
t_values_4 = np.linspace(0, T, time_steps)
P_values_4 = np.zeros(time_steps)
P_values_4[0] = P0  

time_steps = int(T / dt)
t_values_5 = np.linspace(0, T, time_steps)
P_values_5 = np.zeros(time_steps)
P_values_5[0] = P0  


for i in range(1, time_steps):
    P_values_2[i] = P_values_2[i-1] + r_2 * P_values_2[i-1] * (1 - P_values_2[i-1] / K) * dt

for i in range(1, time_steps):
    P_values_1[i] = P_values_1[i-1] + r_1 * P_values_1[i-1] * (1 - P_values_1[i-1] / K) * dt

for i in range(1, time_steps):
    P_values_3[i] = P_values_3[i-1] + r_3 * P_values_3[i-1] * (1 - P_values_3[i-1] / K) * dt

for i in range(1, time_steps):
    P_values_4[i] = P_values_4[i-1] + r_4 * P_values_4[i-1] * (1 - P_values_4[i-1] / K) * dt

for i in range(1, time_steps):
    P_values_5[i] = P_values_5[i-1] + r_5 * P_values_5[i-1] * (1 - P_values_5[i-1] / K) * dt

plt.plot(t_values_1, P_values_1, label="Population at r = 0.2", color="blue")
plt.plot(t_values_2, P_values_2, label="Population at r = 0.4", color="red")
plt.plot(t_values_3, P_values_3, label="Population at r = 0.6", color="green")
plt.plot(t_values_4, P_values_4, label="Population at r = 0.8", color="yellow")
plt.plot(t_values_5, P_values_5, label="Population at r = 1", color="pink")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Logistic Population Growth")
plt.legend()
plt.grid(True)
plt.show()

 


