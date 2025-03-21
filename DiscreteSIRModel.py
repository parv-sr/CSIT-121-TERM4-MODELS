import matplotlib.pyplot as plt 
import numpy as np 

N = 10000

I_0 = 80
S_0 = 9920
R_0 = 0

beta = 0.5
gamma = 0.1

s_t = S_0
i_t = I_0
r_t = R_0

def susceptible_at_tp1(S_t, I_t):
    return S_t - beta * S_t * (I_t/N)

def infected_at_tp1(S_t, I_t):
    return I_t + beta * S_t * (I_t/N) - gamma * I_t

def recovered_at_tp1(R_t, I_t):
    return R_t + gamma * I_t

susceptible_n = []
infected_n = []
recovered_n = []
time = np.linspace(0, 9, 50)  

for i in range(50):
    new_s = susceptible_at_tp1(s_t, i_t)
    new_i = infected_at_tp1(s_t, i_t)
    new_r = recovered_at_tp1(r_t, i_t)

    s_t, i_t, r_t = new_s, new_i, new_r 

    susceptible_n.append(round(s_t))
    infected_n.append(round(i_t))
    recovered_n.append(round(r_t))

x = np.array(time)
y1 = np.array(susceptible_n)
y2 = np.array(infected_n)
y3 = np.array(recovered_n)

a = infected_n.sort()
print("The maximum number of infected people are", infected_n[0])

plt.plot(x, y1, label="Susceptible", color="red")
plt.plot(x, y2, label="Infected", color="blue")
plt.plot(x, y3, label="Recovered", color="green")
plt.legend()
plt.grid(True)
plt.show()
