import random 
import numpy as np 
import matplotlib.pyplot as plt

N = 100
time_lst = list(range(N))  

def randomWalk():
    pos = 50
    step_lst = [pos]  

    for i in range(1, N):
        if pos == 0:
            pos += 1  
        elif pos == N:
            pos -= 1  
        else:
            pos += random.choice([-1, 1])  
        
        step_lst.append(pos)  
    return step_lst  

x = np.array(time_lst)
y1 = np.array(randomWalk())
y2 = np.array(randomWalk())
y3 = np.array(randomWalk())
y4 = np.array(randomWalk())


plt.plot(x, y1, "o-", markersize=3, label="Walk 1")
plt.plot(x, y2, "o-", markersize=3, label="Walk 2")
plt.plot(x, y3, "o-", markersize=3, label="Walk 3")
plt.plot(x, y4, "o-", markersize=3, label="Walk 4")

plt.xlabel("Time Step")
plt.ylabel("Position")
plt.title("1D Random Walk")
plt.grid()
plt.legend()
plt.show()
