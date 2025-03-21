import math 
import numpy as np
import matplotlib.pyplot as plt

g = 9.8
v0 = 10
a = math.radians(60)

pos_x_lst = []


def pos_x(t, a):
    for i in range(5):
        V0 = 0
        y = V0 * math.cos(a) * t
        pos_x_lst.append()

    return y

def pos_y(t, a):
    for i in range(5):
        V0 = 0
        y = V0 * math.sin(a) * t - (1/2)*g*(t**2)
    return y 

h_max = ((v0*math.sin(a))**2)/2*g
print(h_max)


