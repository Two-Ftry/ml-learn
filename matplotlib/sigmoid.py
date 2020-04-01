import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(inX):
    return 1/(1+math.exp(-inX))

x = np.arange(-10, 10, 0.2)
y = []
for xItem in x:
    y.append(sigmoid(xItem))
    
fig, ax = plt.subplots()
ax.plot(x, y)

plt.show()
