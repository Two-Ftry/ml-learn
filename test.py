
import math

x = 2

# print(math.log(x, 2))

x1 = 1
y1 = 0.5

x2 = 1.6
y2 = 0.6

# r = -(x1 * math.log(x1, 2) + x2 * math.log(x2, 2))

r = math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2))

print(r)