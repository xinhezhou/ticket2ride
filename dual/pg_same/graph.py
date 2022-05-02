import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize = (10, 10))
corners_x = [3, 5, 6, 5, 3, 2, 3]
corners_y = [2, 2, 3.73, 5.46, 5.46, 3.73, 2]
center_x, center_y = (4, 3.73)
# draw hexagon
plt.plot(corners_x, corners_y, color='k')
# draw routes 
for i in range(6):
    x, y = corners_x[i], corners_y[i]
    plt.plot([center_x, x], [center_y, y], color='k')
    circle_count = 50
    for j in range(circle_count+1):
        plt.plot(x + (center_x-x)*j/(circle_count+1), y + (center_y-y)*j/(circle_count+1),
                 alpha=0.2,
                 markersize = 8,
                 marker = "o", 
                 color=u'#1f77b4')


#triangle points and connect them
# plt.plot(5, 7,  marker = "o")
# plt.plot([2, 5, 8], [5, 7, 5])
plt.xlim(1, 10)
plt.ylim(1, 10)
plt.show()