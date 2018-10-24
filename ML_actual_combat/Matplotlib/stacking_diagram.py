# coding=utf-8
"""堆叠图用于显示『部分对整体』随时间的关系。
堆叠图基本上类似于饼图，只是随时间而变化
"""

import matplotlib.pyplot as plt

days = [1, 2, 3, 4, 5]

sleeping = [7, 8, 6, 11, 7]
eating = [2, 3, 4, 3, 2]
working = [7, 8, 7, 2, 2]
playing = [8, 5, 7, 8, 13]
plt.plot([], [], color='m', label='Sleeping', linewidth=5)
plt.plot([], [], color='c', label='Eating', linewidth=5)
plt.plot([], [], color='r', label='Working', linewidth=5)
plt.plot([], [], color='k', label='Playing', linewidth=5)

plt.stackplot(days, sleeping, eating, working, playing, colors=['m', 'c', 'r', 'k'])

plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()
