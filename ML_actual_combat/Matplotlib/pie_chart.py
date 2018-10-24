# coding=utf-8

"""
在plt.pie中，我们需要指定『切片』，这是每个部分的相对大小。
然后，我们指定相应切片的颜色列表。 接下来，我们可以选择指定图形的『起始角度』。
这使你可以在任何地方开始绘图。 在我们的例子中，我们为饼图选择了 90 度角，这意味着第一个部分是一个竖直线条。
接下来，我们可以选择给绘图添加一个字符大小的阴影，然后我们甚至可以使用explode拉出一个切片。
我们总共有四个切片，所以对于explode，如果我们不想拉出任何切片，我们传入0,0,0,0。
如果我们想要拉出第一个切片，我们传入0.1,0,0,0。
最后，我们使用autopct，选择将百分比放置到图表上面。
"""
import matplotlib.pyplot as plt

slices = [7, 2, 2, 13]
activities = ['sleeping', 'eating', 'working', 'playing']
cols = ['c', 'm', 'r', 'b']

plt.pie(slices,
        labels=activities,
        colors=cols,
        startangle=90,
        shadow=True,
        explode=(0.02, 0.1, 0, 0),
        autopct='%1.1f%%'
        )

plt.title('Interesting Graph\nCheck it out')
plt.show()
