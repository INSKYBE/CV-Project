'''
我的训练完是csv格式，另存为txt格式即可
画loss图的代码，前提是results.txt文档中只能是数字，先删除掉txt中的字符
我的第1列是epoch是0，1，2，...的格式，删掉逗号
第2、3、4列分别是box、objectness、classification 的loss值
'''
import os

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
data1_loss =np.loadtxt("runs/results.txt")   #result.txt地址
#print(data1_loss[0])


x = data1_loss[:,0]   #冒号左边是行范围，冒号右边列范围。取第1列
y = data1_loss[:,1]   #取第2列
y1 = data1_loss[:,2]   #取第3列
y2 = data1_loss[:,3]   #取第4列

fig = plt.figure(figsize = (7,5))       #figsize是图片的大小`
ax1 = fig.add_subplot(1, 1, 1) # ax1是子图的名字`
pl.plot(x,y,'r-',label=u'Box_Loss')
pl.plot(x,y1,'g-',label=u'Objectness_Loss')
pl.plot(x,y2,'y-',label=u'Classification_Loss')

# ‘’g‘’代表“green”,表示画出的曲线是绿色，“-”代表画的曲线是实线，可自行选择，label代表的是图例的名称，一般要在名称前面加一个u，如果名称是中文，会显示不出来，目前还不知道怎么解决。
# p2 = pl.plot(x,y,'r-', label = u'Box_Loss')
#显示图例
# p3 = pl.plot(x,y1, 'b-', label = u'Objectness_Loss')
# p4 = pl.plot(x,y2, 'y-', label = u'Classification_Loss')

pl.legend()
pl.xlabel(u'epoch')
pl.ylabel(u'loss')
# plt.title(' loss for yolov3 models in training')
plt.savefig(os.path.join('runs/exp5', 'loss.png'))#保存图片，第一个是指存储路径，第二个是图片名字
plt.show()
