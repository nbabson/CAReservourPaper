import numpy as np
import matplotlib.pyplot as plt

e = np.array([     [71.7,2560],[70.8,1920],[6,1280],[0,960],[0,640],  ])
s3 = np.array([    [100,2560],[100,1920],[99.7,1280],[81.7,960],[1.7, 640] ])
n5 = np.array([    [100,2560],[  100,1920], [96.3,1280],[80.7,960],[5.8,640] ])
s4 = np.array([    [100,2560],[  100,1920], [92.5,1280],[35.5,960],[0,640] ])
s5 = np.array([    [100,2560],[  100,1920], [88.2,1280],[38.5,960],[1.3,640] ])

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)   
w = 30

x, y = e.T
ax.bar( y+(w*1.5),x, align='edge',width=w, label='Elementary')

x,y = s3.T
ax.bar( y+(w*1.5),x, align='edge', width=-w, label='Three state')

x,y = n5.T
ax.bar( y-(w*.5),x, align='edge', width=w, label='Neighborhood five')

x,y = s4.T
ax.bar( y-(w*.5),x, align='edge', width=-w, label='Four state')

x,y = s5.T
ax.bar( y-(w*2.5),x, align='edge', width=w, label='Five state')

plt.title('Correct runs vs. reservoir size on 5-bit task')

plt.ylabel('Percent correct')
plt.xlabel('Reservoir size')
#plt.legend(loc='best')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * .75, box.height])
plt.legend(bbox_to_anchor=(1,1), loc='upper left' )
plt.show()
  







