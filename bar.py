import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

ticks = [640, 960, 1280, 1920, 2560]

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

#font = {'family'   : 'serif',
#        'serif'     : ['ariel rounded mt bold']}
#plt.rcParams.update({'font.size': 24})
#plt.rc('font',**font)        
#plt.rcParams['font.family'] = 'cmr10'
plt.rcParams['font.serif'] = "Times New Roman"
plt.rcParams['font.family'] = "serif"
#rc('font', **{'family': 'serif', 'serif': ['Times New Roman']})
#rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
#rc('text', usetex=True)

size = 15
plt.title('Correct runs vs. reservoir size on 5-bit task', fontsize=size, fontname="Times New Roman")
plt.xticks(ticks, ('640', '960', '1280', '1920', '2560'), fontsize=size, fontname="Times New Roman")
plt.yticks(fontsize=size)
plt.ylabel('Percent correct', fontsize=size, fontname="Times New Roman")
plt.xlabel('Reservoir size', fontsize=size, fontname="Times New Roman")

for tick in ax.get_xticklabels():
   tick.set_fontname("Times New Roman")
for tick in ax.get_yticklabels():
   tick.set_fontname("Times New Roman")

#plt.legend(loc='best')
box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * .75, box.height])
#plt.legend(bbox_to_anchor=(1,1), loc='upper left' )
#plt.legend(bbox_to_anchor=(0.5, 0., 0.5, 0.5), loc='best' )
plt.legend( loc='lower right', fontsize=size )
plt.show()
  







