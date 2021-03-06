import numpy as np
import matplotlib.pyplot as plt
from dynamic_model import Physics_Model

physics_model = Physics_Model(0.1, None)

# let us get the measurements
datafile = open('test.txt', 'r')
data = datafile.readlines()
xlist = [float(x) for x in data[0].split('\t')]
ylist = [float(y) for y in data[1].split('\t')]

z = np.array([[[xlist[i]],[ylist[i]]] for i in range(0,len(xlist))])

# let system state be x
x = dict()
#x = [[None]* 10] * 10 
x['n-1'] = np.zeros((6,1))

# first let us look at F -> state transition matrix
F = physics_model.F

# now let us define the process noise matric -> Q
dt=0.01
Q = np.array(
        [[.25 * dt**4, .5 * dt**3, .5 * dt**2, 0, 0, 0],
         [.5 * dt**3, 1 * dt**2, 1 * dt, 0, 0, 0],
         [.5 * dt**2, 1 * dt, 1, 0, 0, 0],
         [0, 0, 0, .25 * dt**4, .5 * dt**3, .5 * dt**2],
         [0, 0, 0, .5 * dt**3, 1 * dt**2, 1 * dt],
         [0, 0, 0, .5 * dt**2, 1 * dt, 1]]
        )

# then let us define the P -> covariance
#P = [[None] * 10] * 1
P = dict()
#P = np.zeros((n, m, 6, 1))
P['n-1'] = np.identity(6) * 500

# the measurement uncertainty R 
R = np.identity(2) * 250

# now the observation matrix -> H
H = np.array(
        [[1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0]]
        )

# now the Kalman Gain -> K
K = dict()
K['n'] = np.zeros((6,6))

# defining the identity matrix
I = np.identity(6)

zn = z[0]

xplot = list()
yplot = list()

x['n+1'] = F @ x['n-1']
P['n+1'] = F @ P['n-1'] @ np.transpose(F) + Q

x['n+1'] = F @ x['n-1']
P['n+1'] = F @ P['n-1'] @ np.transpose(F) + Q

P['n-1'] = P['n+1']
x['n-1'] = x['n+1']
# now the calculations
for zn in z:

    K['n'] = P['n-1'] @ np.transpose(H) @ np.linalg.pinv((H @ P['n-1'] @ np.transpose(H) + R), -1)
    x['n'] = x['n-1'] + K['n'] @ (zn - H @ x['n-1'])

    P['n'] = (I - (K['n'] @ H)) @ P['n-1'] @ np.transpose(I - (K['n'] @ H)) + K['n'] @ R @ np.transpose(K['n'])
    
    x['n+1'] = F @ x['n']
    P['n+1'] = F @ P['n'] @ np.transpose(F) + Q
    
    xplot.append(x['n'][0])
    yplot.append(x['n'][3])

    x['n-1'] = x['n+1']
    P['n-1'] = P['n+1']

fig,ax = plt.subplots(nrows=1, ncols=1)
ax.plot(xplot,yplot)
fig.savefig('fuckyou.png')

print(x['n'])
