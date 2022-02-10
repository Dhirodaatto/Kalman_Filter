import numpy as np
import random
import matplotlib.pyplot as plt

true_data = [ [[x**3], [x]] for x in np.arange(0, 10, 0.1)]
true_data.extend([[[x], [x]] for x in np.arange(10, 20, 0.1)])

n = 0.5

error_m = [ [[x + random.random() * n * random.randint(-1, 1)], [x + random.random() * n * random.randint(-1, 1)]] for x in np.arange(0, 10, 0.1)]
error_m.extend([[[x**2 + random.random() * n * random.randint(-1, 1)], [x + random.random() * n * random.randint(-1, 1)]] for x in np.arange(10, 20, 0.1)])

pdatax = [i[0] for i in error_m]
pdatay = [i[1] for i in error_m]

f = open('test.txt', 'w')
f.write('\t'.join(str(i).replace(']', '').replace('[', '') for i in pdatax))
f.write('\n')
f.write('\t'.join(str(i).replace(']', '').replace('[', '') for i in pdatay))
f.close()

fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(pdatax, pdatay)
fig.savefig('ffs.png')
