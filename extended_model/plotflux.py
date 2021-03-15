import numpy as np
import matplotlib.pyplot as plt

# Rescale back to original units 
w = 0.04
a = 0.005
wr = w + a
e_r = wr - a
e_x = 10**-6
e_t = 3600
rescale = e_r * e_x / e_t

# Load data with the original parameters
baseL = np.loadtxt("base/flux_left.txt", comments="#", delimiter=" ", unpack=False)
baseR = np.loadtxt("base/flux_right.txt", comments="#", delimiter=" ", unpack=False)
times = baseL[:,0]

# Rescale back to original units 
basevaluesL = rescale * baseL[:,1] 
basevaluesR = rescale * baseR[:,1] 

basecumvaluesL = np.cumsum(basevaluesL)
basecumvaluesR = np.cumsum(basevaluesR)

# Load data with experimental parameters
experL = np.loadtxt("Fy11times10DY1up20pc/flux_left.txt", comments="#", delimiter=" ", unpack=False)
experR = np.loadtxt("Fy11times10DY1up20pc/flux_right.txt", comments="#", delimiter=" ", unpack=False)
# Rescale back to original units 
expervaluesL = rescale * experL[:,1] 
expervaluesR = rescale * experR[:,1] 

expercumvaluesL = np.cumsum(expervaluesL)
expercumvaluesR = np.cumsum(expervaluesR)


# Plot cumulative uptake for both scenarios
fig = plt.figure(figsize=(6,4))

plt.plot(times, basecumvaluesL, 'b--', label = 'Orig. (L)')
plt.plot(times, basecumvaluesR, 'r--', label = 'Orig. (R)')

plt.plot(times, expercumvaluesL, 'b', label = '$F_{Y,1,1}$*10, $D_{LY,1}$*1.2 (L)')
plt.plot(times, expercumvaluesR, 'r', label = '$F_{Y,1,1}$*10, $D_{LY,1}$*1.2 (R)')

plt.xlabel('Time [h]')
plt.ylabel('Phosphate taken by root [mol/dm]')
plt.xticks(np.arange(0,25,4))
plt.xlim([0,24])
plt.legend()
plt.show()
# fig.savefig('Fy11times10DY1up20pc.eps', format='eps')
print(f'Total uptake at time {times[-1]}: left: {basecumvaluesL[-1]}, right: {basecumvaluesR[-1]} ')
print(f'Total uptake at time {times[-1]}: left: {expercumvaluesL[-1]}, right: {expercumvaluesR[-1]} ')