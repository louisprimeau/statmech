from scipy.linalg import *
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.animation as animation

d = 2 # dimension
N = 16 # length
J = 1

lr = block_diag(*tuple(np.ones((N,N)) for _ in range(N))) - np.eye(N*N)
def E(s):
    return (lr.dot(s.ravel('C')) + lr.dot(s.ravel('F'))).reshape(N, N)

def b(s, idx):
    return J * np.sum(s[(idx[0] - 1, idx[0] + 1 - N, idx[0], idx[0]), (idx[1], idx[1], idx[1]-1, idx[1]+1-N)])
    
def P(s, beta, idx):
    return 1 / (1 + np.exp(-2 * beta * b(s, idx)))

frames = []
s = np.random.binomial(1, 1/2, size=(N, N))
s[s==0] = -1

energy_mean, energy_std = [], []
mag_mean, mag_std = [], []

temperatures = np.linspace(5, 0.01, 30)
N_iter = 5000
for T in temperatures:
    energy, magnetization = [], []
    beta = 1 / T
    for i in range(N_iter):
        idx = np.random.randint(0, N*N-1)
        idx = idx // N, idx % N
        p = P(s, beta, idx)
        s[idx] = 1 if np.random.binomial(1, p)==1 else -1
        if i > N_iter/3:
            energy.append(E(s) / s.size)
            magnetization.append(np.sum(s) / s.size)

    energy_mean.append(np.mean(energy))
    energy_std.append(np.std(energy))
    mag_mean.append(np.mean(magnetization))
    mag_std.append(np.std(magnetization))
    frames.append(s.copy())

fig, ax = plt.subplots(1)
ims = []
for i, frame in enumerate(frames):
    im = ax.imshow(frame, animated=True)
    if i == 0: ax.imshow(frame)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=True, repeat_delay=1000)

fig, (ax1, ax2) = plt.subplots(2)
ax1.errorbar(temperatures, energy_mean, yerr=energy_std)
ax1.set_xlim(5, 0)
ax1.set_xlabel('Temperature')
ax1.set_ylabel('Mean Energy')
ax2.errorbar(temperatures, mag_mean, yerr=mag_std)
ax2.set_xlim(5, 0)
ax2.set_xlabel('Temperature')
ax2.set_ylabel('Mean Magnetization')
plt.show()
