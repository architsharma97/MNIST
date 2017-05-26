import matplotlib.pyplot as plt

sf = open('Results/disc/SF/training_sf_100_0.0001.txt', 'r').read().splitlines()
pd_soft = open('Results/disc/PD/training_pd_soft_100_0.0001.txt','r').read().splitlines()
pd_hard = open('Results/disc/PD/training_pd_hard_100_0.0001.txt', 'r').read().splitlines()

costs_sf = [float(line.split(',')[2]) for idx, line in enumerate(sf) if idx%100 == 0 ]
costs_pd_soft = [float(line.split(',')[2]) for idx, line in enumerate(pd_soft) if idx%100 == 0]
costs_pd_hard = [float(line.split(',')[2]) for idx, line in enumerate(pd_hard) if idx%100 == 0]

plt.ylim(50,150)
plt.plot(costs_sf)
plt.plot(costs_pd_hard)
plt.plot(costs_pd_soft)
plt.grid()
plt.legend(['SF','PD_Hard','PD_Soft'])

plt.savefig('Results/disc/training_plots', ext='png', close=False, verbose=True, dpi=350, bbox_inches='tight', pad_inches=0)
plt.close()