import matplotlib.pyplot as plt

# sf = open('Results/classification/backprop/training_bp_relu_reg_0.5_100_0.001.txt', 'r').read().splitlines()
# pd_soft = open('Results/classification/synthetic_gradients/training_sg_relu_reg_0.5_100_0.001.txt','r').read().splitlines()
# pd_hard = open('Results/classification/training_sg_relu_reg_0.5_100_0.001.txt', 'r').read().splitlines()

# costs_sf = [float(line.split(',')[2]) for idx, line in enumerate(sf) if idx%50 == 0 ]
# costs_pd_soft = [float(line.split(',')[2]) for idx, line in enumerate(pd_soft) if idx%50 == 0]
# costs_pd_hard = [float(line.split(',')[2]) for idx, line in enumerate(pd_hard) if idx%500 == 0]

# plt.xlim(0,1000)
plt.ylim(0, 300)
# plt.plot(costs_sf)
# plt.plot(costs_pd_hard)
# plt.plot(costs_pd_soft)

for k in [1, 2, 3, 5, 10, 20]:
	f = open('Results/disc/SF/training_sf_' + str(k) + '_100_0.0001.txt','r').read().splitlines()
	plt.plot([float(line.split(',')[2]) for idx, line in enumerate(f) if idx % 50 == 0])

plt.grid()
plt.legend([1, 2, 3, 5, 10, 20])

# plt.savefig('Results/disc/training_plots', ext='png', close=False, verbose=True, dpi=350, bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()
