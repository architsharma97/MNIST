import matplotlib.pyplot as plt

# sf = open('Results/classification/backprop/training_bp_relu_reg_0.5_100_0.001.txt', 'r').read().splitlines()
# pd_soft = open('Results/classification/synthetic_gradients/training_sg_relu_reg_0.5_100_0.001.txt','r').read().splitlines()
# pd_hard = open('Results/classification/training_sg_relu_reg_0.5_100_0.001.txt', 'r').read().splitlines()

# costs_sf = [float(line.split(',')[2]) for idx, line in enumerate(sf) if idx%50 == 0 ]
# costs_pd_soft = [float(line.split(',')[2]) for idx, line in enumerate(pd_soft) if idx%50 == 0]
# costs_pd_hard = [float(line.split(',')[2]) for idx, line in enumerate(pd_hard) if idx%500 == 0]

# plt.xlim(0,1000)
plt.ylim(65, 100)
# plt.plot(costs_sf)
# plt.plot(costs_pd_hard)
# plt.plot(costs_pd_soft)

for k in [1, 5, 10]:
	f = open('Results/disc/synthetic_gradients/training_sg_inp_act_lin_' + str(k) + '_100_0.0001.txt','r').read().splitlines()
	plt.plot([float(line.split(',')[2]) for idx, line in enumerate(f) if idx % 600 == 0])
	f = open('Results/disc/synthetic_gradients/training_sg_inp_act_lin_no_clip_' + str(k) + '_100_0.0001.txt','r').read().splitlines()
	plt.plot([float(line.split(',')[2]) for idx, line in enumerate(f) if idx % 600 == 0])

plt.grid()
plt.legend(['SG_1', 'SG_skip_1', 'SG_5', 'SG_skip_5', 'SG_10', 'SG_skip_10']) 

# plt.savefig('Results/disc/training_plots', ext='png', close=False, verbose=True, dpi=350, bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()
