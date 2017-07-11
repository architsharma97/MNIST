import matplotlib.pyplot as plt

def smooth(vals):
	smooth_vals = []
	alpha = 0.0001
	smooth_vals.append(vals[0])
	for val in vals[1:]:
		smooth_vals.append(alpha * val + (1. - alpha) * smooth_vals[-1])
	return smooth_vals

# plt.xlim(50, 100)
plt.ylim(60, 100)

legend_entries = []
plt.title('Comparing effect of different BN')
for k in [1, 2, 5, 10]:
	f = open('Results/disc/synthetic_gradients/training_sg_inp_act_out_grads_samp_lin_cmr_bn_' + str(k) + '_100_0.0001.txt','r').read().splitlines()
	vals = [float(line.split(',')[2]) for line in f]
	smoothened = smooth(vals) 
	plt.plot(smoothened[::600])
	legend_entries += ['BN0']

	f = open('Results/disc/synthetic_gradients/training_sg_inp_act_out_grads_samp_lin_cmr_pbn_' + str(k) + '_100_0.0001.txt','r').read().splitlines()
	vals = [float(line.split(',')[2]) for line in f]
	smoothened = smooth(vals) 
	plt.plot(smoothened[::600],'-.')
	legend_entries += ['BN1']

	# f = open('Results/disc/SF/training_sf_cmr_pbn_' + str(k) + '_100_0.0001.txt','r').read().splitlines()
	# vals = [float(line.split(',')[2]) for line in f]
	# smoothened = smooth(vals) 
	# plt.plot(smoothened[::600])
	# legend_entries += ['pBN']

plt.grid()
plt.legend(legend_entries)

# plt.savefig('Results/disc/training_plots', ext='png', close=False, verbose=True, dpi=350, bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()
