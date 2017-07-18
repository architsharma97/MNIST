import matplotlib.pyplot as plt

def smooth(vals):
	smooth_vals = []
	alpha = 0.0001
	smooth_vals.append(vals[0])
	for val in vals[1:]:
		smooth_vals.append(alpha * val + (1. - alpha) * smooth_vals[-1])
	return smooth_vals

# plt.xlim(50, 100)
# plt.ylim(50, 100)

legend_entries = []
for k in [1]:
	f = open('training_test_lin_1_gc5.0_100_0.0001.txt', 'r').read().splitlines()
	vals = [float(line.split(',')[3]) for line in f]
	smoothened = smooth(vals) 
	plt.plot(smoothened[::600])
	legend_entries += ['(Linear Subnetworks): Subnetwork Cost']

	vals = [float(line.split(',')[4]) for line in f]
	smoothened = smooth(vals) 
	plt.plot(smoothened[::600])
	legend_entries += ['(Linear Subnetworks): Target Norms']

	# f = open('Results/disc/SF/training_sf_cmr_pbn_' + str(k) + '_100_0.0001.txt','r').read().splitlines()
	# vals = [float(line.split(',')[2]) for line in f]
	# smoothened = smooth(vals) 
	# plt.plot(smoothened[::600])
	# legend_entries += [str(k) + '-REINFORCE']

	# f = open('Results/disc/synthetic_gradients/training_sg_inp_act_out_grads_samp_lin_cmr_pbn_' + str(k) + '_100_0.0001.txt','r').read().splitlines()
	# vals = [float(line.split(',')[2]) for line in f]
	# smoothened = smooth(vals) 
	# plt.plot(smoothened[::600])
	# legend_entries += ['Linear subnetwork: No gradient clipping with ' + str(k) + '-REINFORCE training']

	# f = open('Results/disc/synthetic_gradients/training_sg_inp_act_out_grads_ld_cmr_bn_' + str(k) + '_100_0.0001.txt','r').read().splitlines()
	# vals = [float(line.split(',')[2]) for line in f]
	# smoothened = smooth(vals) 
	# plt.plot(smoothened[::600])
	# legend_entries += ['Deep subnetwork: No gradient clipping']

	# f = open('Results/disc/synthetic_gradients/training_sg_inp_act_out_grads_samp_lin_cmr_pbn_' + str(k) + '_gc1.0_100_0.0001.txt','r').read().splitlines()
	# vals = [float(line.split(',')[2]) for line in f]
	# smoothened = smooth(vals) 
	# plt.plot(smoothened[::600])
	# legend_entries += ['Linear subnetwork: Gradient clipped <= 1.0 elementwise norm with ' + str(k) + '-REINFORCE training']
	

	# f = open('Results/disc/synthetic_gradients/training_sg_inp_act_out_grads_samp_lin_cmr_pbn_' + str(k) + '_gc2.0_100_0.0001.txt','r').read().splitlines()
	# vals = [float(line.split(',')[2]) for line in f]
	# smoothened = smooth(vals) 
	# plt.plot(smoothened[::600])
	# legend_entries += ['Linear subnetwork: Gradient clipped <= 2.0 elementwise norm']

	# f = open('Results/disc/synthetic_gradients/training_sg_inp_act_out_grads_samp_lin_cmr_pbn_' + str(k) + '_gc5.0_100_0.0001.txt','r').read().splitlines()
	# vals = [float(line.split(',')[2]) for line in f]
	# smoothened = smooth(vals) 
	# plt.plot(smoothened[::600])
	# legend_entries += ['Linear subnetwork: Gradient clipped <= 5.0 elementwise norm']

	# f = open('Results/disc/synthetic_gradients/training_sg_inp_act_out_grads_ld_cmr_pbn_' + str(k) + '_gc1.0_100_0.0001.txt','r').read().splitlines()
	# vals = [float(line.split(',')[2]) for line in f]
	# smoothened = smooth(vals) 
	# plt.plot(smoothened[::600])
	# legend_entries += ['Deep subnetwork: Gradient clipped <= 1.0 elementwise norm']
	

	# f = open('Results/disc/synthetic_gradients/training_sg_inp_act_out_grads_ld_cmr_pbn_' + str(k) + '_gc2.0_100_0.0001.txt','r').read().splitlines()
	# vals = [float(line.split(',')[2]) for line in f]
	# smoothened = smooth(vals) 
	# plt.plot(smoothened[::600])
	# legend_entries += ['Deep subnetwork: Gradient clipped <= 2.0 elementwise norm']

	# f = open('Results/disc/synthetic_gradients/training_sg_inp_act_out_grads_ld_cmr_pbn_' + str(k) + '_gc5.0_100_0.0001.txt','r').read().splitlines()
	# vals = [float(line.split(',')[2]) for line in f]
	# smoothened = smooth(vals) 
	# plt.plot(smoothened[::600])
	# legend_entries += ['Deep subnetwork: Gradient clipped <= 5.0 elementwise norm']

	# f = open('Results/disc/synthetic_gradients/training_sg_inp_act_out_grads_samp_ld_cmr_pbn_' + str(k) + '_gc1.0_100_0.0001.txt','r').read().splitlines()
	# vals = [float(line.split(',')[2]) for line in f]
	# smoothened = smooth(vals) 
	# plt.plot(smoothened[::600])
	# legend_entries += ['Deep subnetwork with z conditioning: Gradient clipped <= 1.0 elementwise norm']

plt.grid()
plt.legend(legend_entries)

# plt.savefig('Results/disc/training_plots', ext='png', close=False, verbose=True, dpi=350, bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()
