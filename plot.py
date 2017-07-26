import matplotlib.pyplot as plt

def smooth(vals):
	smooth_vals = []
	alpha = 0.0001
	smooth_vals.append(vals[0])
	for val in vals[1:]:
		smooth_vals.append(alpha * val + (1. - alpha) * smooth_vals[-1])
	return smooth_vals

# plt.xlim(50, 100)
# plt.ylim(0., 0.0000001)

legend_entries = []
for k in [1]:
	f = open('Results/disc/synthetic_gradients/tsgd_sgsch_11111_ld_1_100_0.001.txt', 'r').read().splitlines()[:]
	vals = [float(line.split(',')[2]) for line in f]
	smoothened = smooth(vals) 
	plt.plot(smoothened[::600])
	# plt.plot([float(line) for line in f])
	legend_entries += ['Deep Subnetworks: Conditioned on everything']

	f = open('Results/disc/synthetic_gradients/tsgd_sgsch_11111_lin_1_100_0.001.txt', 'r').read().splitlines()[:]
	vals = [float(line.split(',')[2]) for line in f]
	smoothened = smooth(vals) 
	plt.plot(smoothened[::600])
	# plt.plot([float(line) for line in f])
	legend_entries += ['Linear Subnetworks: Conditioned on everything']

	f = open('Results/disc/SF/training_reinforce_sch_1_100_0.001.txt', 'r').read().splitlines()[1:]
	vals = [float(line.split(',')[2]) for line in f]
	smoothened = smooth(vals) 
	plt.plot(smoothened[::600])
	# plt.plot([float(line) for line in f])
	legend_entries += ['1-REINFORCE']


plt.grid()
plt.legend(legend_entries)

# plt.savefig('Results/disc/training_plots', ext='png', close=False, verbose=True, dpi=350, bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()
