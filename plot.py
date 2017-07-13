import matplotlib.pyplot as plt

def smooth(vals):
	smooth_vals = []
	alpha = 0.0001
	smooth_vals.append(vals[0])
	for val in vals[1:]:
		smooth_vals.append(alpha * val + (1. - alpha) * smooth_vals[-1])
	return smooth_vals

# plt.xlim(50, 100)
# plt.ylim(60, 100)

legend_entries = []
for k in [1]:
	f = open('Results/disc/SF/training_test_' + str(k) + '_100_0.0001.txt','r').read().splitlines()
	vals = [float(line.split(',')[3]) for line in f]
	smoothened = smooth(vals) 
	plt.plot(smoothened[::600])
	legend_entries += ['Average squared norm of 1-REINFORCE']

	# # f = open('Results/disc/SF/gradcomp_pretense_' + str(k) + '_100_0.0001.txt','r').read().splitlines()
	# vals = [float(line.split(',')[idx + 4]) for line in f]
	# smoothened = smooth(vals) 
	# plt.plot(smoothened[::600])
	# legend_entries += ['Straight Through']

	# # f = open('Results/disc/SF/gradcomp_pretense_' + str(k) + '_100_0.0001.txt','r').read().splitlines()
	# vals = [float(line.split(',')[idx + 8]) for line in f]
	# smoothened = smooth(vals) 
	# plt.plot(smoothened[::600])
	# legend_entries += ['Synthetic Gradient']
	
plt.grid()
plt.legend(legend_entries)

# plt.savefig('Results/disc/training_plots', ext='png', close=False, verbose=True, dpi=350, bbox_inches='tight', pad_inches=0)
plt.show()
plt.close()
