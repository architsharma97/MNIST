import sys

f = open(sys.argv[1],'r').read().splitlines()

for i in range(1000):
	sum = 0.0
	for j in range(600):
		sum += float(f[i*600 + j].split(',')[2])
	print "Epoch " + str(i+1) + ":", sum

