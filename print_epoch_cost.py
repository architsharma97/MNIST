import sys

f = open(sys.argv[1],'r').read().splitlines()

if len(sys.argv) > 2:
	r = int(sys.argv[2])
else:
	r = 1000

for i in range(r):
	sum = 0.0
	for j in range(600):
		sum += float(f[i*600 + j].split(',')[3])
	print "Epoch " + str(i+1) + ":", sum
