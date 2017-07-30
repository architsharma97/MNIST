import sys

f = open(sys.argv[1],'r').read().splitlines()

# print training file
if int(sys.argv[2]) == 0:
	if len(sys.argv) > 3:
		r = int(sys.argv[3])
	else:
		r = 1000

	for i in range(r):
		sum = 0.0
		for j in range(600):
			sum += float(f[i*600 + j].split(',')[2])
		print "Epoch " + str(i+1) + ":", sum
# print validation
else:
	if len(sys.argv) > 3:
		r = int(sys.argv[3])
	else:
		r = 1000

	for i in range(r):
		print "Epoch " + str(i+1) + ":", f[i]
