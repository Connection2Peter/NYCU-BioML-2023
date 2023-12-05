##### Import
import os, sys
import pandas as pd

if len(sys.argv) < 3:
	exit("Arg miss")

if not os.path.isfile(sys.argv[1]):
	exit("In file not exist")

n = 50
Ps = pd.read_csv(sys.argv[1], sep='\t', header=None).values.tolist()
Ss = {}

Pf = os.path.join(sys.argv[2], 'positive_{}.fasta'.format(n))
Nf = os.path.join(sys.argv[2], 'negative_{}.fasta'.format(n))

if os.path.isfile(Pf) or os.path.isfile(Nf):
	exit("Out file exist")

Pd = open(Pf, 'w')
Nd = open(Nf, 'w')

for p in Ps:
	if p[3] in Ss:
		Ss[p[3]].append(p[1])
	else:
		Ss[p[3]] = [p[1]]

for k, v in Ss.items():
	sl = len(k)

	for i in range(n, sl-n):
		if k[i] != "K":
			continue
		
		f = k[i-n:i+n+1]

		rp = i+1
		if rp in v:
			Pd.write(">{}\n{}\n".format(i, f))
		else:
			c = True

			for vv in v:
				if abs(vv-rp) < n:
					c = False

			if c:
				Nd.write(">{}\n{}\n".format(i, f))
