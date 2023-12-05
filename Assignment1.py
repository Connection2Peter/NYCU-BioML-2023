##### Import
import os, sys
import pandas as pd

n = 10
Ps = pd.read_csv(sys.argv[1], sep='\t', header=None).values.tolist()
Ss = {}

Pd = open(os.path.join(sys.argv[2], 'positive.fasta'), 'w')
Nd = open(os.path.join(sys.argv[2], 'negative.fasta'), 'w')

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
