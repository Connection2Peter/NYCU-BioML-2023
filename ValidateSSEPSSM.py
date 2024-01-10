import os, sys
import numpy as np
from lib import ssepssm

if len(sys.argv) != 3:
	exit("Usage: python " + sys.argv[0] + " <p> <n>")

pFn, nFn = sys.argv[1], sys.argv[2]

Pfs = os.listdir(pFn)
Nfs = os.listdir(nFn)

for i in Pfs:
    for j in Nfs:
        if i == j:
            ap = ssepssm.Text2Vector(open(os.path.join(pFn, i)).read())
            an = ssepssm.Text2Vector(open(os.path.join(nFn, j)).read())

            f = False
            for k in ap:
                if len(k) != 115:
                    f = True
                    break

            for k in an:
                if len(k) != 115:
                    f = True
                    break

            if f:
                continue

            a = np.array(ap)
            b = np.array(an)

            diff = np.linalg.norm(a - b)

            if diff < 10:
                continue

            print(i, diff)

#PSPSSVVGFGAGPGSGEVVRYVEADLRKTAEDVVEKARRLCIANAMHALIEVIEG.ssepssm 10.87200817867259
#ISSSKSASGHQTAWFGAPGQRSMAEVVKMGRPQNKTTKQNVNVGSEINHEHEVNA.ssepssm 10.36175629188268