##### Import
import os, sys
import numpy as np
from lib import tools
from lib import dataset
from lib import encoder



##### Argument
if len(sys.argv) != 3:
	exit("Usage: python " + sys.argv[0] + " <input> <output>")

pathInput = sys.argv[1]
pathOutput = sys.argv[2]



##### Main
window = 10

Encoder = encoder.EntireSeqEncoder(pathInput)
Encoder.LoadFromSSEPSSM(pathOutput)
SeqMaps = Encoder.showDiff(window)

for seq, Map in SeqMaps.items():
    numIsntance = len(Map[0])

    if numIsntance < 2:
        continue
    
    P, N = None, None
    newMap = Map[0]
    for pin in range(numIsntance):
        if newMap[pin] == 1:
            P = np.array(Map[1][pin])

        if newMap[pin] == 0:
            N = np.array(Map[1][pin])

    if P is None or N is None:
        continue

    diff = np.linalg.norm(P - N)

    if diff < 10:
        continue

    print(seq, diff)
