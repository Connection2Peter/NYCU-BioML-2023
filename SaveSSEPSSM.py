##### Import
import sys
import numpy as np
from lib import cmdline
from lib import dataset
from lib import encoder



##### Argument
if len(sys.argv) != 4:
	exit("Usage: python " + sys.argv[0] + " <input> <output> <saveto>")



##### Main
Encoder = encoder.EntireSeqEncoder(sys.argv[1])
Encoder.LoadFromSSEPSSM(sys.argv[2])

dataset.SaveObject(Encoder.SeqSSEPSSMs, sys.argv[3])
