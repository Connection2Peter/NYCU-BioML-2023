##### Import
import sys
import numpy as np
from lib import chart
from lib import tools
from lib import cmdline
from lib import dataset
from lib import encoder



##### Argument
if len(sys.argv) != 3:
	exit("Usage: python " + sys.argv[0] + " <input> <output>")



##### Main
#Encoder = encoder.FromRawDB(sys.argv[1])
#Encoder.Load2Map()
#Encoder.Statistic()

Encoder = encoder.EntireSeqEncoder(sys.argv[1])
Encoder.LoadFromSSEPSSM(sys.argv[2])

X, y = Encoder.toSeqKmerDB2D(55)

#print(X.shape, y.shape)

chart.VisualVector3D(X, y)
