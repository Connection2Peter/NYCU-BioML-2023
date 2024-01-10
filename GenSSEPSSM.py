##### Import
import os, sys
import numpy as np
from lib import tools
from lib import encoder
from lib import ssepssm



##### Argument
if len(sys.argv) != 3:
	exit("Usage: python " + sys.argv[0] + " <input> <output>")

pathInput = sys.argv[1]
pathOutput = sys.argv[2]



##### Main
Encoder = encoder.EntireSeqEncoder(pathInput)

AllSeqs = list(Encoder.SeqMaps.keys())
lenSeqs = len(AllSeqs)
for seqIdx in range(lenSeqs):
    print(seqIdx+1, "/", lenSeqs, end="\r")
    saveTo = os.path.join(pathOutput, tools.GetSHA256(AllSeqs[seqIdx]) + ".ssepssm")

    if os.path.exists(saveTo):
        continue

    ssepssm.GenerateBySeqAndSave(
        prg="./ssepssm",
        seq=AllSeqs[seqIdx],
        tdb="UniRef25-2015",
        rdb="NrPdb90-2022",
        search="psiblast",
        algo="VSH",
        matrix="pssm",
        threads=26,
        inifile="ssepssm.ini",
        saveTo=saveTo
    )
