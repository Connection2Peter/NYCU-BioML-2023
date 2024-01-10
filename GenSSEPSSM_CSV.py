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



##### Function
def GenSSEPSSM(seq):
    ssepssm.GenerateBySeqAndSave(
        prg="./ssepssm",
        seq=seq,
        tdb="UniRef25-2015",
        rdb="NrPdb90-2022",
        search="psiblast",
        algo="VSH",
        matrix="pssm",
        threads=20,
        inifile="ssepssm.ini",
        saveTo=saveTo
    )


##### Main
Encoder = encoder.IndependentTestCSV(pathInput)
seqIdx = 0

for seqID, seq in Encoder.Seqs.items():
    saveTo = os.path.join(pathOutput, tools.GetSHA256(seq) + ".ssepssm")

    seqIdx += 1

    if os.path.exists(saveTo):
        try:
            Vs = ssepssm.Text2Vector(open(saveTo, "r").read())
            Shapes = np.array(Vs).shape

            if Shapes[1] != 60:
                os.remove(saveTo)
                print(seqIdx, "/", seqID, "ReGegerate : Mis-shape", Shapes)
            else:
                print(seqIdx, "/", seqID, "Exist : Skip", Shapes)
                continue
        except:
            os.remove(saveTo)
            print(seqIdx, "/", seqID, "ReGegerate : Parse error")

    GenSSEPSSM(seq)
