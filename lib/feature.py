##### Import
from Bio import motifs



##### Global Arguments
AA = "ACDEFGHIKLMNPQRSTVWXY"



##### Functions
### One-hot encoding
def OneHot(Seqs):
    Datas = []

    for seq in Seqs:
        SeqEncodes = []

        for aaQry in seq:
            for aa in AA:
                if aa == aaQry:
                    SeqEncodes.append(1)
                else:
                    SeqEncodes.append(0)

        Datas.append(SeqEncodes)

    return Datas

### AAC
def AAsIni():
    AAs = {}

    for aa in AA:
        AAs[aa] = 0
    
    return AAs

def AAC(Seqs):
    Datas = []

    for seq in Seqs:
        seqLen = len(seq)

        AAs = AAsIni()

        for aa in seq:
            if aa in AAs.keys():
                AAs[aa] += 1

        for aa in AAs.keys():
            AAs[aa] /= seqLen

        Datas.append(list(AAs.values()))

    return Datas

### PWM
def PWM(Seqs):
    Datas = []

    Mat = motifs.create(Seqs, AA).counts.normalize()

    for seq in Seqs:
        seqEncode = []

        for idx in range(len(seq)):
            seqEncode.append(Mat[seq[idx]][idx])

        Datas.append(seqEncode)

    return Datas

### PSSM
def PSSM(Seqs):
    Datas = []

    Mat = motifs.create(Seqs, AA).counts.normalize().log_odds()

    for seq in Seqs:
        seqEncode = []

        for idx in range(len(seq)):
            seqEncode.append(Mat[seq[idx]][idx])

        Datas.append(seqEncode)

    return Datas
