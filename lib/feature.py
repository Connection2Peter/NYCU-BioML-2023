##### Import
from Bio import motifs
from Bio.Align import substitution_matrices

import numpy as np
import pandas as pd

##### Global Arguments
AA = "ACDEFGHIKLMNPQRSTVWXY"



##### Functions
### One-hot encoding
def OneHot(Seqs):
    Datas = []

    for seq in Seqs:
        seqEncodess = []

        for aaQry in seq:
            for aa in AA:
                if aa == aaQry:
                    seqEncodess.append(1)
                else:
                    seqEncodess.append(0)

        Datas.append(seqEncodess)

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
        seqEncodes = []

        for idx in range(len(seq)):
            seqEncodes.append(Mat[seq[idx]][idx])

        Datas.append(seqEncodes)

    return Datas

### PSSM
def PSSM(Seqs):
    Datas = []

    Mat = motifs.create(Seqs, AA).counts.normalize().log_odds()

    for seq in Seqs:
        seqEncodes = []

        for idx in range(len(seq)):
            seqEncodes.append(Mat[seq[idx]][idx])

        Datas.append(seqEncodes)

    return Datas

### BLOSUM62
def BLOSUM62(Seqs):
    Datas = []

    Mat = substitution_matrices.load('BLOSUM62')

    for seq in Seqs:
        seqEncodes = []

        for idx in range(len(seq)-1):
            aa1, aa2 = seq[idx], seq[idx+1]

            if (aa1, aa2) in Mat:
                seqEncodes.append(Mat[(aa1, aa2)])
            elif (aa2, aa1) in Mat:
                seqEncodes.append(Mat[(aa2, aa1)])
            else:
                seqEncodes.append(0)

        Datas.append(seqEncodes)

    return Datas


##### Global Arguments
AA         = "ACDEFGHIKLMNPQRSTVWYX"

##### Functions
def MkPWM(Seqs):
    total  = 0
    seqlen = len(Seqs[0])
    PWM = np.zeros( (len(AA), seqlen ) )
    
    for seq in Seqs :
        i = 0
        total += 1
    
        for aa in seq :
            PWM[AA.find(aa)][i] += 1
            i += 1
    
    DF         = pd.DataFrame(PWM / total)
    DF.index   = [aa for aa in AA]
    
    return  DF
    
def MkPWMd(PWM_p, PWM_n):
    A_PWM_p = np.array(PWM_p)
    A_PWM_n = np.array(PWM_n)
    A_PWM_d = np.abs(A_PWM_p - A_PWM_n)
    
    A_PWM_d[ A_PWM_d < 0.02] = 0
    A_PWM_d[ 0.05 <= A_PWM_d ] = 4
    A_PWM_d[(0.04 <= A_PWM_d) & (A_PWM_d < 0.05)] = 3
    A_PWM_d[(0.03 <= A_PWM_d) & (A_PWM_d < 0.04)] = 2
    A_PWM_d[(0.02 <= A_PWM_d) & (A_PWM_d < 0.03)] = 1
    
    DF         = pd.DataFrame(A_PWM_d.astype(int))
    DF.index   = [aa for aa in AA]
    return  DF 

def MkPWMd2(PWM_p, PWM_n):
    A_PWM_p = np.array(PWM_p)
    A_PWM_n = np.array(PWM_n)
    A_PWM_d = (A_PWM_p - A_PWM_n)*1000
    
    DF         = pd.DataFrame(A_PWM_d)
    DF.index   = [aa for aa in AA]
    return  DF

def MkPWMd3(PWM_d2):
    A_PWM_d2   = PWM_d2.to_numpy()
    condition = np.logical_and(-10 < A_PWM_d2, A_PWM_d2 < 10)
    A_PWM_d2[condition] = 0
    
    DF         = pd.DataFrame(A_PWM_d2)
    DF.index   = [aa for aa in AA]
    return  DF    
    
def GetPWM(Seqs, PWM) :
    l = []

    for seq in Seqs :
        seq_pwm = []
        i = 0

        for aa in seq :
            element = PWM.loc[aa, i]
            seq_pwm.append(element)
     
            i += 1

        l.append(seq_pwm)
    
    return l

def Electric(Seqs) :
    Ep = ['K', 'R', 'H']
    En = ['D', 'E']
    Datas = []
    
    for seq in Seqs :
        Data = []
        
        for aa in seq:
            if aa in Ep :
                Data.append(1)
            
            elif aa in En :
                Data.append(-1)
            
            else:
                Data.append(0)
    
        Datas.append(Data)

    return Datas

def Polor(Seqs) :
    P     = ['D','N','E','Q','H','K','R','S','T','Y']
    #NonP  = ['A','C','F','G','L','I','M','P','V','W','X']
    Datas = []
    
    for seq in Seqs :
        Data = []
        
        for aa in seq:
            if aa in P :
                Data.append(1)
            
            else:
                Data.append(0)
    
        Datas.append(Data)

    return Datas
    
def Aromatic(Seqs) :
    A     = ['F','H','W','Y']
    #NonA  = ['A','C','G','L','I','M','P','V','D','N','E','Q','K','R','S','T','X']
    Datas = []
    
    for seq in Seqs :
        Data = []

        for aa in seq:
            if aa in A :
                Data.append(1)

            else:
                Data.append(0)

        Datas.append(Data)

    return Datas