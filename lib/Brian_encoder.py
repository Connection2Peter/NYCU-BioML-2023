##### Import
import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from lib import tools
from lib import feature
from lib import Brian_feature as Bfeature
from lib import Ivern_encoder as Iencoder
from lib import dataset
from lib import ssepssm
import tempfile
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import re

##### Global Arguments

##### Functions
### Encoder
class Encode:
    def __init__(self, db1, db2):
        self.db1 = db1
        self.db2 = db2
        
        self.do_PWM_p   = False
        self.do_PWM_n   = False
        self.do_PWM_all = False
        self.do_PWM_d   = False
        self.do_PWM_d2  = False
        self.do_PWM_d3  = False
        
        self.PWM_p   = None
        self.PWM_n   = None
        self.PWM_all = None
        self.PWM_d   = None
        self.PWM_d2  = None
        self.PWM_d3  = None       
        
    def ToOneHot(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)

        for positiveData in feature.OneHot(NewDBs[0]):
            X.append(positiveData)
            y.append(0)

        for negativeData in feature.OneHot(NewDBs[1]):
            X.append(negativeData)
            y.append(1)

        return pd.DataFrame(X), pd.DataFrame(y)

    def ToAAC(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)

        for positiveData in feature.AAC(NewDBs[0]):
            X.append(positiveData)
            y.append(0)

        for negativeData in feature.AAC(NewDBs[1]):
            X.append(negativeData)
            y.append(1)

        return pd.DataFrame(X), pd.DataFrame(y)

    def ToPWM(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        y = [0] * len(NewDBs[0]) + [1] * len(NewDBs[1])

        del(db1, db2)

        for data in feature.PWM(NewDBs[0] + NewDBs[1]):
            X.append(data)

        return pd.DataFrame(X), pd.DataFrame(y)
    
    def ToPSSM(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        y = [0] * len(NewDBs[0]) + [1] * len(NewDBs[1])

        del(db1, db2)

        for data in feature.PSSM(NewDBs[0] + NewDBs[1]):
            X.append(data)

        return pd.DataFrame(X), pd.DataFrame(y)

    def ToBLOSUM62(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)

        for positiveData in feature.BLOSUM62(NewDBs[0]):
            X.append(positiveData)
            y.append(0)

        for negativeData in feature.BLOSUM62(NewDBs[1]):
            X.append(negativeData)
            y.append(1)

        return pd.DataFrame(X), pd.DataFrame(y)
    
    def ToPWM_p(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]
        
        if self.do_PWM_p == False :
            self.PWM_p = Bfeature.MkPWM(db1)
            self.do_PWM_p = True

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)
        
        
        for positiveData in Bfeature.GetPWM(NewDBs[0], self.PWM_p):
            X.append(positiveData)
            y.append(0)

        for negativeData in Bfeature.GetPWM(NewDBs[1], self.PWM_p):
            X.append(negativeData)
            y.append(1)
        #exit(self.PWM_p)
        return pd.DataFrame(X), pd.DataFrame(y)
    
    def ToPWM_n(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        if self.do_PWM_n == False :
            self.PWM_n = Bfeature.MkPWM(db2)
            self.do_PWM_n = True

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)
        
        for positiveData in Bfeature.GetPWM(NewDBs[0], self.PWM_n):
            X.append(positiveData)
            y.append(0)

        for negativeData in Bfeature.GetPWM(NewDBs[1], self.PWM_n):
            X.append(negativeData)
            y.append(1)
        
        #exit(self.PWM_n)
        return pd.DataFrame(X), pd.DataFrame(y)
    
    def ToPWM_all(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        if self.do_PWM_all == False :
            self.PWM_all = Bfeature.MkPWM(db1+db2)
            self.do_PWM_all = True

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)
        
        for positiveData in Bfeature.GetPWM(NewDBs[0], self.PWM_all):
            X.append(positiveData)
            y.append(0)

        for negativeData in Bfeature.GetPWM(NewDBs[1], self.PWM_all):
            X.append(negativeData)
            y.append(1)
        
        #exit(self.PWM_all)
        return pd.DataFrame(X), pd.DataFrame(y)
    
    def ToPWM_d(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]
        
        if self.do_PWM_p == False :
            self.PWM_p = Bfeature.MkPWM(db1)
            self.do_PWM_p = True
        
        if self.do_PWM_n == False :
            self.PWM_n = Bfeature.MkPWM(db2)
            self.do_PWM_n = True
            
        if self.do_PWM_d == False :
            self.PWM_d = Bfeature.MkPWMd(self.PWM_p, self.PWM_n)
            self.do_PWM_d = True
        
        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)
        
        for positiveData in Bfeature.GetPWM(NewDBs[0], self.PWM_d):
            X.append(positiveData)
            y.append(0)

        for negativeData in Bfeature.GetPWM(NewDBs[1], self.PWM_d):
            X.append(negativeData)
            y.append(1)
        
        #exit(self.PWM_d[20])
        return pd.DataFrame(X), pd.DataFrame(y)
 
    def ToPWM_d2(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        if self.do_PWM_p == False :
            self.PWM_p = Bfeature.MkPWM(db1)
            self.do_PWM_p = True
        
        if self.do_PWM_n == False :
            self.PWM_n = Bfeature.MkPWM(db2)
            self.do_PWM_n = True
            
        if self.do_PWM_d2 == False :
            self.PWM_d2 = Bfeature.MkPWMd2(self.PWM_p, self.PWM_n)
            self.do_PWM_d2 = True
        
        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)
        
        for positiveData in Bfeature.GetPWM(NewDBs[0], self.PWM_d2):
            X.append(positiveData)
            y.append(0)

        for negativeData in Bfeature.GetPWM(NewDBs[1], self.PWM_d2):
            X.append(negativeData)
            y.append(1)
        
        #exit(self.PWM_d2)
        return pd.DataFrame(X), pd.DataFrame(y)
   
    def ToPWM_d3(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        if self.do_PWM_p == False :
            self.PWM_p = Bfeature.MkPWM(db1)
            self.do_PWM_p = True
        
        if self.do_PWM_n == False :
            self.PWM_n = Bfeature.MkPWM(db2)
            self.do_PWM_n = True

        if self.do_PWM_d2 == False :
            self.PWM_d2 = Bfeature.MkPWMd2(self.PWM_p, self.PWM_n)
            self.do_PWM_d2 = True
        
        if self.do_PWM_d3 == False :
            self.PWM_d3 = Bfeature.MkPWMd3(self.PWM_d2)
            self.do_PWM_d3 = True

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)
        
        for positiveData in Bfeature.GetPWM(NewDBs[0], self.PWM_d3):
            X.append(positiveData)
            y.append(0)

        for negativeData in Bfeature.GetPWM(NewDBs[1], self.PWM_d3):
            X.append(negativeData)
            y.append(1)
        
        #exit(self.PWM_d3)
        return pd.DataFrame(X), pd.DataFrame(y)
    
    def ToElectric(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)
        
        for positiveData in Bfeature.Electric(NewDBs[0]):
            X.append(positiveData)
            y.append(0)

        for negativeData in Bfeature.Electric(NewDBs[1]):
            X.append(negativeData)
            y.append(1)

        return pd.DataFrame(X), pd.DataFrame(y)
    
    def ToPolor(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)
        
        for positiveData in Bfeature.Polor(NewDBs[0]):
            X.append(positiveData)
            y.append(0)

        for negativeData in Bfeature.Polor(NewDBs[1]):
            X.append(negativeData)
            y.append(1)
        
        return pd.DataFrame(X), pd.DataFrame(y)

    def ToAromatic(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)
        
        for positiveData in Bfeature.Aromatic(NewDBs[0]):
            X.append(positiveData)
            y.append(0)

        for negativeData in Bfeature.Aromatic(NewDBs[1]):
            X.append(negativeData)
            y.append(1)
        
        return pd.DataFrame(X), pd.DataFrame(y)

### Whole Seq Encoder
class EntireSeqEncoder:
    def __init__(self, db):
        AllSeqPos, Seqs = {}, {}
        Datas = open(db, "r").read().splitlines()
    
        for SeqData in [data.split("\t") for data in Datas]:
            if SeqData[3] in AllSeqPos:
                AllSeqPos[SeqData[3]].append(int(SeqData[1]))
            else:
                AllSeqPos[SeqData[3]] = [int(SeqData[1])]

        for k, v in AllSeqPos.items():
            Seqs[k] = list(set(v))
            
        self.SeqMaps = Seqs
        self.SeqSSEPSSMs = {}

    def LoadFromSSEPSSM(self, path, numFeature=60):
        for seq, Pos in self.SeqMaps.items():
            pathSSEPSSM = os.path.join(path, tools.GetSHA256(seq) + ".ssepssm")

            if not os.path.exists(pathSSEPSSM):
                continue

            isError = False
            SSEPSSM = ssepssm.Text2Vector(open(pathSSEPSSM, "r").read())

            for seqSSE in SSEPSSM:
                if len(seqSSE) != numFeature:
                    isError = True
                    break

            if not isError:
                self.SeqSSEPSSMs[seq] = [Pos, SSEPSSM]

    def toSeqDB(self, maxLen):
        X, y = [], []

        for k, v in self.SeqMaps.items():
            X.append(feature.PaddingSeq(feature.Seq2Int(k), maxLen))
            y.append(feature.PaddingSeq(feature.PositionMatrix(k, v), maxLen))

        return np.array(X), np.array(y)
    
    def toSeqDB3D(self, maxLen, factor):
        X, y = [], []

        for k, v in self.SeqSSEPSSMs.items():
            a, b = len(k),len(v[1])

            if a != b:
                continue

            NormalizeDatas = MinMaxScaler().fit_transform(v[1]) * factor

            X.append(feature.PaddingMat(NormalizeDatas.astype(int).tolist(), maxLen))
            y.append(feature.PaddingSeq(feature.PositionMatrix(k, v[0]), maxLen))

        return np.array(X), np.array(y)
    
    def toSeqKmerDB2D(self, kLen):
        X, y = [], []

        for k, v in self.SeqSSEPSSMs.items():
            seqLen = len(k)

            if seqLen != len(v[1]):
                continue

            tmp = []
            for position in range(kLen, seqLen-kLen):
                if k[position] == 'K':
                    X.append(np.reshape(v[1][position-kLen:position+kLen+1], -1))

                    if position+1 in v[0]:
                        y.append(1)
                    else:
                        y.append(0)

        return np.array(X), np.array(y)
    
    def toSeqKmerDB2DNorm(self, kLen, factor):
        X, y = [], []

        for k, v in self.SeqSSEPSSMs.items():
            seqLen = len(k)

            if seqLen != len(v[1]):
                continue

            tmp = []
            for position in range(kLen, seqLen-kLen):
                if k[position] == 'K':
                    NormDatas = MinMaxScaler().fit_transform(v[1][position-kLen:position+kLen+1]) * factor
                    X.append(np.reshape(NormDatas.astype(int).tolist(), -1))

                    if position+1 in v[0]:
                        y.append(1)
                    else:
                        y.append(0)

        return np.array(X), np.array(y)
    
    def toSeqKmerDB3D(self, kLen):
        X, y = [], []

        for k, v in self.SeqSSEPSSMs.items():
            seqLen = len(k)

            if seqLen != len(v[1]):
                continue

            tmp = []
            for position in range(kLen, seqLen-kLen):
                if k[position] == 'K':
                    X.append(v[1][position-kLen:position+kLen+1])

                    if position+1 in v[0]:
                        y.append(1)
                    else:
                        y.append(0)

        return np.array(X), np.array(y)
    
    def toSeqKmerDB3DNorm(self, kLen):
        X, y = [], []

        for k, v in self.SeqSSEPSSMs.items():
            seqLen = len(k)

            if seqLen != len(v[1]):
                continue

            tmp = []
            for position in range(kLen, seqLen-kLen):
                if k[position] == 'K':
                    X.append(MinMaxScaler().fit_transform(v[1][position-kLen:position+kLen+1]))

                    if position+1 in v[0]:
                        y.append(1)
                    else:
                        y.append(0)

        return np.array(X), np.array(y)

### IndependentTest
class IndependentTest:
    def __init__(self, dataset):
        self.dataset = dataset
        self.Kmers = []
        self.IDs    = []
        self.Pos   = []

    def TSV2Kmers(self, k):
        Seqs = pd.read_csv(self.dataset, sep=',', header=None).values.tolist()
        Kmer = []
        IDs  = []
        Pos  = []

        for Seq in Seqs[1:]:
            Frags, Positions = dataset.Seq2Kmer(Seq[1], k)
            for i, frag in enumerate(Frags):
                Kmer.append(frag)
                IDs.append(Seq[0])
                Pos.append(Positions[i])

        self.Kmers = Kmer
        self.IDs   = IDs
        self.Pos   = Pos


    def _iFeatureEncode(self, encoder, window):
        print(encoder)

        i = 0
        total_fasta = ""
        for seq in self.Kmers:
            total_fasta += f">Seq" + str(i) + "\n" + seq + "\n"
            i += 1


        temp_dir = "bin/iFeature/tmp"
        os.makedirs(temp_dir, exist_ok=True)

        temp_fastaFile = tempfile.NamedTemporaryFile(dir=temp_dir).name + "_total.fasta"

        try:
            fwrite = open(temp_fastaFile, "w")
            fwrite.write(total_fasta)
            fwrite.close()
        except:
            print("Error: Cannot write file " + temp_fastaFile)
            return

        temp_out = temp_fastaFile[:-12] + "_" + encoder + ".csv"

        cmd = f"python bin/iFeature/codes/EAAC.py {temp_fastaFile} {window} {temp_out} > /dev/null"

        try:
            os.system(cmd)
        except:
            print("Error: Cannot execute command " + cmd)
            return

        try:
            os.remove(temp_fastaFile)
        except:
            print("Error: Cannot remove file " + temp_fastaFile)
            return

        X = []
        try:
            content = pd.read_csv(temp_out, sep='\t')
        except:
            print("Error: Cannot open file " + temp_out)
            return

        for i in range ( len(self.Kmers) ):
            X.append( list(content.iloc[i, 1:]) )

        try:
            os.remove(temp_out)
        except:
            print("Error: Cannot remove file " + temp_out)
            return

        return pd.DataFrame(X)

    def ToEAAC(self, window=5):
        return self._iFeatureEncode("EAAC", window)
        
    '''
    def ToEAAC(self):
        return Iencoder.ToEAAC(self.Kmers)
    '''
    '''
    def ToEAAC_allseq(self, k, w):
        w = 5
        AA = 'ARNDCQEGHILKMFPSTWYV' #20為一組
        encodings=[]
        
        Seqs = pd.read_csv(self.dataset, sep='\t', header=None).values.tolist()
        for Seq in Seqs:
            name, sequence = Seq[0], Seq[1]
            code = [name]
            k_position = []
            
            for j in range(len(sequence)):
                if sequence[j] = 'K':
                    k_position.append(j)
                    
                if j < len(sequence) and j + w <= len(sequence):
                    
                    count = Counter(re.sub('-', '', sequence[j:j+w])) #去除破折號(gap)
                    for key in count:
                        count[key] = count[key] / len(re.sub('-', '', sequence[j:j+w]))
                    for aa in AA:
                        code.append(count[aa])
                        
                for pos in k_position
            encodings.append(code)

        return encodings
        '''
       