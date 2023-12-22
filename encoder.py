##### Import
import os
import numpy as np
import pandas as pd
from Bio import SeqIO
from lib import tools
from lib import feature
from lib import dataset
from lib import ssepssm
from sklearn.preprocessing import MinMaxScaler



##### Functions
### Encoder
class Encode:
    def __init__(self, db1, db2):
        self.db1 = db1
        self.db2 = db2

    def ToOneHot(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)

        for negativeData in feature.OneHot(NewDBs[0]):
            X.append(negativeData)
            y.append(0)

        for positiveData in feature.OneHot(NewDBs[1]):
            X.append(positiveData)
            y.append(1)

        return pd.DataFrame(X), pd.DataFrame(y)

    def ToAAC(self):
        X, y = [], []

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        del(db1, db2)

        for negativeData in feature.AAC(NewDBs[0]):
            X.append(negativeData)
            y.append(0)

        for positiveData in feature.AAC(NewDBs[1]):
            X.append(positiveData)
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

        for negativeData in feature.BLOSUM62(NewDBs[0]):
            X.append(negativeData)
            y.append(0)

        for positiveData in feature.BLOSUM62(NewDBs[1]):
            X.append(positiveData)
            y.append(1)

        return pd.DataFrame(X), pd.DataFrame(y)

### FromRawDB
class FromRawDB:
    def __init__(self, db):
        self.db = db
        self.SeqMaps = None
    
    def Load2Map(self):
        AllSeqPos, Seqs = {}, {}
        Datas = open(self.db, "r").read().splitlines()
    
        for SeqData in [data.split("\t") for data in Datas]:
            if SeqData[3] in AllSeqPos:
                AllSeqPos[SeqData[3]].append(int(SeqData[1]))
            else:
                AllSeqPos[SeqData[3]] = [int(SeqData[1])]

        for k, v in AllSeqPos.items():
            Seqs[k] = list(set(v))
            
        self.SeqMaps = Seqs
    
    def Statistic(self):
        Seqs = self.SeqMaps.keys()
        AllLens = [len(i) for i in Seqs]

        print("NumSeq:", len(Seqs))
        print("MaxLen:", max(AllLens))
        print("MinLen:", min(AllLens))
        print("\t".join(["k-Size", "Range", "Posi", "Nega"]))

        for kLen in range(10, 201, 10):
            numPositive = 0
            numNegative = 0

            for k, v in self.SeqMaps.items():
                seqLen = len(k)
                for position in range(kLen, seqLen-kLen):
                    if k[position] == 'K':
                        if position+1 in v:
                            numPositive += 1
                        else:
                            numNegative += 1

            Cals = [kLen, kLen*2+1, numPositive, numNegative]
            print("\t".join([str(i) for i in Cals]))



### Whole Seq Encoder
class EntireSeqEncoder:
    def __init__(self, db=""):
        if db != "":
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

    def LoadFromSSEPSSM_Norm(self, path, numFeature=60):
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
                self.SeqSSEPSSMs[seq] = [Pos, dataset.Normalize2D(SSEPSSM)]

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
    
    def toSeqKmerDB2DNorm(self, kLen):
        X, y = [], []

        for k, v in self.SeqSSEPSSMs.items():
            seqLen = len(k)

            if seqLen != len(v[1]):
                continue

            tmp = []
            for position in range(kLen, seqLen-kLen):
                if k[position] == 'K':
                    NormDatas = MinMaxScaler().fit_transform(v[1][position-kLen:position+kLen+1])
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

    def showDiff(self, kLen):
        X = []
        SeqMatMap = {}

        for k, v in self.SeqSSEPSSMs.items():
            seqLen = len(k)

            if seqLen != len(v[1]):
                continue

            for position in range(kLen, seqLen-kLen):
                if k[position] == 'K':
                    start, end = position-kLen, position+kLen+1
                    sequence = k[start:end]
                    FeatureMats = v[1][start:end]

                    if position+1 in v[0]:
                        category = 1
                    else:
                        category = 0

                    if sequence in SeqMatMap:
                        SeqMatMap[sequence][0].append(category)
                        SeqMatMap[sequence][1].append(FeatureMats)
                    else:
                        SeqMatMap[sequence] = [[category], [FeatureMats]]

        return SeqMatMap

### IndependentTest
class IndependentTest:
    def __init__(self, dataset):
        self.dataset = dataset
        self.Kmers = []

    def TSV2Kmers(self, k):
        Seqs = pd.read_csv(self.dataset, sep='\t', header=None).values.tolist()
        Kmer = []

        for Seq in Seqs[1:]:
            for frag in dataset.Seq2Kmer(Seq[1], k):
                Kmer.append(frag)

        self.Kmers = Kmer

    def ToPSSM(self):
        return feature.BLOSUM62(self.Kmers)
