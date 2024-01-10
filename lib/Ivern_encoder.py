##### Import
import os
import tempfile
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
class Iencoder:
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

    def _iFeatureEncode(self, encoder, ratio=None):
        print(encoder)

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        total_fasta = ""
        for i, seq in enumerate(NewDBs[0]):
            total_fasta += f">Pos{i}\n" + seq + "\n"

        for i, seq in enumerate(NewDBs[1]):
            total_fasta += f">Neg{i}\n" + seq + "\n"

        db1_len = len(NewDBs[0])
        db2_len = len(NewDBs[1])

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

        cmd = f"python bin/iFeature/iFeature.py --file {temp_fastaFile} --type {encoder} --out {temp_out} > /dev/null"

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

        X, y = [], []
        try:
            content = pd.read_csv(temp_out, sep='\t')
        except:
            print("Error: Cannot open file " + temp_out)
            return

        for i in range (db1_len + db2_len):
            X.append( list(content.iloc[i, 1:]) )

        y = [0] * db1_len + [1] * db2_len

        try:
            os.remove(temp_out)
        except:
            print("Error: Cannot remove file " + temp_out)
            return

        return pd.DataFrame(X), pd.DataFrame(y)




    def ToEAAC(self, window=5):

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        NewDBs = dataset.Balance(db1, db2)

        total_fasta = ""
        for i, seq in enumerate(NewDBs[0]):
            total_fasta += f">Pos{i}\n" + seq + "\n"

        for i, seq in enumerate(NewDBs[1]):
            total_fasta += f">Neg{i}\n" + seq + "\n"

        db1_len = len(NewDBs[0])
        db2_len = len(NewDBs[1])

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

        temp_out = temp_fastaFile[:-12] + "_" + "EAAC.csv"

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

        X, y = [], []
        try:
            content = pd.read_csv(temp_out, sep='\t')
        except:
            print("Error: Cannot open file " + temp_out)
            return

        for i in range (db1_len + db2_len):
            X.append( list(content.iloc[i, 1:]) )

        y = [0] * db1_len + [1] * db2_len

        try:
            os.remove(temp_out)
        except:
            print("Error: Cannot remove file " + temp_out)
            return

        return pd.DataFrame(X), pd.DataFrame(y)


    def ToEAAC_np(self, window=5):

        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        total_fasta = ""
        for i, seq in enumerate(db1):
            total_fasta += f">Pos{i}\n" + seq + "\n"

        for i, seq in enumerate(db2):
            total_fasta += f">Neg{i}\n" + seq + "\n"

        db1_len = len(db1)
        db2_len = len(db2)

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

        temp_out = temp_fastaFile[:-12] + "_" + "EAAC.csv"

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

        X, y = [], []
        try:
            content = pd.read_csv(temp_out, sep='\t')
        except:
            print("Error: Cannot open file " + temp_out)
            return

        for i in range (db1_len + db2_len):
            X.append( list(content.iloc[i, 1:]) )

        y = [0] * db1_len + [1] * db2_len

        try:
            os.remove(temp_out)
        except:
            print("Error: Cannot remove file " + temp_out)
            return

        return np.array(X), np.array(y)

    def ToKSCTriad(self):
        return self._iFeatureEncode("KSCTriad")

    def ToCTriad(self):
        return self._iFeatureEncode("CTriad")

    def ToCTDD(self):
        return self._iFeatureEncode("CTDD")

    def ToZSCALE(self):
        return self._iFeatureEncode("ZSCALE")

    def ToGTPC(self):
        return self._iFeatureEncode("GTPC")

    def ToGDPC(self):
        return self._iFeatureEncode("GDPC")

    def ToEGAAC(self):
        return self._iFeatureEncode("EGAAC")

    def ToBINARY(self):
        return self._iFeatureEncode("BINARY")

    def ToCKSAAGP(self):
        return self._iFeatureEncode("CKSAAGP")

    def ToCKSAAP(self):
        return self._iFeatureEncode("CKSAAP")

    def ToCTDC(self):
        return self._iFeatureEncode("CTDC")

    def ToDPC(self):
        return self._iFeatureEncode("DPC")

    def ToDDE(self):
        return self._iFeatureEncode("DDE")

    def ToTPC(self):
        return self._iFeatureEncode("TPC")

    def ToGAAC(self):
        return self._iFeatureEncode("GAAC")

    def ToCTDT(self):
        return self._iFeatureEncode("CTDT")


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


class IndependentTest:
    def __init__(self, dataset):
        self.dataset = dataset
        self.Kmers = []
        self.IDs    = []
        self.Pos   = []

    def TSV2Kmers(self, k):
        Seqs = pd.read_csv(self.dataset, sep='\t', header=None).values.tolist()
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

        return np.array(X)

    def ToEAAC(self, window=5):
        return self._iFeatureEncode("EAAC", window)

