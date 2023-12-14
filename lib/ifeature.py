import os
import tempfile

import pandas as pd
from Bio import SeqIO

class IFeature:

    def __init__(self, db1, db2, encoder):
        self.db1 = db1
        self.db2 = db2
        self.temp = ""
        self.encoder = encoder
        self.neg_len = 0
        self.pos_len = 0


    def iFeature(self):
        db1 = [str(record.seq) for record in SeqIO.parse(self.db1, "fasta")]
        db2 = [str(record.seq) for record in SeqIO.parse(self.db2, "fasta")]

        self.neg_len = len(db1)
        self.pos_len = len(db2)

        # Create a temporary file name
        temp_dir = "../bin/iFeature/tmp"
        if not os.path.exists(temp_dir):
            os.mkdir(temp_dir)
        
        temp_fastaFile = tempfile.NamedTemporaryFile(dir=temp_dir).name + "_total.fasta"

        try:
            fread = open(self.db1, "r")
            txt = fread.read()
            fread.close()
        except:
            print("Error: Cannot open file " + self.db1)
            return

        try:
            fread = open(self.db2, "r")
            txt += fread.read()
            fread.close()
        except:
            print("Error: Cannot open file " + self.db2)
            return

        try:
            fwrite = open(temp_fastaFile, "w")
            fwrite.write(txt)
            fwrite.close()
        except:
            print("Error: Cannot write file " + temp_fastaFile)
            return

        temp_out = temp_fastaFile[:-12] + "_" + self.encoder + ".csv"

        cmd = f"python ../bin/iFeature/iFeature.py --file {temp_fastaFile} --type {self.encoder} --out {temp_out}"

        try:
            os.system(cmd)
        except:
            print("Error: Cannot execute command " + cmd)
            return

        try:
            os.system("rm " + temp_fastaFile)
        except:
            print("Error: Cannot remove file " + temp_fastaFile)
            return

        self.temp = temp_out
        return


    def parse_csv(self):
        X, y = [], []
        try:
            content = pd.read_csv(self.temp, sep='\t')
        except:
            print("Error: Cannot open file " + self.temp)
            return

        X.append( list(content.iloc[0, 1:]) )
        y = [0] * self.pos_len + [1] * self.neg_len

        try:
            os.system("rm " + self.temp)
        except:
            print("Error: Cannot remove file " + self.temp)
            return

        return pd.DataFrame(X), pd.DataFrame(y)


    def Encode(self):
        self.iFeature()
        return self.parse_csv()


if __name__ == '__main__':
    temp_file = IFeature( "../dataset/Junjie/processed/PositiveData_50.fasta", "../dataset/Junjie/processed/NegativeData_50.fasta", "EAAC").Encode()
    print(temp_file)
