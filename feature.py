##### Import
from Bio import motifs
from Bio.Align import substitution_matrices



##### Global Arguments
AA = "ACDEFGHIKLMNPQRSTVWYX"



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

### Seq encoding
def Seq2Int(seq):
	seqEncode = []

	for aa in seq:
		if aa in AA:
			seqEncode.append(AA.index(aa)+1)
		else:
			seqEncode.append(22)

	return seqEncode

def PositionMatrix(seq, Pos):
	seqEncode = []

	for i in range(len(seq)):
		if i+1 in Pos:
			seqEncode.append(1)
		else:
			seqEncode.append(2)

	return seqEncode

def PaddingSeq(Seqs, maxLen):
	for i in range(maxLen - len(Seqs)):
		Seqs.append(0)

	return Seqs

def PaddingMat(Mats, maxLen):
	Nan = [0 for i in range(len(Mats[0]))]

	for i in range(maxLen - len(Mats)):
		Mats.append(Nan)

	return Mats
