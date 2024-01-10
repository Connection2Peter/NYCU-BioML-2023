import sys, csv, requests
from lib import ssepssm
from multiprocessing import Pool



numMachine = len(IPs)



def GenSSEPSSM(pID, seq):
    tID = pID % numMachine
    result = ssepssm.getSSEPSSM(f"???.{IPs[tID]}", seq)
    #print(pID, result)

def DistComp(pathTSV):
    with Pool(numMachine) as pool:
        with open(pathTSV, 'r', newline='', encoding='utf-8') as tsvfile:
            Reader = csv.reader(tsvfile, delimiter='\t')

            index = 0
            for Rows in Reader:
                if len(Rows) >= 2:
                    if not ssepssm.isValidSeq(Rows[1]):
                        continue

                    pool.apply_async(
                        GenSSEPSSM,
                        (index, Rows[1],),
                    )

                    index += 1

        pool.close()
        pool.join()



if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit("Usage: python " + sys.argv[0] + " <tsv>")

    DistComp(sys.argv[1])
