import os, sys
from lib import tools
from lib import encoder
from lib import ssepssm
from flask import Flask, request



app = Flask(__name__)

@app.route('/ssepssm', methods=['GET'])
def ssepssm_route():
    sequence = request.args.get('seq')

    if sequence:
        hashCode = tools.GetSHA256(sequence)
        targPath = os.path.join(sys.argv[1], hashCode + ".ssepssm")

        if os.path.exists(targPath):
            return f'{targPath} already exist'
        else:
            ret = ssepssm.GenerateBySeqAndSave(
                prg="./ssepssm",
                seq=sequence,
                tdb="UniRef25-2015",
                rdb="NrPdb90-2022",
                search="psiblast",
                algo="VSH",
                matrix="pssm",
                threads=sys.argv[2],
                inifile="ssepssm.ini",
                saveTo=targPath
            )
            return f'{ret}'
            
    else:
        return 'Error: parameter seq not found in the request.'



if __name__ == '__main__':
    if len(sys.argv) != 3:
        exit("Usage: python " + sys.argv[0] + " <pathSSEPSSM> <threads>")

    app.run(host='0.0.0.0', port=port)
