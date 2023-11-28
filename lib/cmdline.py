##### Import
import os, argparse



##### Global arguments
Helps = {
    "p": "Path to Positive dataset",
    "n": "Path to Negative dataset",
}



##### Functions
def ArgumentParser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--positive-data", metavar="required", help=Helps['p'], required=True)
    parser.add_argument("-n", "--negative-data", metavar="required", help=Helps['n'], required=True)

    return parser

def ArgumentCheck(Config):
    if not os.path.isfile(Config.positive_data):
        return "Error:\n Positive dataset file '{}' doesn't exist".format(Config.positive_data)

    if not os.path.isfile(Config.negative_data):
        return "Error:\n Negative dataset file '{}' doesn't exist".format(Config.negative_data)

    return ""
