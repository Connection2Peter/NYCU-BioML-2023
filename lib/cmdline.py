##### Import
import os, argparse



##### Global arguments
Helps = {
    "p": "Path to Positive dataset",
    "n": "Path to Negative dataset",
    "o": "Path to Output model",
    "i": "Path to input TSV file",
    "m": "Path to Saved model",
}



##### Functions
def ArgumentParser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--positive-data", metavar="required", help=Helps['p'], required=True)
    parser.add_argument("-n", "--negative-data", metavar="required", help=Helps['n'], required=True)

    return parser

def ArgumentCheck(Config):
    if not os.path.isfile(Config.positive_data):
        return "Error:\n {} '{}' doesn't exist".format(Helps['p'], Config.positive_data)

    if not os.path.isfile(Config.negative_data):
        return "Error:\n {} '{}' doesn't exist".format(Helps['n'], Config.negative_data)

    return ""

def ArgumentParser_TrainTest():
    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--positive-data", metavar="required", help=Helps['p'], required=True)
    parser.add_argument("-n", "--negative-data", metavar="required", help=Helps['n'], required=True)
    parser.add_argument("-o", "--output-model", metavar="required", help=Helps['o'], required=True)

    return parser

def ArgumentCheck_TrainTest(Config):
    if not os.path.isfile(Config.positive_data):
        return "Error:\n {} '{}' doesn't exist".format(Helps['p'], Config.positive_data)

    if not os.path.isfile(Config.negative_data):
        return "Error:\n {} '{}' doesn't exist".format(Helps['n'], Config.negative_data)

    if os.path.isfile(Config.output_model):
        return "Error:\n {} '{}' already exist".format(Helps['o'], Config.output_model)

    return ""

def ArgumentParser_IndependentTest():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input-file", metavar="required", help=Helps['i'], required=True)
    parser.add_argument("-m", "--model-path", metavar="required", help=Helps['m'], required=True)

    return parser

def ArgumentCheck_IndependentTest(Config):
    if not os.path.isfile(Config.input_file):
        return "Error:\n {} '{}' doesn't exist".format(Helps['i'], Config.input_file)

    if not os.path.isfile(Config.model_path):
        return "Error:\n {} '{}' doesn't exist".format(Helps['m'], Config.model_path)

    return ""
