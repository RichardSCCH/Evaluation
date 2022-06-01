import string
from bleu.bleu import Bleu
from rouge.rouge import Rouge
import numpy as np
import sys


def evaluate(predictions, references):
    """
    Evaluates the overlap of two dicts containing a text and a reference text

    Args:
        predictions (_type_): test text
        references (_type_): reference text
    """
    score_Bleu, scores_Bleu, bleu = Bleu(4).compute_score(references, predictions, 0)
    print("Bleu_1: ", np.mean(scores_Bleu[0]))
    print("Bleu_2: ", np.mean(scores_Bleu[1]))
    print("Bleu_3: ", np.mean(scores_Bleu[2]))
    print("Bleu_4: ", np.mean(scores_Bleu[3]))
    print("BLEU Score: ", score_Bleu)

    score_Rouge, scores_Rouge = Rouge().compute_score(references, predictions)
    print("ROUGe: ", score_Rouge)


def evaluate_rencos(prediction:string, reference:string, length:int = 30):
    """
    Opens the files on the paths and reads the contents into dicts which are 
    passed to the main evaluation method.

    Args:
        prediction (string): path to prediction-file
        reference (string): path to reference-file
        length (int, optional): max-length of a prediction. Defaults to 30.
    """
    with open(prediction, 'r') as r:
        hypothesis = r.readlines()
        res = {k: [" ".join(v.strip().lower().split()[:length])] for k, v in enumerate(hypothesis)}
    with open(reference, 'r') as r:
        references = r.readlines()
        gts = {k: [v.strip().lower()] for k, v in enumerate(references)}
    evaluate(res, gts)

if __name__ == '__main__':
    # main(sys.argv[1], sys.argv[2], eval(sys.argv[3]))
    evaluate_rencos(
        "C:/git/CodeSummarization/Rencos-funcom/trained/samples/java/output/test.out",
        "C:/git/CodeSummarization/Rencos-funcom/trained/samples/java/test/test.txt.tgt", 
        30)
