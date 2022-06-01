import json
import re
import string
from tkinter.ttk import Separator

import pandas as pd
from bleu.bleu import Bleu
from rouge.rouge import Rouge
import numpy as np
import csv
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
    print("BLEU: ", bleu)

    score_Rouge, scores_Rouge = Rouge().compute_score(references, predictions)
    print("ROUGe: ", score_Rouge)

    return (bleu, score_Rouge)


def evaluate_rencos(prediction_file:string, reference_file:string, length:int = 30):
    """
    Opens the files on the paths and reads the contents into dicts which are 
    passed to the main evaluation method.

    Args:
        prediction (string): path to prediction-file
        reference (string): path to reference-file
        length (int, optional): max-length of a prediction. Defaults to 30.
    
    Returns:
        [type]: [bleu, rouge
    """
    with open(prediction_file, 'r') as r:
        hypothesis = r.readlines()
        res = {k: [" ".join(v.strip().lower().split()[:length])] for k, v in enumerate(hypothesis)}
    with open(reference_file, 'r') as r:
        references = r.readlines()
        gts = {k: [v.strip().lower()] for k, v in enumerate(references)}
    return evaluate(res, gts)

def evaluate_codegnn(prediction_file:string, reference_file:string):
    """Read the prediction and reference files into a dictionaries of id:text

    Args:
        prediction_file (string): path to prediction file
        reference_file (string): path to reference file
    
    Returns:
        [type]: [bleu, rouge]
    """
    predictions = {}

    with open(prediction_file, mode='r') as inp:
        reader = csv.reader(inp, delimiter="\t")
        predictions = {rows[0]: [re.sub(r"<((s)|(/s)|(NULL))>", r"", rows[2]).strip().lower()] for rows in reader}
    
    with open(reference_file, mode='r') as inp:
        reader = csv.reader(inp, delimiter="\t")
        references = {rows[0]: [rows[1].strip().lower()] for rows in reader}
    
    del_list = []
    for ref_key in references.keys():
        if ref_key not in predictions.keys():
            del_list.append(ref_key)
    for k in del_list:
        del references[k]

    return evaluate(predictions, references)

def evaluate_ncs(prediction_json_file:string):
    """
    Evaluate the performance of NCS by reading the json file and splitting the 
    data into a prediction dict and a reference dict

    Args:
        prediction_json_file (string): path to json file
    
    Returns:
        [type]: [bleu, rouge]
    """
    print(prediction_json_file)
    with open(prediction_json_file, 'r') as j:
        ncs_pred = json.load(j)

    prediction = {x['id']:x['predictions'] for x in ncs_pred}
    reference = {x['id']:x['references'] for x in ncs_pred}

    return evaluate(prediction, reference)

def print_separator(title:string):
    print(title)
    print("="*40)


if __name__ == '__main__':
    # main(sys.argv[1], sys.argv[2], eval(sys.argv[3]))
    
    print("="*40)
    print_separator("Evaluating Rencos:")
    rencos_bleu, rencos_rouge = evaluate_rencos(
        "C:/git/CodeSummarization/Rencos-funcom/samples/java/output/test.out",
        "C:/git/CodeSummarization/Rencos-funcom/samples/java/test/test.txt.tgt", 
        30
    )

    print("="*40)
    print_separator("Evaluating CodeGnn (AstAttendGru):")
    codegnn_bleu, codegnn_rouge = evaluate_codegnn(
        "C:/git/CodeSummarization/AstAttendGru/modelout/predictions/predict-codegnngru.tsv",
        "C:/git/CodeSummarization/data/comments.tsv"
    )

    print("="*40)
    print_separator("Evaluating NCS:")
    ncs_bleu, ncs_rouge = evaluate_ncs("C:/git/CodeSummarization/NCS/tmp/code2jdoc.json")
