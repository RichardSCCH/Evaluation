import json
import re
import string

import pandas as pd
from bleu.bleu import Bleu
from rouge_impl import RougeScorer
import numpy as np
import csv


def evaluate(predictions, references):
    """
    Evaluates the overlap of two dicts containing a text and a reference text

    Args:
        predictions (_type_): test text
        references (_type_): reference text
    """
    scores_bleu, bleu, sentence_scores_bleu = Bleu(4).compute_score(references, predictions, 0)
    print("Bleu_1: ", np.mean(scores_bleu[0]))
    print("Bleu_2: ", np.mean(scores_bleu[1]))
    print("Bleu_3: ", np.mean(scores_bleu[2]))
    print("Bleu_4: ", np.mean(scores_bleu[3]))
    print("BLEU: ", bleu)

    sentence_scores_rouge, score_rouge = RougeScorer().compute_score(references, predictions)
    print("ROUGE-L: ", score_rouge)

    sentence_scores_rouge = [s['rougeL'].fmeasure for s in sentence_scores_rouge]
    return sentence_scores_bleu, sentence_scores_rouge


def evaluate_rencos(prediction_file: string, reference_file: string, length: int = 30):
    """
    Opens the files on the paths and reads the contents into dicts which are 
    passed to the main evaluation method.

    Args:
        prediction_file (string): path to prediction-file
        reference_file (string): path to reference-file
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


def evaluate_codegnngru(prediction_file: string, reference_file: string):
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


def evaluate_ncs(prediction_json_file: string, reference_file: string):
    """Read the prediction and reference files into a dictionaries of id:text

    Args:
        prediction_json_file (string): path to prediction file
        reference_file (string): path to reference file
    
    Returns:
        [type]: [bleu, rouge]
    """
    with open(prediction_json_file, 'r') as j:
        prediction = json.load(j)
        prediction = {int(k): v for k, v in prediction.items()}

    with open(reference_file, 'r') as inp:
        lines = inp.readlines()
        references = {k: [v.strip().lower()] for k, v in enumerate(lines) if k in prediction}

    return evaluate(prediction, references)


def print_separator(title: string):
    print(title)
    print("=" * 40)


if __name__ == '__main__':
    print("=" * 40)
    print_separator("Evaluating CodeGnnGru:")
    codegnngru_bleu, codegnngru_rouge = evaluate_codegnngru(
        "C:/git/CodeSummarization/AstAttendGru/modelout/predictions/predict-codegnngru.tsv",
        "C:/git/CodeSummarization/data/comments.tsv"
    )

    print("=" * 40)
    print_separator("Evaluating NCS:")
    # TODO: run "bash generate.sh 0 code2jdoc test/code.original_subtoken" - same data as in rencos
    # wait for ncs to stop training, then the test method in transformer.sh is called 
    ncs_bleu, ncs_rouge = evaluate_ncs(
        "C:/git/CodeSummarization/NCS/tmp/code2jdoc_beam.json",
        "C:/git/CodeSummarization/data/test/javadoc.original"
    )

    print("=" * 40)
    print_separator("Evaluating Rencos:")
    rencos_bleu, rencos_rouge = evaluate_rencos(
        "C:/git/CodeSummarization/Rencos-funcom/samples/java/output/test.out",
        "C:/git/CodeSummarization/Rencos-funcom/samples/java/test/test.txt.tgt",
        30
    )

    evaluation_result = {
        'rencos_bleu': rencos_bleu, 'rencos_rouge': rencos_rouge,
        'codegnngru_bleu': codegnngru_bleu, 'codegnngru_rouge': codegnngru_rouge,
        'ncs_bleu': ncs_bleu, 'ncs_rouge': ncs_rouge,
    }
    evaluation_result = pd.DataFrame({key: pd.Series(value) for key, value in evaluation_result.items()})
    evaluation_result.to_csv('C:/git/CodeSummarization/Evaluation/evaluation_result_exploration/evaluation-result.csv')

# Last run output:
# ========================================
# Evaluating Rencos:
# ========================================
# Bleu_1:  0.38583091553849247
# Bleu_2:  0.27740133566394837
# Bleu_3:  0.2051497379216083
# Bleu_4:  0.15166545833788106
# BLEU:  0.21021624180197823
# ROUGE-L:  0.460496412261527
# ========================================
# Evaluating CodeGnn (AstAttendGru):
# ========================================
# Bleu_1:  0.35541200019939956
# Bleu_2:  0.24411771292323572
# Bleu_3:  0.17100619576244402
# Bleu_4:  0.1214528468965409
# BLEU:  0.17935059884145138
# ROUGE-L:  0.4501288006668832
# ========================================
# Evaluating NCS:
# ========================================
# Bleu_1:  0.4431230405433319
# Bleu_2:  0.38011031291184033
# Bleu_3:  0.32941010636025475
# Bleu_4:  0.27237912297683153
# BLEU:  0.32433435685428397
# ROUGE-L:  0.5415620067748933
