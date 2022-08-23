from numpy import average
from rouge_score import rouge_scorer


def compute_rouge_score(refs, pred):
    """
    Compute the Rouge-L score for a reference and a prediction.
    Iterates over all ids in the dicts and extracts the first string from
    the lists to compute a score.

    Args:
        refs (dict{id:['string']}): dict of id:list<string> pairs
        pred (dict{id:['string']}): dict of id:list<string> pairs

    Returns:
        (list<Score>, float): list of Score objects and mean of f-measures
    """
    assert (sorted(refs.keys()) == sorted(pred.keys()))
    ids = list(refs.keys())

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = []

    for id in ids:
        scores.append(scorer.score(refs[id][0], pred[id][0]))

    avg_score = average([x['rougeL'].fmeasure for x in scores])
    return scores, avg_score
