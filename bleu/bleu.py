from numpy import average
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


class Bleu:
    def __init__(self, n=4):
        self._n = n
        self._hypo_for_image = {}
        self.ref_for_image = {}

    def compute_score(self, gts, res, verbose=0):
        assert (sorted(gts.keys()) == sorted(res.keys()))
        ids = list(gts.keys())

        scores = [[], [], [], []]
        score = []

        for id in ids:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) >= 1)
            reference = [ref[0].split()]
            candidate = hypo[0].split()
            score.append(sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method4))
            scores[0].append(sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
            scores[1].append(sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
            scores[2].append(sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
            scores[3].append(sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))

        avg_scores = [average(x) for x in scores]

        return avg_scores, average(score), score

    def method(self):
        return "Bleu"
