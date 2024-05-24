import torch
import random
import numpy as np

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.deterministic = True  # Guarantees reproducibility.
    torch.backends.cudnn.benchmark = False


def calculate_bleu_score(references, hypotheses):
    smoothing_function = SmoothingFunction().method1
    return np.mean([sentence_bleu([ref], hyp, smoothing_function=smoothing_function) for ref, hyp in zip(references, hypotheses)])