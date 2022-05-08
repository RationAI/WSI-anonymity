from scipy import spatial
import numpy as np

def compare_cosine(vector1, vector2):
    """
    Applies cosine similarity on the provided vectors
    Args:
        vector1: vector to the image
        vector2: vector to the image
    """
    vector1 = vector1.flatten()
    vector2 = vector2.flatten()
    if vector1.size > vector2.size:
            vector1 = vector1[:vector2.size]
    else:
        vector2 = vector2[:vector1.size]
    return 1 - spatial.distance.cosine(vector1, vector2)

def compare_euclidean(vector1, vector2):
    """
    Applies cosine euclidean on the provided vectors
    Args:
        vector1: vector to the image
        vector2: vector to the image
    """
    vector1 = vector1.flatten()
    vector2 = vector2.flatten()
    if vector1.size > vector2.size:
            vector1 = vector1[:vector2.size]
    else:
        vector2 = vector2[:vector1.size]
    return 1/np.linalg.norm(vector1 - vector2)

def scores_to_probability(scores):
    """
    Translates scores to probability
    Args:
        scores: measured vector similarities
    """
    scores = np.nan_to_num(np.array(scores))
    scores = np.square(scores)
    scores = scores - np.min(scores)
    scores = scores / np.sum(scores)
    return scores

def choose_max(scores):
    """
    Assigns probability 1 to the most similar vector
    Args:
        scores: measured vector similarities
    """
    scores = np.nan_to_num(np.array(scores))
    index = np.argmax(scores)
    scores = np.zeros(scores.size)
    scores[index] = 1
    return scores

def get_probability(probe, originals, metric='cosine'):
    """
    Assigns probability field to the probe
    Args:
        probe: vector of the probe
        originals: vectors of the background knowledge
        feature_extractor: selected feature extractor
        metric: selected vector similarity metric
    """
    scores = []
    for original in originals:
        if metric == 'cosine':
            scores.append(compare_cosine(probe, original))
        else:
            scores.append(compare_euclidean(probe, original))
    return choose_max(scores), scores