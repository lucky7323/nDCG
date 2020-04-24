import numpy as np


def ndcg(rel_true, rel_pred, p=None, form="linear"):
    """ Returns normalized Discounted Cumulative Gain

    Args:
        rel_true (1-D Array): relevance lists for particular user, (n_songs,)
        rel_pred (1-D Array): predicted relevance lists, (n_pred,)
        p (int): particular rank position
        form (string): two types of nDCG formula, 'linear' or 'exponential'

    Returns:
        ndcg (float): normalized discounted cumulative gain score [0, 1]
    """
    rel_true = np.sort(rel_true)[::-1]
    p = min(len(rel_true), min(len(rel_pred), p))
    discount = 1 / (np.log2(np.arange(p) + 2))

    if form == "linear":
        idcg = np.sum(rel_true[:p] * discount)
        dcg = np.sum(rel_pred[:p] * discount)
    elif form == "exponential" or form == "exp":
        idcg = np.sum([2**x - 1 for x in rel_true[:p]] * discount)
        dcg = np.sum([2**x - 1 for x in rel_pred[:p]] * discount)
    else:
        raise ValueError("Only supported for two formula, 'linear' or 'exp'")

    return dcg / idcg


if __name__ == "__main__":
    song_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8}
    user_lists = ["USER1", "USER2", "USER3"]

    relevance_true = {
        "USER1": [3, 3, 2, 2, 1, 1, 0, 0, 0],
        "USER2": [3, 2, 1, 1, 2, 0, 1, 1, 1],
        "USER3": [0, 1, 0, 1, 2, 3, 3, 1, 0]
    }

    s1_prediction = {
        "USER1": ['A', 'E', 'C', 'D', 'F'],
        "USER2": ['G', 'E', 'A', 'B', 'D'],
        "USER3": ['C', 'G', 'F', 'B', 'E']
    }

    s2_prediction = {
        "USER1": ['A', 'B', 'C', 'G', 'E'],
        "USER2": ['B', 'A', 'G', 'E', 'F'],
        "USER3": ['E', 'G', 'F', 'B', 'I']      
    }

    for user in user_lists:
        print(f'===={user}===')
        r_true = relevance_true[user]

        s1_pred = [r_true[song_index[song]] for song in s1_prediction[user]]
        s2_pred = [r_true[song_index[song]] for song in s2_prediction[user]]
   
        print(f'S1 nDCG@5 (linear): {ndcg(r_true, s1_pred, 5, "linear")}')
        print(f'S2 nDCG@5 (linear): {ndcg(r_true, s2_pred, 5, "linear")}')
        
        print(f'S1 nDCG@5 (exponential): {ndcg(r_true, s1_pred, 5, "exp")}')
        print(f'S2 nDCG@5 (exponential): {ndcg(r_true, s2_pred, 5, "exp")}')

