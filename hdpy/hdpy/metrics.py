import numpy as np
from sklearn.metrics import recall_score, precision_score


def compute_top_n_enrichment(scores, labels, n, n_samples=10):

    '''
    scores: numpy array (vector) of scores. 1:1 correspondence (position) with labels
    labels: ground truth labels that correspond to scores
    n: the value n for top-n rankings    
    '''


    sorted_scores = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)

    top_n_sorted_scores = sorted_scores[:n]


    # counting number of true positives
    score_tp = sum([y for x, y in top_n_sorted_scores])


    score_list = []
    for i in range(n_samples):

        random_tp = sum(np.random.choice(a=labels.squeeze(), size=n))

        score = score_tp / random_tp
        score_list.append(score)

    return np.mean(score_list), np.std(score_list)


# def compute_enrichment_factor(scores, labels, n):

    # sorted_scores = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)

    # top_n_sorted_scores = sorted_scores[:n]

    # counting number of true positives
    # score_tp = sum([y for x, y in top_n_sorted_scores])
    # labels are binary, sum to count number of actives
    # actives_database = sum(labels)

    # norm_factor = (n/len(labels))

    # return score_tp / (norm_factor * actives_database)
def compute_enrichment_factor(scores, labels, n_percent):
    # this variant implements the equation from Xiaohua's paper
    
    sample_n = int(np.ceil(n_percent * labels.shape[0]))
    
    sorted_scores = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)

    top_n_sorted_scores = sorted_scores[:sample_n]

    # counting number of true positives in top x% of sorted compounds
    actives_sampled = sum([y for x, y in top_n_sorted_scores])
    # labels are binary, sum to count number of actives
    actives_database = sum(labels)

    norm_factor = (sample_n/len(labels))

    return (actives_sampled / actives_database) * (labels.shape[0]/sample_n)


def validate(labels, pred_labels, pred_scores):
    n_correct = (pred_labels == labels).sum()
    n_labels = len(labels)
    print(f"acc: {n_correct, n_labels, n_correct / float(n_labels) * 100}")
    print(f"recall: {recall_score(y_pred=pred_labels, y_true=labels)}")


    precision = precision_score(y_pred=pred_labels, y_true=labels)
    print(f"precision: {precision}")

    random_precision = labels[labels == 1].shape[0] / labels.shape[0]

    print(f"random precision {random_precision}")
    print(f"FPDE: {precision/random_precision}")

    # import pdb 
    # pdb.set_trace()
    top_10_enrichment = compute_top_n_enrichment(scores=pred_scores, labels=labels, n=10)
    print(f"top-10 enrichment: {top_10_enrichment[0]} +/- ({top_10_enrichment[1]})")
    top_100_enrichment = compute_top_n_enrichment(scores=pred_scores, labels=labels, n=100)
    print(f"top-100 enrichment: {top_100_enrichment[0]} +/- ({top_100_enrichment[1]})")
    top_1000_enrichment = compute_top_n_enrichment(scores=pred_scores, labels=labels, n=1000)
    print(f"top-1000 enrichment: {top_1000_enrichment[0]} +/- ({top_1000_enrichment[1]})")



    enrich_fact_10 = compute_enrichment_factor(scores=pred_scores, labels=labels, n=10)
    print(f"enrichment-factor (EF) (10): {enrich_fact_10}")
    enrich_fact_100 = compute_enrichment_factor(scores=pred_scores, labels=labels, n=100)
    print(f"enrichment-factor (EF) (100) enrichment: {enrich_fact_100}")
    enrich_fact_1000 = compute_enrichment_factor(scores=pred_scores, labels=labels, n=1000)
    print(f"enrichment-factor (EF) (1000): {enrich_fact_1000}")
