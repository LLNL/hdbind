################################################################################
# Copyright (c) 2021-2024, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory.
# Written by D. Jones <djones@llnl.gov> and UCSD collaborators in listed in CONTRIB.md
#
# All rights reserved.
################################################################################
import numpy as np
from sklearn.metrics import recall_score, precision_score, roc_curve


def compute_enrichment_factor(scores, labels, n_percent):
    # this variant implements the equation from Xiaohua's paper

    sample_n = int(np.ceil(n_percent * labels.shape[0]))

    sorted_scores = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)

    top_n_sorted_scores = sorted_scores[:sample_n]

    # counting number of true positives in top x% of sorted compounds
    actives_sampled = sum([y for x, y in top_n_sorted_scores])
    # labels are binary, sum to count number of actives
    actives_database = sum(labels)

    return (actives_sampled / actives_database) * (labels.shape[0] / sample_n)


def compute_roc_enrichment(scores, labels, fpr_thresh):

    fpr, tpr, _ = roc_curve(y_true=labels, y_score=scores)

    er = np.interp(fpr_thresh, fpr, tpr) * 100

    return er

def validate(labels, pred_labels, pred_scores):
    n_correct = (pred_labels == labels).sum()
    n_labels = len(labels)
    # print(f"acc: {n_correct, n_labels, n_correct / float(n_labels) * 100}")
    # print(f"recall: {recall_score(y_pred=pred_labels, y_true=labels)}")

    precision = precision_score(y_pred=pred_labels, y_true=labels, zero_division=0)
    # print(f"precision: {precision}")

    random_precision = labels[labels == 1].shape[0] / labels.shape[0]

    # print(f"random precision {random_precision}")
    # print(f"FPDE: {precision/random_precision}")

    enrich_fact_1 = compute_enrichment_factor(
        scores=pred_scores, labels=labels, n_percent=0.01
    )
    print(f"enrichment-factor (EF) (1%): {enrich_fact_1}")
    enrich_fact_10 = compute_enrichment_factor(
        scores=pred_scores, labels=labels, n_percent=0.1
    )
    print(f"enrichment-factor (EF) (10%) enrichment: {enrich_fact_10}")

    er_1 = compute_roc_enrichment(scores=pred_scores, labels=labels, fpr_thresh=.01)

    print(f"roc-enrichment (ER-1%): {er_1}")
