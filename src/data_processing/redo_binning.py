#!/usr/bin/env python
# coding: utf-8

import json
from collections import Counter
import pandas as pd
import numpy as np

with open('../data/json/ICLR.cc-2022-Conference_processed_papers.json', 'r') as f:
    iclr22_papers: dict = json.load(f)

with open('../data/json/ICLR.cc-2023-Conference_processed_papers.json', 'r') as f:
    iclr23_papers: dict = json.load(f)

with open('../data/json/ICLR.cc-2022-Conference.json', 'r') as f:
    iclr22_reviews: dict = json.load(f)

with open('../data/json/ICLR.cc-2023-Conference.json', 'r') as f:
    iclr23_reviews: dict = json.load(f)

with open('../data/json/ICLR.cc-2022-Conference-research-ideas.json', 'r') as f:
    iclr22_research_ideas: dict = json.load(f)

with open('../data/json/ICLR.cc-2023-Conference-research-ideas.json', 'r') as f:
    iclr23_research_ideas: dict = json.load(f)

with open('../data/json/ICLR.cc-2022-Conference-related-works.json', 'r') as f:
    iclr22_related_works: dict = json.load(f)

with open('../data/json/ICLR.cc-2023-Conference-related-works.json', 'r') as f:
    iclr23_related_works: dict = json.load(f)


def merge_data(papers, reviews, ideas, related_works):
    dicts = []
    for e in range(len(reviews)):
        d = {}
        d['reviews'] = reviews[e]
        d['research_idea'] = ideas[e]
        idt = reviews[e]['pdf'].split('https://openreview.net/pdf/')[1].split('.pdf')[0]
        idx = [n for n, p in enumerate(papers) if p['id'] == idt + ".mmd"][0]
        d['related_works']= related_works[idx]

        dicts.append(d)

    return dicts
iclr22_merged = merge_data(papers=iclr22_papers, reviews=iclr22_reviews, ideas=iclr22_research_ideas, related_works=iclr22_related_works)
iclr23_merged = merge_data(papers=iclr23_papers, reviews=iclr23_reviews, ideas=iclr23_research_ideas, related_works=iclr23_related_works)


Counter([p['reviews']['novelty_bin'] for p in iclr22_merged])
Counter([p['reviews']['novelty_bin'] for p in iclr23_merged])

def novelty_consistent(dict):
    """
    Determines if reviewers' novelty judgments are consistent.

    Args:
        review_dict (dict): A dictionary representing the paper and its reviews.

    Returns:
        bool: True if reviewers' novelty scores differ by at most 1, else False.
    """
    tech_scores = []
    emp_scores = []
    review_dict = dict['reviews']
    for reply in review_dict.get("replies", []):
        if reply["reply_type"] != "review":
            continue
        content = reply.get("content", {})

        # Technical novelty
        tech = content.get("technical_novelty_and_significance", "")
        if tech:
            try:
                tech_score = int(tech.split(":")[0].strip())
                tech_scores.append(tech_score)
            except ValueError:
                pass  # skip non-numeric or N/A

        # Empirical novelty
        emp = content.get("empirical_novelty_and_significance", "")
        if emp:
            try:
                emp_score = int(emp.split(":")[0].strip())
                emp_scores.append(emp_score)
            except ValueError:
                pass  # skip non-numeric or N/A

    # Check if differences exceed 1filtered out papers
    if tech_scores and (max(tech_scores) - min(tech_scores) > 1):
        return False
    if emp_scores and (max(emp_scores) - min(emp_scores) > 1):
        return False
    if tech_scores and emp_scores and (max(emp_scores+tech_scores) - min(emp_scores+tech_scores) > 1):
        return False

    return True

iclr22_merged = [p for p in iclr22_merged if novelty_consistent(p)]
iclr23_merged = [p for p in iclr23_merged if novelty_consistent(p)]


def compute_binned_novelties(papers: list, quantile_mix: float = 0.7, n_bins: int = 5):
    """
    Assigns each paper to one of several bins based on its 'mean_novelty' score,
    using a hybrid of equal-width and equal-frequency (quantile) binning.

    Parameters:
        papers (list[dict]): List of papers, each with a 'mean_novelty' key.
        quantile_mix (float): Degree to favor quantile-based bins (0.0–1.0).
                             0.0 = pure equal-width, 1.0 = pure quantile.
        n_bins (int): Number of bins (default 5).

    Returns:
        dict:
            {
                "papers": list of updated papers,
                "bin_edges": array of bin boundaries,
                "bin_counts": Counter of bin label frequencies
            }
    """

    # --- Step 1: Collect novelty scores ---
    mean_novelty_scores = [p['reviews']['mean_novelty'] for p in papers]
    min_novelty, max_novelty = min(mean_novelty_scores), max(mean_novelty_scores)

    # --- Step 2: Compute equal-width and quantile bin edges ---
    width_bins = np.linspace(min_novelty, max_novelty, n_bins + 1)
    quantile_edges = np.quantile(mean_novelty_scores, np.linspace(0, 1, n_bins + 1))

    # --- Step 3: Blend them ---
    blended_bins = quantile_mix * quantile_edges + (1 - quantile_mix) * width_bins

    # --- Step 4: Assign bins ---
    bin_labels = list(range(1, n_bins + 1))
    bin_indices = pd.cut(mean_novelty_scores, blended_bins, labels=bin_labels, include_lowest=True)

    # --- Step 5: Add to papers ---
    for i, paper in enumerate(papers):
        paper['reviews']['novelty_bin'] = int(bin_indices[i])

    # --- Step 6: Summarize results ---
    counts = Counter(int(x) for x in bin_indices)

    return {
        "papers": papers,
        "bin_edges": blended_bins,
        "bin_counts": counts
    }



binned_papers = compute_binned_novelties(iclr23_merged+iclr22_merged, quantile_mix=0.7, n_bins=5)

