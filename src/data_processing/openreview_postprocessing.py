import json
import pandas as pd
import os
import subprocess
from urllib.request import urlretrieve
from tqdm import tqdm

def compute_binned_novelties(json_path: str):
    """
    Reads a JSON file containing a list of papers, each with a 'mean_novelty' score,
    and assigns each paper to one of five equal-width bins based on that score.

    The bin assignments (ranging from 1 to 5) are stored in a new field called 'novelty_bin'
    in each paper's dictionary. The JSON file is then overwritten with the updated data.

    Parameters:
        json_path (str): Path to the input JSON file containing a list of paper dictionaries.
                         Each dictionary must include a 'mean_novelty' key with a numeric value.

    Returns:
        None
    """

    # Load the list of papers from a JSON file
    with open(json_path, 'r') as file:
        papers = json.load(file)

    # Extract all the mean novelty scores from the papers
    mean_novelty_scores = []
    for paper in papers:
        mean_novelty_scores.append(paper['mean_novelty'])

    # Determine the range of novelty scores
    min_novelty = min(mean_novelty_scores)
    max_novelty = max(mean_novelty_scores)

    # Create 5 equal-width bins based on the novelty score range
    step = (max_novelty - min_novelty) / 5 # get 5 equal-width bins
    bins = [min_novelty + i * step for i in range(6)]  # 6 points for 5 bins
    bin_labels = [1, 2, 3, 4, 5]

    # Assign each novelty score to a bin using pandas.cut
    bin_indices = pd.cut(mean_novelty_scores, bins, labels=bin_labels, include_lowest=True, right=True)

    # Add the bin index to each paper's data
    for i, paper in enumerate(papers):
        paper['novelty_bin'] = int(bin_indices[i])

    # Save the updated list of papers back to the JSON file
    with open(json_path, 'w') as file:
        json.dump(papers, file, indent=4)



def scrape_pdfs(json_path: str, save_path: str):
    """
    Downloads PDF files for papers listed in a JSON file and saves them into a directory
    named after the venue (derived from the input filename).

    Parameters:
        json_path (str): Path to a JSON file containing a list of papers with 'pdf' fields.
        save_path (str): Path to the directory where the PDFs should be saved. A subdirectory
                         with the venue name (from the JSON filename) will be created here.

    Returns:
        None
    """

    # Load the list of papers from a venue from the JSON file
    with open(json_path, 'r') as file:
        papers = json.load(file)

    # Extract the venue name from the JSON filename (e.g., "ICLR2023" from "ICLR2023.json")
    venue_name = json_path.split('/')[-1].split('.json')[0]

    # Create the full save path including the venue subdirectory
    save_path = save_path + "/" + venue_name

    # Create the directory if it doesn't already exist
    os.makedirs(save_path, exist_ok=True)

    # Download the paper PDFs
    for paper in tqdm(papers):
        pdf_url = paper['pdf']  # Get the PDF URL
        pdf_name = pdf_url.split('/')[-1]  # Use the last part of the URL as the filename
        urlretrieve(pdf_url, f"{save_path}/{pdf_name}")  # Download and save the PDF