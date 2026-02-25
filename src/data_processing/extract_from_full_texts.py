#!/usr/bin/env python
# coding: utf-8
import os
import json
import glob
import re
from tqdm import tqdm


def post_ocr_correction(markdown_text):
    """
    Applies post-OCR corrections to markdown scientific text.

    Fixes:
    1. Removes all figure captions: '\n\nFigure X: ...\n\n'
    2. Fixes broken citations: 'Smith et al.\n\n(2021)' → 'Smith et al. (2021)'
    3. Removes false newlines/paragraphs within sentences, preserving sentence/paragraph boundaries

    Parameters:
        markdown_text (str): Raw OCR markdown.

    Returns:
        str: Cleaned markdown.
    """

    # 1. Remove all figure captions
    markdown_text = re.sub(
        r'\n{2}Figure\s+\d+:.*?(?=\n{2})',
        '',
        markdown_text,
        flags=re.IGNORECASE | re.DOTALL
    )

    # 2. Fix broken citations like "Smith et al.\n\n(2021)"
    markdown_text = re.sub(
        r'(\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+)?et al\.)\s*\n+\s*\((\d{4})\)',
        r'\1 (\2)',
        markdown_text
    )

    # Also fix broken citations with single author
    markdown_text = re.sub(
        r'(\b[A-Z][a-z]+)\s*\n+\s*\((\d{4})\)',
        r'\1 (\2)',
        markdown_text
    )

    # 3. Fix in-sentence newlines (flatten broken lines inside sentences)
    # Match: any word or punctuation not ending a sentence, followed by newline, followed by a lowercase word or mid-sentence punctuation
    markdown_text = re.sub(
        r'(?<![.\?!:;\n])\n+(?=\s*[a-z\(])',
        ' ',
        markdown_text
    )

    # 4. Also remove unnecessary multiple spaces caused by replacements
    markdown_text = re.sub(r' {2,}', ' ', markdown_text)

    return markdown_text.strip()


def extract_sections_with_subsections(markdown_text):
    """
    Extracts the 'Introduction', 'Related Work', and 'References' sections, including any subsections,
    from a markdown-formatted scientific paper.

    The function identifies top-level section headers (e.g., '## Introduction') and their 
    corresponding content blocks, then extracts all content within each section, including 
    any nested subsections (e.g., '### Motivation' under '## Introduction'), up until the 
    next top-level section.

    Parameters:
        markdown_text (str): The full markdown-formatted text of the paper.

    Returns:
        dict: A dictionary with keys 'introduction', 'related_work', and 'references', containing the 
              full markdown content of these sections, including nested subsections.
    """

    # Match all markdown section headers of level 2 and deeper
    header_pattern = re.compile(r'^(#{2,})\s+(.*)', re.MULTILINE)
    matches = list(header_pattern.finditer(markdown_text))

    # Build a list of section metadata
    section_boundaries = []
    for i, match in enumerate(matches):
        header_level = len(match.group(1))  # ## → level 2, ### → level 3, etc.
        title = match.group(2).strip().lower()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        section_boundaries.append({
            "level": header_level,
            "title": title,
            "start": start,
            "end": end
        })

    # Section keys to extract
    paper = {
        'introduction': None,
        'related_work': None,
        'references': None
    }

    def is_substring_match(title, substrings):
        """Returns True if any of the substrings is found within the title."""
        return any(sub in title for sub in substrings)

    # Substring patterns to match
    intro_keys = ['introduction']
    related_keys = ['related work', 'prior work', 'literature', 'background', 'relevant work', 'existing work']
    references_keys = ['references']

    # Iterate through section metadata and match based on substring
    for i, section in enumerate(section_boundaries):
        title = section["title"]

        # Check for Introduction
        if is_substring_match(title, intro_keys) and paper['introduction'] is None:
            intro_start = section["start"]
            intro_end = next((s["start"] for s in section_boundaries[i+1:] if s["level"] == 2), len(markdown_text))
            paper['introduction'] = markdown_text[intro_start:intro_end].strip()

        # Check for Related Work
        elif is_substring_match(title, related_keys) and paper['related_work'] is None:
            related_start = section["start"]
            related_end = next((s["start"] for s in section_boundaries[i+1:] if s["level"] == 2), len(markdown_text))
            paper['related_work'] = markdown_text[related_start:related_end].strip()

        # Check for References 
        elif is_substring_match(title, references_keys) and paper['references'] is None:
            references_start = section["start"]
            references_end = next((s["start"] for s in section_boundaries[i+1:] if s["level"] == 2), len(markdown_text))
            paper['references'] = markdown_text[references_start:references_end].strip()

    return paper


def process_mmd_files(directory):
    # Step 1: Verify directory exists and is valid
    directory = os.path.abspath(directory)  # Resolve to absolute path
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a directory.")
        return

    # Step 2: Find all .mmd files (case-insensitive, including subdirectories)
    mmd_files = glob.glob(os.path.join(directory, '**', '*.[mM][mM][dD]'), recursive=True)
    mmd_files += glob.glob(os.path.join(directory, '**', '.*.[mM][mM][dD]'), recursive=True)  # Hidden files
    mmd_files = [f for f in mmd_files if os.path.isfile(f)]

    print(f"Found {len(mmd_files)} .mmd files in '{directory}'")

    # Step 3: Read each file
    papers = []
    for file_path in tqdm(mmd_files):
        try:
            with open(file_path, 'r') as file:
                # parse specific paper sections into a paper dict
                content = file.read()
                content = post_ocr_correction(content)
                paper = extract_sections_with_subsections(content)
                paper['full_text'] = content
                paper['id'] = os.path.basename(file_path)
                papers.append(paper)
        except Exception as e:
            print(f"Error reading '{file_path}': {str(e)}")
    return papers


# postprocess .mmd files, extract sections, and save to disk
iclr22_papers = process_mmd_files("../data/mmd/ICLR.cc-2022-Conference")
iclr23_papers = process_mmd_files("../data/mmd/ICLR.cc-2023-Conference")

# Save as a JSON files
with open("../data/json/ICLR.cc-2022-Conference_processed_papers.json", 'w') as f:
    json.dump(iclr22_papers, f, indent=4)  # indent makes it easy to read

with open("../data/json/ICLR.cc-2023-Conference_processed_papers.json", 'w') as f:
    json.dump(iclr23_papers, f, indent=4)  # indent makes it easy to read


def find_invalid_dicts(dict_list):
    """
    Identifies indices of dictionaries in a list that:
    1. Do not have the same set of keys as the first dictionary.
    2. Contain any empty (falsy) values for required keys.

    Parameters:
        dict_list (list of dict): The list of dictionaries to check.

    Returns:
        list: Indices of invalid dictionaries.
    """
    if not dict_list:
        return []  # No dictionaries to validate

    # Use the first dictionary to define the required key set
    expected_keys = set(dict_list[0].keys())
    invalid_indices = []

    for idx, d in enumerate(dict_list):
        current_keys = set(d.keys())
        if current_keys != expected_keys:
            invalid_indices.append(idx)
            continue

        # Check for any falsy (empty) value
        for key in expected_keys:
            if not d[key]:  # Catches None, '', [], {}, 0, etc.
                invalid_indices.append(idx)
                break

    return invalid_indices

