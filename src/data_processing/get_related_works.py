#!/usr/bin/env python
# coding: utf-8

import json
import requests
import re
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm 
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# load data from disk
with open('../data/json/ICLR.cc-2022-Conference_processed_papers.json', 'r') as f:
    iclr22_papers: dict = json.load(f)

with open('../data/json/ICLR.cc-2023-Conference_processed_papers.json', 'r') as f:
    iclr23_papers: dict = json.load(f)

with open('../data/json/ICLR.cc-2022-Conference.json', 'r') as f:
    iclr22_reviews: dict = json.load(f)

with open('../data/json/ICLR.cc-2023-Conference.json', 'r') as f:
    iclr23_reviews: dict = json.load(f)


# load ScaDS.AI API key from file
api_key = ""
path_to_key = os.path.join(os.path.expanduser("~"), ".scadsai-api-key")
if os.path.exists(path_to_key):
    with open(path_to_key) as keyfile:
        api_key = keyfile.readline().strip()
if len(api_key) < 1:
    print("Error: The key file '.scadsai-api-key' did not contain any key. Please make sure the file exists and contains only your API key.")
    exit(1)


# Initialize the OpenAI client with instructor patching
api_key = ""
llm = "gpt-5-nano-2025-08-07"
client = instructor.from_openai(OpenAI(api_key=api_key), mode=instructor.Mode.JSON)


def extract_cited_references(sections_dict):
    """
    Extracts references from introduction and related work sections
    that are actually cited and present in the references list.
    Handles multiple citation formats like:
      - Author et al. (2017)
      - McCloskey & Cohen, 1989
      - Li and Hoiem, 2016
      - Liu et al., 2020; 2021
    """

    # Sections to scan for citations
    intro_text = sections_dict.get("introduction")
    related_text = sections_dict.get("related_work")
    text_sections = ""
    if intro_text:
        text_sections += intro_text + "\n"
    if related_text:
        text_sections += related_text
    references_text = sections_dict.get("references", "")

    # Ensure references_text is a string
    if not references_text:
        return []

    else:

        # Regex pattern 1: (Author et al., 2017)
        pattern_parentheses = re.compile(r"([A-Z][A-Za-z\-]+(?:\s+et al\.)?\s*\(\d{4}\))")

        # Regex pattern 2: Author & Author, 2017 or Li and Hoiem, 2016 etc.
        pattern_inline = re.compile(
            r"([A-Z][A-Za-z\-]+(?:\s+(?:et al\.|and|&|[A-Z][A-Za-z\-]+))*[,\s]*\d{4})"
        )

        # Collect citations from both patterns
        citations = set(re.findall(pattern_parentheses, text_sections))
        citations.update(re.findall(pattern_inline, text_sections))

        # Normalize citations
        citations = {c.strip() for c in citations}

        # Extract full references from references section
        ref_pattern = re.compile(r"\*\s*(.+?)\n")
        all_references = re.findall(ref_pattern, references_text)

        # Match citations to references
        matched_references = []
        if all_references:
            for ref in all_references:
                for cit in citations:
                    # Handle cases with parentheses vs commas
                    if "(" in cit:
                        author_part, year_part = cit.split("(")
                        year_part = year_part.strip(")")
                    else:
                        if "," in cit:
                            author_part, year_part = cit.rsplit(",", 1)
                        else:
                            continue
                    author_part = author_part.strip()
                    year_part = year_part.strip()

                    if author_part.split()[0] in ref and year_part in ref:
                        matched_references.append(ref.strip())
                        break

        return matched_references


iclr22_references = []
for i, paper in enumerate(iclr22_papers):
    #print(i)
    iclr22_references.append(extract_cited_references(paper))

def extract_paper_title(citation):
    """
    Extract the paper title from a citation by:
    1. Removing the author list iteratively
    2. Returning everything up to the first period, question mark, or comma followed by pp. or a year
    
    Args:
        citation (str): Full citation starting with authors.
        
    Returns:
        str: Paper title only
    """
    remaining = citation
    
    # Step 1: Remove author list iteratively
    while True:
        idx = remaining.find('.')
        if idx == -1:
            title_candidate = remaining.strip()
            break
        
        left = remaining[:idx+1].strip()
        right = remaining[idx+1:].strip()
        
        # Continue if left ends with 'et al.' or a single initial
        if left.endswith('et al.') or (len(left) >= 2 and left[-2].isupper() and left[-1] == '.'):
            remaining = right
        else:
            title_candidate = right
            break
    
    # Step 2: Extract title considering '.', '?', or ',' followed by year/pp.
    # Pattern explanation:
    # - Match up to the first '?'
    # - Or up to the first '.' 
    # - Or up to a ',' followed by space and either 'pp.' or 4 digits
    match = re.search(r'^(.*?)(\?|\.|,(?:\s*(?:pp\.|\d{4})))', title_candidate)
    
    if match:
        title = match.group(1).strip()
        # Include question mark if that was the match
        if match.group(2) == '?':
            title += '?'
    else:
        title = title_candidate.strip()
    
    return title# Example usage
#now, sometimes the paper title can be followed by a question mark instead of a full stop or followed by a comma that is followed by a month (written, not numerical) and year, year directly or pp. In cases of question marks, they should be included in the title
for c in iclr23_references[2]:
    print(extract_paper_title(c))
# In[7]:


class PaperTitle(BaseModel):
    """
    Structured output from LLM: the exact paper title extracted from a reference.
    Example JSON: {"title": "Neural machine translation by jointly learning to align and translate"}
    """
    title: str = Field(
        ...,
        description=(
            'The exact paper title extracted from the reference.'
        )
    )

class VerifyPaperOutput(BaseModel):
    """
    Structured output from the LLM for verifying if reference matches paper metadata.
    Example JSON: {"match": True}
    """
    match: bool = Field(
        ...,
        description=(
            'Indicates whether the reference matches paper metadata. True if both represent teh same paper, else False.'
        )
    )


def extract_paper_title(reference: str) -> PaperTitle:
    """
    Uses LLM to extract the exact paper title from a reference string.
    """
    system_prompt = (
        "You are an assistant that extracts the **exact** paper title from a reference. "
        "Return JSON with a single key 'title'. Do not include venue, authors, or year."
    )
    user_prompt = f"Reference: {reference}\nOutput JSON:"

    # LLM call
    messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    result = client.chat.completions.create(
        model=llm,
        response_model=PaperTitle, 
        messages=messages#,
        #temperature=0
    )

    return result.title


extract_paper_title("Baker et al. (2019) Jack Baker, Paul Fearnhead, Emily B Fox, and Christopher Nemeth. Control variates for stochastic gradient mcmc. _Statistics and Computing_, 29(3):599-615, 2019.")


def search_paper_by_title(title: str,
                          limit: int = 3,
                          max_retries: int = 10,
                          backoff_factor: float = 3.0) -> Dict[str, Any]:
    """
    Query Semantic Scholar for papers whose title matches ``title`` and
    return the first simplified result.

    Each result contains only:
        - title   : str
        - abstract: str (may be empty)
        - authors : List[str]   # just the author names
        - year    : int
        - venue   : str (may be empty)
        - url     : str

    Parameters
    ----------
    title : str
        The title (or a keyword phrase) to search for.
    limit : int, optional (default=3)
        Maximum number of results to return.
    max_retries : int, optional (default=3)
        Number of times to retry in case of rate limit (429) or transient errors.
    backoff_factor : float, optional (default=2.0)
        Multiplier for exponential backoff delay between retries.

    Returns
    -------
    Dict[str, Any]
        A dictionary with paper details, or {} if nothing found.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search/match"
    params = {
        "query": title,
        "limit": limit,
        "fields": "title,abstract,authors,year,venue,url"
    }

    for attempt in range(max_retries):
        resp = requests.get(
            url,
            headers={'x-api-key': "gBU0VL0TGj9u5bhv2MWFz8pVnEqFgCwpsxacR980"},
            params=params,
            timeout=10
        )

        if resp.status_code == 404:
            try:
                payload = resp.json()
                if payload.get("error") == "Title match not found":
                    return {}
            except ValueError:
                return {}
            # If it's a 404 but not "Title match not found", don't retry
            break

        if resp.status_code == 429:  # Rate limit
            sleep_time = backoff_factor * (2 ** attempt)
            time.sleep(sleep_time)
            continue

        if 500 <= resp.status_code < 600:  # Server errors
            sleep_time = backoff_factor * (2 ** attempt)
            time.sleep(sleep_time)
            continue

        if resp.status_code != 200:
            raise RuntimeError(
                f"Semantic Scholar API error {resp.status_code}: {resp.text}"
            )

        data = resp.json().get("data", [])
        if not data:
            return {}

        first = data[0]
        return {
            "title": first.get("title", ""),
            "abstract": first.get("abstract", ""),
            "year": first.get("year", ""),
            "venue": first.get("venue", ""),
            "authors": [a.get("name", "") for a in first.get("authors", [])],
            "url": first.get("url", "")
        }

    raise RuntimeError("Failed to fetch data after retries.")



x = search_paper_by_title(title="Control variates for stochastic gradient MCMC")


def verify_paper_match(reference: str, paper_metadata: dict) -> bool:
    """
    Uses OpenAI API to check if reference and metadata represent the same paper.
    Allows small differences in authors, publisher, and year.
    """
    system_prompt = (
        "You are a smart bibliographic verifier. "
    )

    task_prompt = ("Given a paper reference string and paper metadata, determine if they represent the same paper. "
        "Consider: "
        "1. Author names may differ slightly in variations as some might use initials and others don't or use special characters. "
        "2. Reasonable publisher/venue differences (e.g., arXiv vs. conference) are acceptable. "
        "3. Year may differ. "
        "If the metadata is empty, return False"
        "Return only 'True' or 'False'."

        "Example:"
        "Reference: Baker et al. (2019) Jack Baker, Paul Fearnhead, Emily B Fox, and Christopher Nemeth. Control variates for stochastic gradient mcmc. _Statistics and Computing_, 29(3):599-615, 2019.\n"
        """Metadata: {'title': 'Control variates for stochastic gradient MCMC',
          'year': 2017,
          'venue': 'Statistics and computing',
          'authors': ['Jack Baker', 'P. Fearnhead', 'E. Fox', 'C. Nemeth']}"
        """

        "True"
                  )

    metadata = {k: x[k] for k in ['title', 'year', 'venue', 'authors']}

    user_prompt = task_prompt + "\n" + f"Reference: {reference}\nMetadata:\n{metadata}\nAre these the same paper?"

    response = client.chat.completions.create(
        model=llm,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_model=VerifyPaperOutput#,
        #temperature=0
    )

    return response.match



verify_paper_match(reference="Baker et al. (2019) Jack Baker, Paul Fearnhead, Emily B Fox, and Christopher Nemeth. Control variates for stochastic gradient mcmc. _Statistics and Computing_, 29(3):599-615, 2019.", paper_metadata=x)


# ----------------------------------------------------------------------
# 1️⃣  Worker that does the whole “extract → search → verify” pipeline
# ----------------------------------------------------------------------
def _process_one_reference(p: Any) -> Dict[str, Any] | None:
    """
    Returns the metadata dict for *p* if it passes all checks,
    otherwise returns None.
    """
    try:
        # 1️⃣ Extract title from the raw reference entry
        paper_title = extract_paper_title(p)

        # 2️⃣ Query Semantic‑Scholar (our function now returns {} on 404)
        paper_metadata = search_paper_by_title(title=paper_title)

        # 3️⃣ Verify the match **and** make sure an abstract exists
        if (
            paper_metadata                # {} → falsy
        #    #and verify_paper_match(reference=p, paper_metadata=paper_metadata)
            and paper_metadata.get("abstract")
        ):
            return paper_metadata
        #return paper_title
    except Exception as exc:
        # You can log the exception if you wish; the thread should never crash the whole program.
        # Example: logging.warning(f"Failed processing reference {p}: {exc}")
        pass

    return None


# ----------------------------------------------------------------------
# 2️⃣  Parallel driver that uses ThreadPoolExecutor + tqdm
# ----------------------------------------------------------------------
def collect_related_work_titles_parallel(paper_references: List[Any],
                                   max_workers: int = 8,
                                   show_progress_bar: bool = True) -> List[Dict[str, Any]]:
    """
    Run the reference‑search pipeline in parallel while preserving the order
    of ``paper_references``.

    Parameters
    ----------
    paper_references : list
        Iterable of reference objects (e.g. the list you previously iterated
        over with ``for p in paper_references[20]:``).

    max_workers : int, optional (default = 8)
        Number of threads to use. Tune this to your CPU / API rate‑limit.

    show_progress : bool, optional (default = True)
        Whether to display a tqdm progress bar.

    Returns
    -------
    list[dict]
        Metadata dictionaries for the references that passed verification,
        **in the same order as the input** (minus any that were filtered out).
    """
    # Pre‑allocate a list the same length as the input – each slot will hold
    # either a dict (successful result) or None (failed / filtered out).
    ordered_results: List[Optional[Dict[str, Any]]] = [None] * len(paper_references)

    # Map each submitted Future to the index of the reference it belongs to.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_process_one_reference, ref): idx
            for idx, ref in enumerate(paper_references)
        }

        # Use tqdm only if the caller asked for it.
        iterator = as_completed(future_to_index)
        if show_progress_bar:
            iterator = tqdm(
                iterator,
                total=len(future_to_index),
                desc="Processing references",
                leave=False,
            )

        for fut in iterator:
            idx = future_to_index[fut]          # original position of this job
            try:
                result = fut.result()           # re‑raise any exception from the worker
            except Exception:
                # If something unexpected happened inside the worker we treat it
                # as a failed job – keep the slot as None.
                result = None

            if result is not None:
                ordered_results[idx] = result   # store exactly where it belongs

    # Remove all the Nones while preserving order.
    return [res for res in ordered_results if res is not None]


iclr22_related_works = []
for r in tqdm(iclr22_references):
    iclr22_related_works.append(collect_related_work_titles_parallel(r, max_workers=10, show_progress_bar=False))

# Save as a JSON files
with open("../data/json/ICLR.cc-2022-Conference-related-works.json", 'w') as f:
    json.dump(iclr22_related_works, f, indent=4)  # indent makes it easy to read

