import json
from pathlib import Path
from upycli import command

from venues import OpenReviewVenue
from openreview_postprocessing import compute_binned_novelties, scrape_pdfs

VENUES = [
    # ICLR
    OpenReviewVenue('ICLR.cc/2023/Conference'),
    OpenReviewVenue('ICLR.cc/2022/Conference'),
]


@command
def scrape(save_to: str):
    for venue in VENUES:
        _, submissions = venue.submissions
        papers = venue.extract(submissions)

        with open(Path(save_to) / f"{venue.venue_id.replace('/', '-')}.json", "w") as f:
            json.dump(papers, f, indent=4)


if __name__ == "__main__":
    scrape("../data/json")
    compute_binned_novelties("../data/json/ICLR.cc-2023-Conference.json")
    compute_binned_novelties("../data/json/ICLR.cc-2022-Conference.json")
    scrape_pdfs(json_path="../data/json/ICLR.cc-2023-Conference.json", save_path="../data/pdf")
    scrape_pdfs(json_path="../data/json/ICLR.cc-2022-Conference.json", save_path="../data/pdf")