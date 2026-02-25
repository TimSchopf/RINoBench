import re
from typing import List
from openreview import api, Client, Note
from datatypes import Paper


def find_all_strings(text, starts_with, ends_with):
    pattern = f'{re.escape(starts_with)}(.*?){re.escape(ends_with)}'
    return re.findall(pattern, text)

# Fetch both submissions and reviews for the venue in one API call and try different invitation names
def get_submissions(client, venue_id):
    invitation_types = ["Submission", "Blind_Submission", "submission", "blind_submission"]
    submission_types = ["default", "blind", "default", "blind"]
    
    for invitation_type, submission_type in zip(invitation_types, submission_types):
        submissions = client.get_all_notes(invitation=f'{venue_id}/-/{invitation_type}', details='replies')
        if submissions:
            return submissions, submission_type
        
    raise RuntimeError(f"{venue_id} did not return for any invitation type")


class OpenReviewV1Venue:
    """ Open Review
    """
    
    client = Client(baseurl="https://api.openreview.net")
    api_version = "v1"
    
    venue_id: str
    _submissions: List[Note]
    
    def __init__(self, venue: str):
        self.venue_id = venue

    @property
    def submissions(self):
        """
        Get OpenReview paper submissions for a given venue
        """
        # Cache submissions
        if hasattr(self, "_submissions"):
            return self._submissions

        # Fetch both submissions and reviews for the venue
        submissions, submission_type = get_submissions(self.client, self.venue_id)
            
        self._submissions = (submission_type, submissions)
        return submission_type, submissions
    
    def extract(self, submissions: List[api.client.Note]):
        papers = []
        
        for submission in submissions:
            paper = {
                'venueid': self.venue_id,
                'title': submission.content['title'],
                'abstract': submission.content['abstract'],
                'authors': submission.content['authors'],
                'url': (find_all_strings(submission.content['_bibtex'], "\nurl={", "}") or [""])[0] if '_bibtex' in submission.content else None,
                'pdf': "https://openreview.net" + submission.content['pdf'] if "pdf" in submission.content else None,
                'replies': [],
                'novelty': 0
            }

            try:
                paper['decision'] = submission.details['replies'][-1]['content']['decision']
            except:
                # withdrawn papers do not have an accept/reject decision
                paper['decision'] = None

            # get submission replies
            n_novelty_reviews = 0
            for submission_reply in submission.details['replies']:
                reply = {
                    'id': submission_reply['id'],
                    'replyto': submission_reply['replyto']
                }
                
                if 'review' in submission_reply['content'] or 'summary_of_the_review' in submission_reply['content']:
                    reply['reply_type'] = 'review'
                    
                    if "technical_novelty_and_significance" in submission_reply['content'] and \
                        "empirical_novelty_and_significance" in submission_reply['content']:
                        novlety_score = \
                            float((re.findall(r'\d+', str(submission_reply['content']['technical_novelty_and_significance']))or[0])[0]) + \
                            float((re.findall(r'\d+', str(submission_reply['content']['empirical_novelty_and_significance']))or[0])[0])
                        novlety_score /= 2
                        paper['novelty'] = ((paper['novelty'] * n_novelty_reviews) + novlety_score)/(n_novelty_reviews+1)
                        n_novelty_reviews += 1
                    
                elif 'decision' in submission_reply['content']:
                    reply['reply_type'] = 'metareview'
                elif 'comment' in submission_reply['content'] or 'reply' in submission_reply['content']: # comment but no decision (do not move above)
                    reply['reply_type'] = 'comment'
                else:
                    continue
                    # ignore replies which are not in this set ["review", "metareview", "comment"]

                reply['content'] = "\n".join([
                    f"{k}: {v}"
                    for k, v in submission_reply['content'].items()
                ])
                
                reply['scores'] = {
                    k: float(v)
                    for k, v in submission_reply['content'].items()
                    if str(v).isnumeric()
                }
                
                paper['replies'].append(reply)
                
            # validate paper model
            # Paper.model_validate(paper)
            papers.append(paper)
        
        return papers


class OpenReviewV2Venue:
    """ Open Review
    """
    
    client = api.OpenReviewClient(baseurl='https://api2.openreview.net')
    api_version = "v2"
    
    venue_id: str
    _submissions: List[api.client.Note]
    
    def __init__(self, venue: str):
        self.venue_id = venue

    @property
    def submissions(self):
        """
        Get OpenReview paper submissions for a given venue
        """
        # Cache submissions
        if hasattr(self, "_submissions"):
            return self._submissions

        # Fetch both submissions and reviews for the venue
        submissions, submission_type = get_submissions(self.client, self.venue_id)

        if not submissions and self.api_version == "v2":
            raise Warning("Incorrent API Version")

        self._submissions = (submission_type, submissions)
        return submission_type, submissions
    
    def extract(self, submissions: List[api.client.Note]):
        papers = []
        
        for submission in submissions:
            paper = {
                'venueid': self.venue_id,
                'title': submission.content['title']['value'],
                'abstract': submission.content['abstract']['value'],
                'authors': submission.content['authors']['value'],
                'url': (find_all_strings(submission.content['_bibtex']['value'], "\nurl={", "}") or [""])[0] if '_bibtex' in submission.content else None,
                'pdf': "https://openreview.net" + submission.content['pdf']['value'] if "pdf" in submission.content else None,
                'replies': [],
                'novelty': 0
            }

            try:
                paper['decision'] = submission.details['replies'][-1]['content']['decision']['value']
            except:
                # withdrawn papers do not have an accept/reject decision
                paper['decision'] = None

            # get submission replies
            n_novelty_reviews = 0
            for submission_reply in submission.details['replies']:
                reply = {
                    'id': submission_reply['id'],
                    'replyto': submission_reply['replyto']
                }

                # parse info from reviews
                if "summary" in submission_reply['content']:
                    reply['reply_type'] = 'review'
                    reply['content'] = "\n".join([
                        f"{k}: {v['value']}"
                        for k, v in submission_reply['content'].items() if k in \
                            [
                                'summary',
                                'strengths',
                                'weaknesses',
                                'questions',
                            ]
                    ])
                    
                    reply['scores'] = { 
                        # if only number then just that, otherwise first number in string
                        k: float(v['value']) if str(v).isnumeric() else float(re.findall(r'\d+', str(v['value']))[0])
                        for k, v in submission_reply['content'].items() if k in \
                            [
                                "rating",
                                "confidence",
                                "soundness",
                                "presentation",
                                "contribution",
                            ]
                    }
                    
                    if "contribution" in submission_reply['content']:
                        novlety_score = float((re.findall(r'\d+', str(submission_reply['content']["contribution"]["value"]))or[0])[0])
                        paper['novelty'] = ((paper['novelty'] * n_novelty_reviews) + novlety_score)/(n_novelty_reviews+1)
                        n_novelty_reviews += 1
                
                # parse metareview info
                elif "metareview" in submission_reply['content']:
                    reply['reply_type'] = 'metareview'
                    reply['content'] = submission_reply['content']['metareview']['value']
                
                # parse comments from author and reviewer responses
                elif all(key not in submission_reply['content'] for key in ['decision','desk_reject_comments','withdrawal_confirmation']):
                    reply['reply_type'] = 'comment'
                    
                    if "comment" in submission_reply['content']:
                        reply['content'] = submission_reply['content']['comment']['value']
                    else:
                        reply['content'] = "\n".join([
                            f"{k}: {v['value']}"
                            for k, v in submission_reply['content'].items()
                        ])
                    
                    # reply['comment'] = {
                    #     'value': submission_reply['content']['comment']['value'],
                    #     'comment_type': submission_reply['content']['title']['value'] if "title" in submission_reply['content'] else None, 
                    # }
                else:
                    continue
                    # ignore replies which are not in this set ["review", "metareview", "comment"]
                
                paper['replies'].append(reply)
            
            # validate paper model
            # Paper.model_validate(paper)
            papers.append(paper)
        
        return papers


class ICLR22_23(OpenReviewV1Venue):
    """
    ICLR 2023 custom extraction logic
    """

    def extract(self, submissions: List[api.client.Note]):
        papers = []

        for submission in submissions:
            paper = {
                'venueid': self.venue_id,
                'title': submission.content['title'],
                'abstract': submission.content['abstract'],
                'authors': submission.content['authors'],
                'url': (find_all_strings(submission.content['_bibtex'], "\nurl={", "}") or [""])[
                    0] if '_bibtex' in submission.content else None,
                'pdf': "https://openreview.net" + submission.content['pdf'] if "pdf" in submission.content else None,
                'replies': [],
                'mean_novelty': None
            }

            try:
                paper['decision'] = submission.details['replies'][-1]['content']['decision']
            except:
                # withdrawn papers do not have an accept/reject decision
                paper['decision'] = None

            # get submission replies
            novelty_scores = []
            for submission_reply in submission.details['replies']:
                reply = {
                    'id': submission_reply['id'],
                    'replyto': submission_reply['replyto']
                }

                if 'review' in submission_reply['content'] or 'summary_of_the_review' in submission_reply['content']:
                    reply['reply_type'] = 'review'

                    novelty_values = []
                    for key in ["technical_novelty_and_significance", "empirical_novelty_and_significance"]:
                        value = submission_reply['content'].get(key, "")
                        match = re.findall(r'\d+', str(value))
                        if match:
                            novelty_values.append(float(match[0]))

                    if novelty_values:
                        novelty_scores.extend(novelty_values)

                elif 'decision' in submission_reply['content']:
                    reply['reply_type'] = 'metareview'
                elif 'comment' in submission_reply['content'] or 'reply' in submission_reply[
                    'content']:  # comment but no decision (do not move above)
                    reply['reply_type'] = 'comment'
                else:
                    continue
                    # ignore replies which are not in this set ["review", "metareview", "comment"]

                reply['content'] = submission_reply['content']
                paper['mean_novelty'] =  sum(novelty_scores) / len(novelty_scores) if novelty_scores else None
                paper['replies'].append(reply)

            # validate paper model
            # Paper.model_validate(paper)
            papers.append(paper)

        return papers



def OpenReviewVenue(venue: str):
    """ Open Review
    """
    
    client = api.OpenReviewClient(baseurl='https://api2.openreview.net')
    venue_group = client.get_group(venue)

    if venue_group.id == "ICLR.cc/2023/Conference" or venue_group.id == "ICLR.cc/2022/Conference":
        return ICLR22_23(venue)
    elif venue_group.domain:
        return OpenReviewV2Venue(venue)
    else:
        return OpenReviewV1Venue(venue)
