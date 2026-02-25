from typing import List, Dict
from pydantic import BaseModel


class Reply(BaseModel):
    id: str
    replyto: str
    reply_type: str
    content: str
    scores: Dict[str, float]

class Paper(BaseModel):
    title: str
    abstract: str
    authors: List[str]
    url: str
    pdf: str
    decision: str
    replies: List[Reply]
    novelty: float