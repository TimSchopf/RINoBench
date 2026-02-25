from pydantic import BaseModel, Field
from typing import List

# ---------------------------
# For aspects about similarities/differences with existing works
# ---------------------------

class RelatedWorkAspect(BaseModel):
    """An aspect or comparison point extracted from reasoning that describes similarities or differences with existing works."""
    aspect: str = Field(
        ...,
        description="The text of the identified aspect referring to similarities or differences with existing works."
    )
    supported_in_gold: bool = Field(
        ...,
        description="Whether this aspect is explicitly or implicitly included in the gold output."
    )

class RelatedWorkComparison(BaseModel):
    """Complete structured evaluation of predicted vs. gold reasoning outputs focusing on related work aspects."""
    predicted_aspects: List[RelatedWorkAspect] = Field(
        ..., description="List of related work aspects extracted from the predicted output with support check against gold."
    )
    gold_aspects: List[str] = Field(
        ..., description="List of related work aspects extracted from the gold output."
    )


# ---------------------------
# For aspects about novelty
# ---------------------------

class NoveltyAspect(BaseModel):
    """An aspect or point extracted from reasoning that highlights novelty, originality, or innovation."""
    aspect: str = Field(
        ...,
        description="The text of the identified aspect highlighting novelty, originality, or innovation."
    )
    supported_in_gold: bool = Field(
        ...,
        description="Whether this novelty aspect is explicitly or implicitly included in the gold output."
    )

class NoveltyComparison(BaseModel):
    """Complete structured evaluation of predicted vs. gold reasoning outputs focusing on novelty aspects."""
    predicted_aspects: List[NoveltyAspect] = Field(
        ..., description="List of novelty aspects extracted from the predicted output with support check against expected."
    )
    gold_aspects: List[str] = Field(
        ..., description="List of novelty aspects extracted from the gold output."
    )

# ---------------------------
# Fact checking models
# ---------------------------

class FactCheckClaim(BaseModel):
    #claim: str = Field(..., description="The text of the extracted claim.")
    supported: bool = Field(
        ...,
        description="True if the claim is supported, False otherwise."
    )

class FactCheckResult(BaseModel):
    checked_aspects: List[FactCheckClaim] = Field(
        ..., description="List of fact-check claims."
    )
