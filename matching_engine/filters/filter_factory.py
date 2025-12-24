"""
Comprehensive filter factory - creates filter pipeline with all available filters
"""
from matching_engine.filters.pipeline.filter_pipeline import FilterPipeline
from matching_engine.filters.rules.age_filter import AgeFilter
from matching_engine.filters.rules.height_filter import HeightFilter
from matching_engine.filters.rules.gender_filter import GenderFilter
from matching_engine.filters.rules.politics_filter import PoliticsFilter
from matching_engine.filters.rules.religion_filter import ReligionFilter
from matching_engine.filters.rules.longterm_readiness_filter import LongTermReadinessFilter
from matching_engine.filters.rules.trait_matching_filter import TraitMatchingFilter
from matching_engine.filters.rules.children_filter import ChildrenFilter


def create_filters(bidirectional: bool = True, include_advanced: bool = True) -> FilterPipeline:
    """
    Create filter pipeline with all available filters
    
    Args:
        bidirectional: If True, age and height filters check both ways
        include_advanced: If True, include all advanced filters (politics, religion, etc.)
        
    Returns:
        FilterPipeline with comprehensive filtering
    """
    # Basic demographic filters (always included)
    filters = [
        GenderFilter(),               # Hetero matching (always one-way)
        AgeFilter(bidirectional),     # Age compatibility
        HeightFilter(bidirectional),  # Height compatibility
    ]

    # Advanced compatibility filters
    if include_advanced:
        filters.extend([
            LongTermReadinessFilter(),    # Remove indecisive "MOŻE" responses
            PoliticsFilter(),             # Political compatibility (left vs right)
            ReligionFilter(),             # Religion compatibility (Catholic rules)
            ChildrenFilter(),             # Children compatibility (complex rules)
            TraitMatchingFilter(),        # Friend descriptions match sought traits
        ])

    return FilterPipeline(filters)


def create_basic_filters(bidirectional: bool = True) -> FilterPipeline:
    """
    Create filter pipeline with only basic demographic filters
    
    Args:
        bidirectional: If True, age and height filters check both ways
        
    Returns:
        FilterPipeline with only gender + age + height filters
    """
    return create_filters(bidirectional=bidirectional, include_advanced=True)
