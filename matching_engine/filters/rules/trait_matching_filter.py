"""
Trait matching filter - UNIVERSAL RULE for perfect symmetric scoring
ALWAYS checks: Does the WOMAN have traits that the MAN seeks?
NEVER checks: Does the MAN have traits that the WOMAN seeks?
"""
from typing import List, Set
from matching_engine.filters.interfaces.base_filter import BaseFilter, FilterResult
from commons.enums import Gender


class TraitMatchingFilter(BaseFilter):
    """
    Universal trait compatibility filter for PERFECT SYMMETRIC MATCHING SCORES
    
    BUSINESS RULE: For any pair (Man, Woman) - ALWAYS apply the same logic:
    - Does the WOMAN have ≥1 trait that the MAN seeks?
    - Woman's trait preferences are IGNORED for filtering
    
    This ensures PERFECT symmetry:
    - Man A seeking Woman B: Does B have what A seeks?
    - Woman B seeking Man A: Does B have what A seeks? (SAME CHECK!)
    
    Result: If Man A rates Woman B at 60%, then Woman B rates Man A at 60% too!
    
    Examples:
    - Man seeks "ambitna" → Woman's friends say "ambitna" ✅ → MATCH
    - Same pair, woman's perspective: Still checks if woman has what man seeks ✅ → MATCH
    """
    
    def __init__(self, min_trait_matches: int = 1):
        # Bidirectional filter: both directions apply trait matching
        super().__init__("trait_matching_filter", bidirectional=True)
        self.min_trait_matches = min_trait_matches
    
    def apply(self, target_profile, candidates: List) -> FilterResult:
        """
        Apply UNIVERSAL trait matching for perfect symmetry
        
        BUSINESS RULE: For any pair (Man, Woman) - ALWAYS check the same thing:
        - Does the WOMAN have traits that the MAN seeks?
        - NEVER check what woman seeks from man
        
        This creates perfect symmetry:
        - Man seeking Woman: Does woman have what man seeks? ✅
        - Woman seeking Man: Does woman have what man seeks? ✅ (SAME CHECK!)
        """
        passed = []
        rejected = []
        
        for candidate in candidates:
            # Determine who is the man and who is the woman in this pair
            if target_profile.gender == Gender.MALE and candidate.gender == Gender.FEMALE:
                # Target is man, candidate is woman
                man_profile = target_profile
                woman_profile = candidate
            elif target_profile.gender == Gender.FEMALE and candidate.gender == Gender.MALE:
                # Target is woman, candidate is man  
                man_profile = candidate
                woman_profile = target_profile
            else:
                # Same gender pair (shouldn't happen due to gender filter) - skip
                passed.append(candidate)
                continue
            
            # UNIVERSAL RULE: Does woman have traits that man seeks?
            man_seeks = self._normalize_traits(man_profile.ideal_partner_traits)
            woman_traits = self._normalize_traits(woman_profile.friend_described_traits)
            
            if not man_seeks:
                # Man has no trait preferences - woman passes
                passed.append(candidate)
                continue
                
            if not woman_traits:
                # Woman has no friend descriptions - reject (need data)
                rejected.append(candidate)
                continue
            
            # Check: Does woman have what man seeks?
            matching_traits = man_seeks.intersection(woman_traits)
            
            if len(matching_traits) >= self.min_trait_matches:
                passed.append(candidate)
            else:
                rejected.append(candidate)
        
        return FilterResult(
            passed=passed, 
            rejected=rejected, 
            reason=f"universal_trait_matching_min_{self.min_trait_matches}"
        )
    
    def _normalize_traits(self, traits: List[str]) -> Set[str]:
        """Normalize trait strings for EXACT 1:1 matching only"""
        if not traits:
            return set()
        
        normalized = set()
        for trait in traits:
            if trait:
                # ONLY normalize: lowercase, remove extra spaces
                clean_trait = trait.strip().lower()
                if clean_trait:
                    normalized.add(clean_trait)
        
        return normalized
