"""
Simple Matching Engine - The main orchestrator for finding compatible matches
Super simple but functional approach using embeddings and similarity search
"""
import logging
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from commons.enums import Gender

from matching_engine.types.matching_profile import MatchingProfile
from matching_engine.types.matching_types import MatchRequest, MatchResult, MatchCandidate, MatchReason, PairwiseMatchResult
from matching_engine.filters.filter_factory import create_filters
from matching_engine.scoring.compatibility_scorer import CompatibilityScorer, ScoringDecision
from questionnaire.domain.questionnaire_response import ExtendedQuestionnaireResponse
from questionnaire.questionnaire_facade import QuestionnaireFacade


logger = logging.getLogger(__name__)


class MatchingEngine:
    """
    Clean orchestrator for the matching pipeline - delegates business logic to domain services
    
    Flow:
    1. Load user profiles from questionnaire data
    2. Apply filters to eliminate incompatible candidates (age, height, etc.)
    3. Delegate compatibility scoring to CompatibilityScorer (embeddings + appearance)
    4. Create match candidates and return ranked results
    
    This follows proper DDD - orchestration only, no business logic.
    """
    
    def __init__(self, 
                 bidirectional_filters: bool = True,
                 compatibility_scorer: Optional[CompatibilityScorer] = None):
        # Simple orchestration components only
        self.filter_pipeline = create_filters(bidirectional_filters)
        self.questionnaire_facade = QuestionnaireFacade()
        self.compatibility_scorer = compatibility_scorer or CompatibilityScorer()

    
    def _create_enhanced_match_candidate(self, 
                                       user: MatchingProfile, 
                                       life_goals_score: float,
                                       confidence_score: float,
                                       appearance_score: float,
                                       compatibility_score: float,
                                       scoring_decision: Optional[ScoringDecision],
                                       user_id: str = "") -> MatchCandidate:
        """Create a match candidate with enhanced scoring breakdown"""
        
        # Determine primary reasons based on compatibility score
        primary_reasons = [MatchReason.LIFE_GOALS_ALIGNMENT]
        if compatibility_score > 0.7:
            primary_reasons.append(MatchReason.VALUES_COMPATIBILITY)
        
        # Generate enhanced explanation
        explanation = self._generate_enhanced_explanation(
            life_goals_score, appearance_score, compatibility_score, user
        )
        
        # Build metadata with optional scoring decision
        metadata = {
            "embedding_fields_used": ["life_goals"],
            "similarity_method": "weighted_compatibility",
            "life_goals_score": life_goals_score,
            "appearance_score": appearance_score,
            "compatibility_score": compatibility_score,
        }
        
        if scoring_decision:
            metadata["scoring_decision"] = {
                "seeker_gender": scoring_decision.seeker_gender,
                "life_goals_similarity": scoring_decision.life_goals_similarity,
                "appearance_mean_score": scoring_decision.appearance_mean_score,
                "life_goals_weight": scoring_decision.life_goals_weight,
                "appearance_weight": scoring_decision.appearance_weight,
                "weighted_life_goals": scoring_decision.weighted_life_goals,
                "weighted_appearance": scoring_decision.weighted_appearance,
                "final_compatibility_score": scoring_decision.final_compatibility_score,
                "calculation_formula": scoring_decision.calculation_formula
            }
        
        return MatchCandidate(
            profile=user,
            questionnaire_id=user.questionnaire_id,
            user_id=user_id,  # Real user_id passed from caller
            similarity_score=compatibility_score,  # Keep original for backward compatibility
            confidence_score=confidence_score,
            primary_reasons=primary_reasons,
            explanation=explanation,
            metadata=metadata,
        )

    def _generate_enhanced_explanation(self, 
                                     life_goals_score: float, 
                                     appearance_score: float, 
                                     compatibility_score: float, 
                                     user: MatchingProfile) -> str:
        """Generate enhanced explanation including life goals and appearance"""
        if compatibility_score > 0.8:
            return f"Exceptional match with {user.questionnaire_id}. Strong life goals alignment ({life_goals_score:.2f}) and high overall compatibility ({compatibility_score:.2f})."
        elif compatibility_score > 0.6:
            return f"Strong match with {user.questionnaire_id}. Good life goals compatibility ({life_goals_score:.2f}) with overall score of {compatibility_score:.2f}."
        elif compatibility_score > 0.4:
            return f"Good potential match with {user.questionnaire_id}. Compatible outlook with overall score of {compatibility_score:.2f}."
        else:
            return f"Moderate match with {user.questionnaire_id}. Overall compatibility score: {compatibility_score:.2f}."
    
    def _empty_result(self, target_user_id: str, total_candidates: int, processing_time: float) -> MatchResult:
        """Create empty result when no matches found"""
        return MatchResult(
            target_user_id=target_user_id,
            candidates=[],
            total_candidates_considered=total_candidates,
            embeddings_used=["life_goals"],
            processing_time_ms=processing_time * 1000,
            metadata={"error": "no_valid_embeddings"}
        )
    
    def _safe_profile_conversion(self, questionnaire: ExtendedQuestionnaireResponse) -> Optional[MatchingProfile]:
        """Safely convert questionnaire to MatchingProfile, returning None on error"""
        try:
            return MatchingProfile.from_questionnaire(questionnaire)
        except Exception as e:
            logger.warning(f"Failed to create profile for {questionnaire.questionnaire_id}: {e}")
            return None
    
    async def find_matches(self, 
                         target_user_id: str, 
                         max_matches: int = 10) -> MatchResult:
        """
        Find matches for a specific user by loading all profiles from database
        
        Args:
            target_user_id: The user ID to find matches for
            max_matches: Maximum number of matches to return
            
        Returns:
            MatchResult with ranked candidates
        """
        start_time = time.time()
        
        # Get ALL questionnaires from database in single call
        all_questionnaires = self.questionnaire_facade.get_all_extended_questionnaires()
        logger.info(f"Loaded {len(all_questionnaires)} total questionnaires from database")
        
        # Extract target user's questionnaire using collections method
        target_questionnaire = next(
            (q for q in all_questionnaires if q.questionnaire_id == target_user_id), 
            None
        )
        if not target_questionnaire:
            logger.error(f"Could not find questionnaire for user {target_user_id}")
            return self._empty_result(target_user_id, 0, time.time() - start_time)
        
        # Convert target questionnaire to MatchingProfile
        try:
            target_profile = MatchingProfile.from_questionnaire(target_questionnaire)
            logger.info(f"Target user {target_user_id}: {target_profile.gender.value}, age {target_profile.age}")
        except Exception as e:
            logger.error(f"Failed to create target profile for {target_user_id}: {e}")
            return self._empty_result(target_user_id, 0, time.time() - start_time)
        
        # Filter and convert questionnaires using pythonic list comprehension
        opposite_gender = Gender.FEMALE if target_profile.gender == Gender.MALE else Gender.MALE
        
        candidate_profiles = [
            profile for profile in [
                self._safe_profile_conversion(q) for q in all_questionnaires
                if q.questionnaire_id != target_user_id and q.regular_data.gender == opposite_gender
            ] if profile is not None
        ]
        
        logger.info(f"Found {len(candidate_profiles)} {opposite_gender.value.lower()} candidates for {target_profile.gender.value.lower()} target")
        
        if not candidate_profiles:
            logger.warning(f"No {opposite_gender.value.lower()} candidates found")
            return self._empty_result(target_user_id, 0, time.time() - start_time)
        
        # Apply filters FIRST to eliminate incompatible candidates
        print(f"🔍 Applying filters to {len(candidate_profiles)} candidates...")
        filter_result = self.filter_pipeline.apply_all(target_profile, candidate_profiles)
        filtered_candidates = filter_result.final_candidates
        
        if not filtered_candidates:
            print("❌ No candidates passed filters")
            return self._empty_result(target_user_id, len(candidate_profiles), time.time() - start_time)
        
        print(f"✅ Filters passed: {len(filtered_candidates)}/{len(candidate_profiles)} candidates")
        
        # Delegate ALL scoring logic to CompatibilityScorer (returns sorted results)
        compatibility_results = await self.compatibility_scorer.calculate_compatibility_scores(
            target_profile, filtered_candidates
        )
        
        if not compatibility_results:
            return self._empty_result(target_user_id, len(candidate_profiles), time.time() - start_time)
        
        # Create match candidates from compatibility results (already sorted by scorer)
        match_candidates = []
        candidate_lookup = {c.questionnaire_id: c for c in filtered_candidates}
        
        for scoring_decision in compatibility_results:  # Already sorted by compatibility score
            candidate = candidate_lookup[scoring_decision.candidate_user_id]
            
            # Get real user_id from questionnaire
            candidate_questionnaire = next((q for q in all_questionnaires if q.questionnaire_id == candidate.questionnaire_id), None)
            real_user_id = candidate_questionnaire.user_id if candidate_questionnaire else ""
            
            match_candidate = self._create_enhanced_match_candidate(
                candidate,
                scoring_decision.life_goals_similarity,
                0.8,  # Default confidence since scorer handles embedding confidence
                scoring_decision.appearance_mean_score,
                scoring_decision.final_compatibility_score,
                scoring_decision,  # Pass full scoring decision for API transparency
                real_user_id  # Pass real user_id
            )
            
            match_candidates.append(match_candidate)
        
        # No need to sort - results already sorted by CompatibilityScorer at domain layer
        
        # Take top matches
        top_matches = match_candidates[:max_matches]
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return MatchResult(
            target_user_id=target_user_id,
            candidates=top_matches,
            total_candidates_considered=len(candidate_profiles),
            embeddings_used=["life_goals"],
            processing_time_ms=processing_time,
            metadata={
                "compatibility_matches": len(match_candidates),
                "filter_stats": filter_result.filter_stats,
                "candidates_after_filtering": len(filtered_candidates),
                "candidates_filtered_out": filter_result.total_rejected,
                "filters_used": self.filter_pipeline.get_filter_names()
            }
        )
    
    async def find_pairwise_match(self, target_user_id: str, candidate_user_id: str) -> Optional[PairwiseMatchResult]:
        """
        Find compatibility between two specific users.
        Returns typed PairwiseMatchResult or None if not found.
        """
        start_time = time.time()
        
        # Get questionnaires for both users
        all_questionnaires = self.questionnaire_facade.get_all_extended_questionnaires()
        
        target_questionnaire = next((q for q in all_questionnaires if q.questionnaire_id == target_user_id), None)
        candidate_questionnaire = next((q for q in all_questionnaires if q.questionnaire_id == candidate_user_id), None)
        
        if not target_questionnaire or not candidate_questionnaire:
            return None
        
        # Convert to MatchingProfile
        try:
            target_profile = MatchingProfile.from_questionnaire(target_questionnaire)
            candidate_profile = MatchingProfile.from_questionnaire(candidate_questionnaire)
        except Exception as e:
            logger.error(f"Failed to create profiles: {e}")
            return None
        
        # Apply filters to check basic compatibility
        filter_result = self.filter_pipeline.apply_all(target_profile, [candidate_profile])
        if not filter_result.final_candidates:
            # Throw clear error for admin panel
            failed_filters = [name for name, passed in filter_result.filter_stats.items() if not passed]
            raise ValueError(f"Users are incompatible due to failed filters: {', '.join(failed_filters)}")
        
        # Calculate compatibility score
        compatibility_results = await self.compatibility_scorer.calculate_compatibility_scores(
            target_profile, [candidate_profile]
        )
        
        if not compatibility_results:
            raise ValueError("Failed to calculate compatibility scores - no valid embeddings or appearance data found")
        
        scoring_decision = compatibility_results[0]
        match_candidate = self._create_enhanced_match_candidate(
            candidate_profile,
            scoring_decision.life_goals_similarity,
            0.9,  # High confidence for pair comparison
            scoring_decision.appearance_mean_score,
            scoring_decision.final_compatibility_score,
            scoring_decision,
            candidate_questionnaire.user_id  # Pass real user_id directly
        )
        
        processing_time = (time.time() - start_time) * 1000
        return PairwiseMatchResult(
            target_questionnaire=target_questionnaire,
            candidate_questionnaire=candidate_questionnaire,
            match_candidate=match_candidate,
            processing_time_ms=processing_time
        )
