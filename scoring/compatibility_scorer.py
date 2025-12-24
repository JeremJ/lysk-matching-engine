"""
Compatibility Scorer - Handles all compatibility calculation including embeddings and scoring
"""
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from numpy.typing import NDArray

from matching_engine.types.matching_profile import MatchingProfile
from matching_engine.types.embedding_result import EmbeddingResult
from matching_engine.types.embedder_types import EmbedderType
from matching_engine.embeddings.embedders.life_goals_embedder import LifeGoalsEmbedder
from matching_engine.adapters.database.matching_details_adapter import MatchingDetailsAdapter, EmbeddingData

logger = logging.getLogger(__name__)


@dataclass
class ScoringDecision:
    """Detailed breakdown of how the final compatibility score was calculated"""
    candidate_user_id: str
    seeker_gender: str  # "man" or "woman" 
    
    # Raw component scores
    life_goals_similarity: float  # From embeddings cosine similarity
    appearance_mean_score: float  # From OpenAI appearance analysis
    
    # Applied weights based on seeker gender
    life_goals_weight: float      # 0.6 for men, 0.8 for women
    appearance_weight: float      # 0.4 for men, 0.2 for women
    
    # Weighted components
    weighted_life_goals: float    # life_goals_similarity * life_goals_weight
    weighted_appearance: float    # appearance_mean_score * appearance_weight
    
    # Final result
    final_compatibility_score: float  # weighted_life_goals + weighted_appearance
    
    # Calculation details for transparency
    calculation_formula: str      # e.g., "0.6×0.85 + 0.4×0.72 = 0.798"


class SeekerGender(str, Enum):
    """Gender of the person seeking a match"""
    MAN = "man"
    WOMAN = "woman"


class CompatibilityScorer:
    """
    Combines life goals and appearance scores using gender-based weighting:
    - Men seeking: 0.4 x life_goals + 0.6 x appearance (beauty prioritized)
    - Women seeking: 0.8 x life_goals + 0.2 x appearance (life goals prioritized)
    """
    
    # Gender-based weighting constants - men prioritize appearance more
    WEIGHTS = {
        SeekerGender.MAN: {"life_goals": 0.4, "appearance": 0.6},    # Men: beauty more important
        SeekerGender.WOMAN: {"life_goals": 0.8, "appearance": 0.2}   # Women: life goals more important
    }
    
    def __init__(self, appearance_scorer=None, matching_details_adapter=None):
        """Initialize with all scoring dependencies"""
        self.appearance_scorer = appearance_scorer
        self.life_goals_embedder = LifeGoalsEmbedder()
        self.matching_details_adapter = matching_details_adapter
    
    def calculate_compatibility(
        self, 
        candidate_user_id: str,
        seeker_gender: SeekerGender,
        life_goals_score: float, 
        appearance_score: float
    ) -> ScoringDecision:
        """
        Calculate weighted compatibility score with detailed breakdown.
        
        Args:
            candidate_user_id: ID of the candidate being scored
            seeker_gender: Gender of person seeking match
            life_goals_score: Semantic similarity score for life goals (0.0-1.0)
            appearance_score: Physical attractiveness score (0.0-1.0)
            
        Returns:
            ScoringDecision with detailed breakdown
        """
        # Validate input scores are in expected range (0.0-1.0)
        if life_goals_score > 1.0 or life_goals_score < 0.0:
            logger.warning(f"⚠️ Life goals score out of range for {candidate_user_id}: {life_goals_score} (expected 0.0-1.0)")
        if appearance_score > 1.0 or appearance_score < 0.0:
            logger.warning(f"⚠️ Appearance score out of range for {candidate_user_id}: {appearance_score} (expected 0.0-1.0)")
        
        # Clamp scores to valid range as safety measure
        life_goals_score = max(0.0, min(1.0, life_goals_score))
        appearance_score = max(0.0, min(1.0, appearance_score))
        
        weights = self.WEIGHTS[seeker_gender]
        
        # Calculate weighted components
        weighted_life_goals = weights["life_goals"] * life_goals_score
        weighted_appearance = weights["appearance"] * appearance_score
        final_score = weighted_life_goals + weighted_appearance
        
        # Final validation - should always be ≤ 1.0 with proper normalization
        if final_score > 1.0:
            logger.error(f"🚨 SCORING BUG: Final compatibility score > 1.0 for {candidate_user_id}: {final_score}")
            logger.error(f"   Life goals: {life_goals_score}, Appearance: {appearance_score}")
            logger.error(f"   Weights: {weights}")
            final_score = min(1.0, final_score)  # Emergency clamp
        
        # Create detailed calculation formula
        formula = (
            f"{weights['life_goals']:.1f}×{life_goals_score:.3f} + "
            f"{weights['appearance']:.1f}×{appearance_score:.3f} = {final_score:.3f}"
        )
        
        decision = ScoringDecision(
            candidate_user_id=candidate_user_id,
            seeker_gender=seeker_gender.value,
            life_goals_similarity=life_goals_score,
            appearance_mean_score=appearance_score,
            life_goals_weight=weights["life_goals"],
            appearance_weight=weights["appearance"],
            weighted_life_goals=round(weighted_life_goals, 3),
            weighted_appearance=round(weighted_appearance, 3),
            final_compatibility_score=round(final_score, 3),
            calculation_formula=formula
        )
        
        logger.debug(f"Scoring decision for {candidate_user_id}: {formula}")
        return decision
    
    def _embedding_data_to_result(self, questionnaire_id: str, embedding_data: EmbeddingData, embedder_type: EmbedderType = EmbedderType.LIFE_GOALS) -> EmbeddingResult:
        """Convert EmbeddingData to proper EmbeddingResult"""
        import numpy as np
        return EmbeddingResult(
            questionnaire_id=questionnaire_id,
            embedder_name=embedder_type.value,
            embedding_vector=np.array(embedding_data.embedding_vector, dtype=np.float32),
            dimension_count=len(embedding_data.embedding_vector),
            metadata={},
            confidence_score=embedding_data.confidence_score
        )

    async def _get_embedding(self, profile: MatchingProfile, embedder_type: EmbedderType = EmbedderType.LIFE_GOALS) -> Optional[EmbeddingResult]:
        """Get embedding - adapter handles all the complexity (cache, support, etc.)"""
        if not self.matching_details_adapter:
            logger.error("No matching details adapter available")
            return None
        
        # Adapter handles everything: cache, support checking, fallbacks
        cached = await self.matching_details_adapter.get_embedding(profile.questionnaire_id, embedder_type)
        if cached:
            return self._embedding_data_to_result(profile.questionnaire_id, cached, embedder_type)
        
        # Only fallback if adapter returns None and it's a supported type we can generate
        if embedder_type == EmbedderType.LIFE_GOALS:
            logger.warning(f"Adapter returned None, generating new embedding for {profile.questionnaire_id}")
            embedding = await self.life_goals_embedder.embed_single(profile)
            if embedding.confidence_score > 0.0:
                # IMPORTANT: Store the newly generated embedding back to DB+cache
                if self.matching_details_adapter:
                    try:
                        await self.matching_details_adapter.store_embedding(
                            profile.questionnaire_id,
                            embedder_type, 
                            embedding.embedding_vector.tolist(), 
                            embedding.confidence_score
                        )
                        logger.info(f"💾 Stored new {embedder_type.value} embedding for {profile.questionnaire_id}")
                    except Exception as e:
                        logger.error(f"❌ Failed to store embedding for {profile.questionnaire_id}: {e}")
                
                return embedding
        
        return None
    
    def _calculate_similarities(self, 
                              target_vector: NDArray[np.float32], 
                              candidate_vectors: List[NDArray[np.float32]]) -> List[float]:
        """Calculate cosine similarities between target and candidates"""
        similarities = []
        
        # Normalize target vector
        target_norm = np.linalg.norm(target_vector)
        if target_norm == 0:
            return [0.0] * len(candidate_vectors)
        
        normalized_target = target_vector / target_norm
        
        for candidate_vector in candidate_vectors:
            # Normalize candidate vector
            candidate_norm = np.linalg.norm(candidate_vector)
            if candidate_norm == 0:
                similarities.append(0.0)
                continue
            
            normalized_candidate = candidate_vector / candidate_norm
            
            # Calculate cosine similarity
            similarity = np.dot(normalized_target, normalized_candidate)
            similarities.append(float(similarity))
        
        return similarities
    
    async def calculate_compatibility_scores(
        self,
        target_profile: MatchingProfile,
        candidate_profiles: List[MatchingProfile],
        min_similarity_threshold: float = 0.3
    ) -> List[ScoringDecision]:
        """
        Calculate full compatibility pipeline: embeddings → similarities → appearance → weighted scores
        
        Args:
            target_profile: The user seeking matches
            candidate_profiles: Potential matches
            min_similarity_threshold: Minimum life goals similarity to include
            
        Returns:
            List of ScoringDecision objects sorted by compatibility score (descending)
        """
        seeker_gender = SeekerGender.MAN if target_profile.gender.name == "MALE" else SeekerGender.WOMAN
        embedder_type = EmbedderType.LIFE_GOALS
        
        # Get target embedding - adapter handles all complexity
        target_embedding = await self._get_embedding(target_profile, embedder_type)
        if not target_embedding:
            logger.error(f"Failed to get embedding for target user {target_profile.questionnaire_id}")
            return []
        
        # Get candidate embeddings - adapter handles caching, support, etc.
        candidate_embeddings = []
        valid_candidates = []
        
        for candidate in candidate_profiles:
            embedding = await self._get_embedding(candidate, embedder_type)
            if embedding:
                candidate_embeddings.append(embedding.embedding_vector)
                valid_candidates.append(candidate)
        
        if not candidate_embeddings:
            logger.warning("No valid candidate embeddings found")
            return []
        
        # Calculate life goals similarities
        similarities = self._calculate_similarities(
            target_embedding.embedding_vector, 
            candidate_embeddings
        )
        
        # Filter by similarity threshold and calculate final scores
        results = []
        for candidate, similarity in zip(valid_candidates, similarities):
            if similarity >= min_similarity_threshold:
                # Get appearance score (scorer handles cache vs generate)
                if self.appearance_scorer:
                    appearance_score = await self.appearance_scorer.get_or_calculate_score(candidate)
                else:
                    appearance_score = 0.5
                
                # Calculate detailed compatibility with breakdown
                scoring_decision = self.calculate_compatibility(
                    candidate.questionnaire_id, seeker_gender, similarity, appearance_score
                )
                
                results.append(scoring_decision)
        
        # Sort by compatibility score (descending) - DOMAIN LAYER SORTING
        results.sort(key=lambda x: x.final_compatibility_score, reverse=True)
        
        logger.info(f"Calculated compatibility for {len(results)}/{len(candidate_profiles)} candidates (sorted by compatibility)")
        return results
