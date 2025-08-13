import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from clients.openai_client import OpenAIClient
from clients.gemini_client import GeminiClient
from clients.qdrant_client import get_qdrant_client, search_points_in_collection
from clients.token_tracker import TokenTracker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserProfile:
    """User information and preferences."""
    age: int
    gender: str
    location: str
    income_range: str
    family_status: str  # single, married, divorced, etc.
    dependents: int
    health_conditions: List[str]
    occupation: str
    lifestyle: str  # active, sedentary, etc.
    risk_tolerance: str  # low, medium, high
    budget_range: str
    coverage_preferences: List[str]  # health, life, auto, home, etc.
    additional_notes: str = ""

@dataclass
class PolicyRecommendation:
    """Represents a recommended policy."""
    policy_content: str
    company_name: str
    policy_type: str
    policy_sub_type: str
    policy_title: str
    pdf_link: str
    keywords: List[str]
    relevance_score: float
    gemini_reasoning: str
    qdrant_score: float
    final_rank: int

class PolicyRecommender:
    """Main class for policy recommendation system."""
    
    def __init__(self):
        self.token_tracker = TokenTracker()
        self.openai_client = OpenAIClient(self.token_tracker)
        self.gemini_client = GeminiClient(self.token_tracker)
        self.qdrant_client = get_qdrant_client()
        self.collection_name = "policy_chunks2"
    
    async def generate_optimized_search_query(self, user_profile: UserProfile) -> str:
        """Use Gemini to generate an optimized search query for better vector search results."""
        try:
            user_context = self._create_user_context(user_profile)
            
            prompt = f"""
            You are an expert insurance search specialist. Your task is to create an optimized search query that will retrieve the most relevant insurance policies from a vector database.

            {user_context}

            Based on this user profile, create a highly targeted search query that will find the most relevant insurance policies. The query should:
            1. Focus on the most important insurance needs for this user
            2. Include relevant demographic and risk factors
            3. Prioritize the user's coverage preferences
            4. Consider their financial situation and risk tolerance
            5. Include specific insurance terminology and keywords
            6. Be concise but comprehensive (50-100 words)

            Create a search query that an insurance vector database would understand to return the most relevant policies for this user.

            Return only the optimized search query, nothing else.
            """

            response = await self.gemini_client.generate_content(
                model="gemini-2.0-flash",
                prompt=prompt
            )
            
            optimized_query = response.strip()
            logger.info(f"Generated optimized search query: {optimized_query}")
            return optimized_query
            
        except Exception as e:
            logger.error(f"Error generating optimized query with Gemini: {e}")
            # Fallback to basic query generation
            return self._generate_basic_search_query(user_profile)
    
    def _generate_basic_search_query(self, user_profile: UserProfile) -> str:
        """Fallback method for basic query generation."""
        query_parts = []
        
        # Add demographic info
        query_parts.append(f"{user_profile.age} years old {user_profile.gender}")
        query_parts.append(f"located in {user_profile.location}")
        query_parts.append(f"income {user_profile.income_range}")
        query_parts.append(f"{user_profile.family_status}")
        
        if user_profile.dependents > 0:
            query_parts.append(f"{user_profile.dependents} dependents")
        
        # Add occupation and lifestyle
        query_parts.append(f"works as {user_profile.occupation}")
        query_parts.append(f"{user_profile.lifestyle} lifestyle")
        
        # Add health conditions
        if user_profile.health_conditions:
            health_str = ", ".join(user_profile.health_conditions)
            query_parts.append(f"health conditions: {health_str}")
        
        # Add preferences
        query_parts.append(f"interested in {', '.join(user_profile.coverage_preferences)} insurance")
        query_parts.append(f"budget range {user_profile.budget_range}")
        query_parts.append(f"{user_profile.risk_tolerance} risk tolerance")
        
        # Add additional notes
        if user_profile.additional_notes:
            query_parts.append(user_profile.additional_notes)
        
        return " ".join(query_parts)
    
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for the search query."""
        try:
            embedding = await self.openai_client.generate_embedding(query)
            logger.info(f"Generated embedding for query (length: {len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def search_similar_policies(
        self, 
        query_embedding: List[float], 
        top_k: int = 20,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar policies in Qdrant and extract comprehensive metadata."""
        try:
            results = search_points_in_collection(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True
            )
            
            policies = []
            for result in results:
                payload = result.payload
                policies.append({
                    'content': payload.get('chunk', ''),
                    'company_name': payload.get('company_name', 'Unknown'),
                    'type': payload.get('type', 'Unknown'),
                    'sub_type': payload.get('sub_type', 'Unknown'),
                    'title': payload.get('title', 'Unknown'),
                    'pdf_link': payload.get('pdf_link', ''),
                    'keywords': payload.get('keywords', []),
                    'metadata': payload,
                    'qdrant_score': result.score,
                    'id': result.id
                })
            
            logger.info(f"Retrieved {len(policies)} policies from Qdrant with comprehensive metadata")
            return policies
        
        except Exception as e:
            logger.error(f"Error searching policies: {e}")
            return []
    
    async def rerank_with_gemini(
        self, 
        user_profile: UserProfile, 
        policies: List[Dict[str, Any]], 
        top_n: int = 5
    ) -> List[PolicyRecommendation]:
        """Re-rank policies using Gemini with comprehensive policy metadata."""
        try:
            user_context = self._create_user_context(user_profile)
            
            recommendations = []
            
            for i, policy in enumerate(policies):
                logger.info(f"Re-ranking policy {i+1}/{len(policies)}")
                
                # Create comprehensive relevance assessment prompt
                prompt = self._create_enhanced_relevance_prompt(user_context, policy)
                
                # Get Gemini assessment
                response = await self.gemini_client.generate_content(
                    model="gemini-2.0-flash",
                    prompt=prompt
                )
                
                # Parse the response to extract score and reasoning
                relevance_score, reasoning = self._parse_gemini_response(response)
                
                recommendations.append(PolicyRecommendation(
                    policy_content=policy['content'],
                    company_name=policy['company_name'],
                    policy_type=policy['type'],
                    policy_sub_type=policy['sub_type'],
                    policy_title=policy['title'],
                    pdf_link=policy['pdf_link'],
                    keywords=policy['keywords'] if isinstance(policy['keywords'], list) else [],
                    relevance_score=relevance_score,
                    gemini_reasoning=reasoning,
                    qdrant_score=policy['qdrant_score'],
                    final_rank=0  # Will be set after sorting
                ))
            
            # Sort by relevance score (higher is better)
            recommendations.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Set final ranks and return top_n
            for i, rec in enumerate(recommendations[:top_n]):
                rec.final_rank = i + 1
            
            logger.info(f"Re-ranked and returning top {top_n} policies")
            return recommendations[:top_n]
        
        except Exception as e:
            logger.error(f"Error re-ranking with Gemini: {e}")
            raise
    
    def _create_user_context(self, user_profile: UserProfile) -> str:
        """Create a structured user context for Gemini."""
        context = f"""
        User Profile:
        - Age: {user_profile.age}
        - Gender: {user_profile.gender}
        - Location: {user_profile.location}
        - Income Range: {user_profile.income_range}
        - Family Status: {user_profile.family_status}
        - Number of Dependents: {user_profile.dependents}
        - Occupation: {user_profile.occupation}
        - Lifestyle: {user_profile.lifestyle}
        - Health Conditions: {', '.join(user_profile.health_conditions) if user_profile.health_conditions else 'None reported'}
        - Risk Tolerance: {user_profile.risk_tolerance}
        - Budget Range: {user_profile.budget_range}
        - Coverage Preferences: {', '.join(user_profile.coverage_preferences)}
        - Additional Notes: {user_profile.additional_notes if user_profile.additional_notes else 'None'}
        """
        return context.strip()
    
    def _create_enhanced_relevance_prompt(self, user_context: str, policy: Dict[str, Any]) -> str:
        """Create an enhanced prompt for Gemini with comprehensive policy metadata."""
        keywords_str = ', '.join(policy['keywords']) if policy['keywords'] else 'None'
        
        prompt = f"""
        You are an expert insurance advisor. Analyze how relevant the following insurance policy is for the given user profile.

        {user_context}

        Insurance Policy Information:
        - Company: {policy['company_name']}
        - Policy Type: {policy['type']}
        - Policy Sub-type: {policy['sub_type']}
        - Policy Title: {policy['title']}
        - Keywords: {keywords_str}
        - PDF Link: {policy['pdf_link']}

        Policy Content:
        {policy['content']}

        Evaluate the relevance of this policy for the user based on:
        1. Age appropriateness and life stage compatibility
        2. Financial suitability (income/budget alignment)
        3. Coverage needs and preferences match
        4. Risk tolerance compatibility
        5. Health conditions consideration
        6. Family situation appropriateness
        7. Lifestyle compatibility
        8. Company reputation and policy type suitability
        9. Geographic availability and relevance
        10. Policy features vs user requirements

        Consider the policy metadata (company, type, keywords) in your assessment. A well-known company with relevant policy type and matching keywords should score higher.

        Provide your assessment in this exact format:
        RELEVANCE_SCORE: [number from 0.0 to 10.0]
        REASONING: [3-4 sentences explaining why this policy is or isn't suitable for the user, considering both content and metadata]

        Be precise with the score - use decimal places. A score of 10.0 means perfect match, 0.0 means completely irrelevant.
        """
        return prompt
    
    def _parse_gemini_response(self, response: str) -> tuple[float, str]:
        """Parse Gemini response to extract relevance score and reasoning."""
        try:
            lines = response.strip().split('\n')
            relevance_score = 0.0
            reasoning = "No reasoning provided"
            
            for line in lines:
                if line.startswith('RELEVANCE_SCORE:'):
                    score_str = line.replace('RELEVANCE_SCORE:', '').strip()
                    relevance_score = float(score_str)
                elif line.startswith('REASONING:'):
                    reasoning = line.replace('REASONING:', '').strip()
            
            return relevance_score, reasoning
        
        except Exception as e:
            logger.warning(f"Error parsing Gemini response: {e}")
            return 5.0, "Unable to parse reasoning from Gemini response"
    
    async def recommend_policies(
        self, 
        user_profile: UserProfile, 
        top_k: int = 20, 
        top_n: int = 5,
        score_threshold: float = 0.7
    ) -> List[PolicyRecommendation]:
        """
        Complete pipeline with Gemini-optimized search and enhanced metadata.
        
        Args:
            user_profile: User information and preferences
            top_k: Number of policies to retrieve from Qdrant
            top_n: Number of final recommendations to return
            score_threshold: Minimum similarity score for Qdrant search
        
        Returns:
            List of top_n PolicyRecommendation objects with comprehensive metadata
        """
        logger.info("Starting enhanced policy recommendation pipeline")
        
        # Step 1: Generate optimized search query using Gemini
        logger.info("Step 1: Generating optimized search query with Gemini")
        search_query = await self.generate_optimized_search_query(user_profile)
        logger.info(f"Optimized query: {search_query}")
        
        # Step 2: Generate embeddings for the optimized query
        logger.info("Step 2: Generating query embeddings")
        query_embedding = await self.generate_query_embedding(search_query)
        
        # Step 3: Search for similar policies in Qdrant with comprehensive metadata
        logger.info("Step 3: Searching for similar policies with metadata")
        similar_policies = self.search_similar_policies(
            query_embedding, 
            top_k=top_k, 
            score_threshold=score_threshold
        )
        
        if not similar_policies:
            logger.warning("No similar policies found")
            return []
        
        # Step 4: Re-rank using Gemini with enhanced prompts and metadata
        logger.info("Step 4: Re-ranking with Gemini using enhanced metadata")
        final_recommendations = await self.rerank_with_gemini(
            user_profile, 
            similar_policies, 
            top_n=top_n
        )
        
        logger.info(f"Enhanced pipeline complete. Returning {len(final_recommendations)} recommendations")
        return final_recommendations
    
    def print_recommendations(self, recommendations: List[PolicyRecommendation]):
        """Print formatted recommendations with comprehensive metadata."""
        print("\n" + "="*100)
        print("ENHANCED POLICY RECOMMENDATIONS")
        print("="*100)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\nRank {i}:")
            print(f"Company: {rec.company_name}")
            print(f"Policy Type: {rec.policy_type}")
            print(f"Policy Sub-type: {rec.policy_sub_type}")
            print(f"Title: {rec.policy_title}")
            print(f"Keywords: {', '.join(rec.keywords) if rec.keywords else 'None'}")
            print(f"PDF Link: {rec.pdf_link}")
            print(f"Relevance Score: {rec.relevance_score}/10.0")
            print(f"Qdrant Similarity: {rec.qdrant_score:.3f}")
            print(f"Reasoning: {rec.gemini_reasoning}")
            print(f"Policy Content (first 300 chars): {rec.policy_content[:300]}...")
            print("-" * 80)
        
        # Print token usage summary
        self.token_tracker.print_summary()

# Example usage
async def main():
    """Example usage of the enhanced PolicyRecommender."""
    
    # Create user profile
    user_profile = UserProfile(
        age=35,
        gender="male",
        location="Delhi, India",
        income_range="20-22 LPA",
        family_status="married",
        dependents=2,
        health_conditions=["diabetes", "high blood pressure"],
        occupation="software engineer",
        lifestyle="sedentary",
        risk_tolerance="medium",
        budget_range="30-40K per month",
        coverage_preferences=["health", "life", "disability"],
        additional_notes="Looking for comprehensive family coverage with good preventive care benefits"
    )
    
    # Initialize recommender
    recommender = PolicyRecommender()
    
    try:
        # Get recommendations with enhanced pipeline
        recommendations = await recommender.recommend_policies(
            user_profile=user_profile,
            top_k=15,  # Retrieve 15 policies from Qdrant
            top_n=5,   # Return top 5 after re-ranking
            score_threshold=0.6
        )
        
        # Print results with enhanced formatting
        recommender.print_recommendations(recommendations)  
        
    except Exception as e:
        logger.error(f"Error in enhanced recommendation pipeline: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())