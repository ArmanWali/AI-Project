"""
Intelligent Search Agent Module for AI Resume Screening System
================================================================
Implements A* Search Algorithm for optimal candidate matching.

Author: AI Project Team
Course: CS-351 Artificial Intelligence
"""

import heapq
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Candidate:
    """
    Represents a job candidate with their resume information.
    
    Attributes:
        id: Unique identifier
        resume_text: Full resume text
        category: Job category
        skills: List of detected skills
        experience_years: Years of experience
        score: Computed matching score
    """
    id: int
    resume_text: str
    category: str
    skills: List[str] = field(default_factory=list)
    experience_years: int = 0
    score: float = 0.0
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Candidate):
            return self.id == other.id
        return False


class SearchAgent:
    """
    Intelligent agent using A* Search for optimal candidate matching.
    
    The agent searches through a pool of candidates to find the best matches
    for given job requirements using heuristic-guided search.
    
    Attributes:
        candidates: List of Candidate objects
        skill_database: Set of all skills in the candidate pool
    """
    
    # Common technical skills database
    SKILL_PATTERNS = [
        'python', 'java', 'javascript', 'c++', 'c#', 'sql', 'r', 'go', 'rust',
        'machine learning', 'deep learning', 'data science', 'data analysis',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'linux',
        'html', 'css', 'react', 'angular', 'vue', 'nodejs', 'django', 'flask',
        'excel', 'tableau', 'power bi', 'spark', 'hadoop', 'hive',
        'nlp', 'computer vision', 'neural network', 'cnn', 'rnn', 'transformer',
        'git', 'agile', 'scrum', 'jira', 'ci/cd', 'devops',
        'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch'
    ]
    
    def __init__(self, candidates_df=None):
        """
        Initialize the search agent.
        
        Args:
            candidates_df: Optional DataFrame with resume data
        """
        self.candidates = []
        self.skill_database = set()
        
        if candidates_df is not None:
            self._load_candidates(candidates_df)
    
    def _load_candidates(self, df):
        """Load candidates from DataFrame."""
        for idx, row in df.iterrows():
            resume_text = row.get('Resume_str', '')
            skills = self._extract_skills(resume_text)
            experience = self._extract_experience(resume_text)
            
            candidate = Candidate(
                id=idx,
                resume_text=resume_text,
                category=row.get('Category', 'Unknown'),
                skills=skills,
                experience_years=experience
            )
            self.candidates.append(candidate)
            self.skill_database.update(skills)
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text."""
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.SKILL_PATTERNS:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill)
        
        return found_skills
    
    def _extract_experience(self, text: str) -> int:
        """Extract years of experience from text."""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of)?\s*experience',
            r'experience\s*(?:of)?\s*(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*in'
        ]
        
        max_years = 0
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                years = int(match)
                if years < 50:
                    max_years = max(max_years, years)
        
        return max_years
    
    def heuristic(self, candidate: Candidate, job_requirements: Dict) -> float:
        """
        Compute heuristic score for A* search.
        
        The heuristic estimates how well a candidate matches job requirements
        based on skill overlap and experience.
        
        Args:
            candidate: Candidate object
            job_requirements: Dict with 'skills' and 'min_experience'
            
        Returns:
            Heuristic score (higher = better match)
        """
        required_skills = set(job_requirements.get('skills', []))
        candidate_skills = set(candidate.skills)
        
        # Skill match score (0-1)
        if required_skills:
            skill_match = len(required_skills & candidate_skills) / len(required_skills)
        else:
            skill_match = 1.0
        
        # Experience score (0-1)
        min_exp = job_requirements.get('min_experience', 0)
        if min_exp > 0:
            exp_score = min(candidate.experience_years / min_exp, 1.5)
        else:
            exp_score = 1.0
        
        # Bonus skills (additional relevant skills)
        bonus_skills = len(candidate_skills - required_skills) * 0.02
        
        # Category match bonus
        target_category = job_requirements.get('target_category', '')
        category_bonus = 0.3 if candidate.category.lower() == target_category.lower() else 0
        
        # Combined heuristic
        h_score = (0.5 * skill_match + 0.3 * exp_score + 0.1 * bonus_skills + 0.1 * category_bonus)
        
        return min(h_score, 1.0)
    
    def a_star_search(self, job_requirements: Dict, top_k: int = 10) -> List[Dict]:
        """
        A* Search Algorithm for finding optimal candidates.
        
        Uses priority queue with heuristic scoring to efficiently
        search through candidate pool.
        
        Args:
            job_requirements: Dict containing:
                - skills: List of required skills
                - min_experience: Minimum years of experience
                - target_category: Preferred job category
            top_k: Number of top candidates to return
            
        Returns:
            List of top matching candidates with scores
        """
        if not self.candidates:
            return []
        
        # Priority queue: (negative_score, counter, candidate)
        # Using negative score because heapq is min-heap
        frontier = []
        visited = set()
        results = []
        counter = 0
        
        # Initialize with all candidates
        for candidate in self.candidates:
            h_score = self.heuristic(candidate, job_requirements)
            candidate.score = h_score
            heapq.heappush(frontier, (-h_score, counter, candidate))
            counter += 1
        
        # A* Search main loop
        while frontier and len(results) < top_k:
            neg_score, _, current = heapq.heappop(frontier)
            
            if current.id in visited:
                continue
            
            visited.add(current.id)
            
            # Add to results
            results.append({
                'rank': len(results) + 1,
                'candidate_id': current.id,
                'category': current.category,
                'score': -neg_score,
                'skills': current.skills,
                'experience_years': current.experience_years,
                'resume_preview': current.resume_text[:200] + '...'
            })
        
        return results
    
    def print_search_results(self, results: List[Dict], job_requirements: Dict):
        """Pretty print search results."""
        print("\n" + "=" * 70)
        print("A* SEARCH RESULTS - OPTIMAL CANDIDATE MATCHING")
        print("=" * 70)
        
        print(f"\n🎯 Job Requirements:")
        print(f"   Required Skills: {job_requirements.get('skills', [])}")
        print(f"   Min Experience: {job_requirements.get('min_experience', 0)} years")
        print(f"   Target Category: {job_requirements.get('target_category', 'Any')}")
        
        print(f"\n📋 Top {len(results)} Candidates Found:\n")
        
        for r in results:
            print(f"  #{r['rank']:2d} | Score: {r['score']:.2%} | {r['category']}")
            print(f"      Skills: {', '.join(r['skills'][:5])}{'...' if len(r['skills']) > 5 else ''}")
            print(f"      Experience: {r['experience_years']} years")
            print("-" * 50)
        
        print(f"\n✅ Search complete. Nodes explored: {len(results)}")


class BreadthFirstSearch:
    """Alternative BFS implementation for comparison."""
    
    def __init__(self, candidates: List[Candidate]):
        self.candidates = candidates
    
    def search(self, job_requirements: Dict, top_k: int = 10) -> List[Candidate]:
        """Simple BFS without heuristic."""
        from collections import deque
        
        queue = deque(self.candidates)
        results = []
        visited = set()
        
        while queue and len(results) < top_k:
            candidate = queue.popleft()
            
            if candidate.id in visited:
                continue
            visited.add(candidate.id)
            
            # Check basic requirements
            required_skills = set(job_requirements.get('skills', []))
            if required_skills & set(candidate.skills):
                results.append(candidate)
        
        return results


class DepthFirstSearch:
    """Alternative DFS implementation for comparison."""
    
    def __init__(self, candidates: List[Candidate]):
        self.candidates = candidates
    
    def search(self, job_requirements: Dict, top_k: int = 10) -> List[Candidate]:
        """Simple DFS without heuristic."""
        stack = list(self.candidates)
        results = []
        visited = set()
        
        while stack and len(results) < top_k:
            candidate = stack.pop()
            
            if candidate.id in visited:
                continue
            visited.add(candidate.id)
            
            required_skills = set(job_requirements.get('skills', []))
            if required_skills & set(candidate.skills):
                results.append(candidate)
        
        return results


if __name__ == "__main__":
    # Demo with sample data
    print("Search Agent Module - A* Algorithm Demo")
    print("=" * 50)
    
    # Create sample candidates
    sample_data = [
        Candidate(1, "Python ML TensorFlow 5 years experience", "Data Science", 
                  ['python', 'machine learning', 'tensorflow'], 5),
        Candidate(2, "Java Spring AWS 3 years experience", "Backend", 
                  ['java', 'aws'], 3),
        Candidate(3, "Python SQL Data Analysis 2 years", "Data Analysis", 
                  ['python', 'sql', 'data analysis'], 2),
    ]
    
    agent = SearchAgent()
    agent.candidates = sample_data
    
    job_req = {
        'skills': ['python', 'machine learning'],
        'min_experience': 3,
        'target_category': 'Data Science'
    }
    
    results = agent.a_star_search(job_req, top_k=3)
    agent.print_search_results(results, job_req)
