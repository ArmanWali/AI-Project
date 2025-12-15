"""
Constraint Satisfaction Problem (CSP) Module for Job-Resume Matching
======================================================================
Implements CSP formulation with backtracking and arc consistency algorithms
for optimal job-candidate assignment.

Author: AI Project Team
Course: CS-351 Artificial Intelligence
"""

from typing import List, Dict, Set, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
from collections import deque


@dataclass
class JobPosition:
    """Represents a job position with requirements."""
    id: str
    title: str
    required_skills: Set[str]
    min_experience: int
    max_candidates: int = 1
    location: str = "Any"
    
    def __hash__(self):
        return hash(self.id)


@dataclass  
class CandidateCSP:
    """Represents a candidate in CSP formulation."""
    id: str
    name: str
    skills: Set[str]
    experience_years: int
    location: str = "Any"
    assigned_job: Optional[str] = None
    
    def __hash__(self):
        return hash(self.id)


class JobResumeCSP:
    """
    Constraint Satisfaction Problem solver for job-resume matching.
    
    CSP Formulation:
    - Variables: Candidates
    - Domains: Available job positions for each candidate
    - Constraints: 
        * Skill requirements
        * Experience requirements
        * Location preferences
        * Job capacity limits
    
    Algorithms Implemented:
    - Backtracking Search
    - Arc Consistency (AC-3)
    - Forward Checking
    - MRV (Minimum Remaining Values) heuristic
    """
    
    def __init__(self):
        """Initialize the CSP solver."""
        self.candidates: Dict[str, CandidateCSP] = {}
        self.jobs: Dict[str, JobPosition] = {}
        self.domains: Dict[str, Set[str]] = {}  # candidate_id -> set of job_ids
        self.constraints: List[Callable] = []
        self.assignments: Dict[str, str] = {}  # candidate_id -> job_id
        self.job_assignments: Dict[str, List[str]] = {}  # job_id -> list of candidate_ids
    
    def add_candidate(self, candidate: CandidateCSP):
        """Add a candidate to the CSP."""
        self.candidates[candidate.id] = candidate
        self.domains[candidate.id] = set()
    
    def add_job(self, job: JobPosition):
        """Add a job position to the CSP."""
        self.jobs[job.id] = job
        self.job_assignments[job.id] = []
    
    def add_candidates_from_dataframe(self, df, limit: int = 100):
        """Load candidates from a DataFrame."""
        from src.preprocessing import ResumePreprocessor
        
        for idx, row in df.head(limit).iterrows():
            skills = set(ResumePreprocessor.extract_skills(
                ResumePreprocessor.clean_text(row.get('Resume_str', ''))
            ))
            experience = ResumePreprocessor.extract_experience_years(row.get('Resume_str', ''))
            
            candidate = CandidateCSP(
                id=f"C{idx:04d}",
                name=f"Candidate_{idx}",
                skills=skills,
                experience_years=experience
            )
            self.add_candidate(candidate)
    
    def initialize_domains(self):
        """Initialize domains for all candidates based on basic compatibility."""
        for cand_id, candidate in self.candidates.items():
            for job_id, job in self.jobs.items():
                # Check basic compatibility
                if self._is_compatible(candidate, job):
                    self.domains[cand_id].add(job_id)
    
    def _is_compatible(self, candidate: CandidateCSP, job: JobPosition) -> bool:
        """Check if a candidate is compatible with a job (basic constraints)."""
        # Skill constraint: at least 60% of required skills
        if job.required_skills:
            skill_match = len(candidate.skills & job.required_skills) / len(job.required_skills)
            if skill_match < 0.6:
                return False
        
        # Experience constraint
        if candidate.experience_years < job.min_experience:
            return False
        
        # Location constraint
        if job.location != "Any" and candidate.location != "Any":
            if job.location != candidate.location:
                return False
        
        return True
    
    def _is_consistent(self, candidate_id: str, job_id: str) -> bool:
        """Check if assignment is consistent with all constraints."""
        candidate = self.candidates[candidate_id]
        job = self.jobs[job_id]
        
        # Basic compatibility
        if not self._is_compatible(candidate, job):
            return False
        
        # Job capacity constraint
        current_count = len(self.job_assignments.get(job_id, []))
        if current_count >= job.max_candidates:
            return False
        
        return True
    
    def _select_unassigned_variable(self) -> Optional[str]:
        """
        Select next unassigned variable using MRV heuristic.
        
        MRV (Minimum Remaining Values): Choose the variable with 
        the fewest legal values remaining.
        """
        unassigned = [cid for cid in self.candidates if cid not in self.assignments]
        
        if not unassigned:
            return None
        
        # MRV heuristic
        return min(unassigned, key=lambda cid: len(self.domains[cid]))
    
    def _order_domain_values(self, candidate_id: str) -> List[str]:
        """
        Order domain values using LCV heuristic.
        
        LCV (Least Constraining Value): Prefer values that rule out 
        the fewest choices for neighboring variables.
        """
        domain = list(self.domains[candidate_id])
        
        def constraining_count(job_id):
            count = 0
            for other_cid in self.candidates:
                if other_cid != candidate_id and other_cid not in self.assignments:
                    if job_id in self.domains[other_cid]:
                        count += 1
            return count
        
        return sorted(domain, key=constraining_count)
    
    def backtracking_search(self) -> Optional[Dict[str, str]]:
        """
        Backtracking algorithm with CSP optimizations.
        
        Returns:
            Dictionary of assignments {candidate_id: job_id} or None if no solution
        """
        return self._backtrack()
    
    def _backtrack(self) -> Optional[Dict[str, str]]:
        """Recursive backtracking helper."""
        # Check if assignment is complete
        if len(self.assignments) == len(self.candidates):
            return dict(self.assignments)
        
        # Select next variable (MRV)
        candidate_id = self._select_unassigned_variable()
        if candidate_id is None:
            return dict(self.assignments)
        
        # Try each value in domain (LCV ordering)
        for job_id in self._order_domain_values(candidate_id):
            if self._is_consistent(candidate_id, job_id):
                # Make assignment
                self.assignments[candidate_id] = job_id
                self.job_assignments[job_id].append(candidate_id)
                
                # Save domains for backtracking
                saved_domains = {k: set(v) for k, v in self.domains.items()}
                
                # Forward checking
                if self._forward_check(candidate_id, job_id):
                    result = self._backtrack()
                    if result is not None:
                        return result
                
                # Backtrack
                del self.assignments[candidate_id]
                self.job_assignments[job_id].remove(candidate_id)
                self.domains = saved_domains
        
        return None
    
    def _forward_check(self, assigned_cid: str, assigned_job: str) -> bool:
        """
        Forward checking: Remove inconsistent values from other domains.
        
        Returns:
            True if no domain becomes empty, False otherwise
        """
        job = self.jobs[assigned_job]
        
        # If job is at capacity, remove from all other domains
        if len(self.job_assignments[assigned_job]) >= job.max_candidates:
            for cid in self.candidates:
                if cid != assigned_cid and cid not in self.assignments:
                    self.domains[cid].discard(assigned_job)
                    if len(self.domains[cid]) == 0:
                        return False
        
        return True
    
    def ac3(self) -> bool:
        """
        AC-3 (Arc Consistency Algorithm 3).
        
        Enforces arc consistency by removing values from domains
        that cannot participate in any consistent assignment.
        
        Returns:
            True if CSP is arc consistent, False if no solution exists
        """
        # Initialize queue with all arcs
        queue = deque()
        for cid in self.candidates:
            for job_id in self.domains[cid]:
                queue.append((cid, job_id))
        
        while queue:
            (cid, job_id) = queue.popleft()
            
            if self._revise(cid, job_id):
                if len(self.domains[cid]) == 0:
                    return False
                
                # Add affected arcs back to queue
                for other_cid in self.candidates:
                    if other_cid != cid:
                        queue.append((other_cid, job_id))
        
        return True
    
    def _revise(self, cid: str, job_id: str) -> bool:
        """
        Revise domain of candidate for given job.
        
        Returns:
            True if domain was revised
        """
        revised = False
        
        if job_id in self.domains[cid]:
            # Check if this value is still valid
            if not self._is_compatible(self.candidates[cid], self.jobs[job_id]):
                self.domains[cid].remove(job_id)
                revised = True
        
        return revised
    
    def solve(self, use_ac3: bool = True) -> Tuple[Dict[str, str], Dict]:
        """
        Solve the CSP and return results.
        
        Args:
            use_ac3: Whether to apply AC-3 preprocessing
            
        Returns:
            Tuple of (assignments, statistics)
        """
        # Initialize domains
        self.initialize_domains()
        
        stats = {
            'initial_domain_sizes': {cid: len(d) for cid, d in self.domains.items()},
            'ac3_applied': use_ac3,
            'total_candidates': len(self.candidates),
            'total_jobs': len(self.jobs)
        }
        
        # Apply AC-3 if requested
        if use_ac3:
            ac3_result = self.ac3()
            stats['ac3_consistent'] = ac3_result
            stats['domain_sizes_after_ac3'] = {cid: len(d) for cid, d in self.domains.items()}
            
            if not ac3_result:
                return {}, stats
        
        # Run backtracking search
        solution = self.backtracking_search()
        
        stats['solution_found'] = solution is not None
        stats['assignments'] = solution if solution else {}
        
        return solution or {}, stats
    
    def get_match_details(self, assignments: Dict[str, str]) -> List[Dict]:
        """Get detailed information about matches."""
        details = []
        
        for cand_id, job_id in assignments.items():
            candidate = self.candidates[cand_id]
            job = self.jobs[job_id]
            
            skill_match = len(candidate.skills & job.required_skills) / len(job.required_skills) if job.required_skills else 1.0
            
            details.append({
                'candidate_id': cand_id,
                'candidate_skills': list(candidate.skills)[:5],
                'experience': candidate.experience_years,
                'job_id': job_id,
                'job_title': job.title,
                'skill_match': f"{skill_match:.0%}",
                'meets_experience': candidate.experience_years >= job.min_experience
            })
        
        return sorted(details, key=lambda x: x['skill_match'], reverse=True)
    
    def print_solution(self, assignments: Dict[str, str], stats: Dict):
        """Pretty print the CSP solution."""
        print("\n" + "=" * 70)
        print("🧩 CSP SOLUTION - JOB-RESUME MATCHING")
        print("=" * 70)
        
        print(f"\n📊 Statistics:")
        print(f"   Total Candidates: {stats['total_candidates']}")
        print(f"   Total Jobs: {stats['total_jobs']}")
        print(f"   AC-3 Applied: {'Yes' if stats['ac3_applied'] else 'No'}")
        if stats.get('ac3_applied'):
            print(f"   AC-3 Consistent: {'Yes' if stats.get('ac3_consistent', True) else 'No'}")
        print(f"   Solution Found: {'Yes' if stats['solution_found'] else 'No'}")
        
        if assignments:
            print(f"\n📋 Assignments ({len(assignments)} matches):\n")
            
            details = self.get_match_details(assignments)
            for d in details:
                print(f"  ✓ {d['candidate_id']} → {d['job_title']}")
                print(f"    Skills: {', '.join(d['candidate_skills'])}")
                print(f"    Match: {d['skill_match']} | Exp: {d['experience']} yrs")
                print("-" * 50)
        else:
            print("\n❌ No valid assignment found satisfying all constraints.")
        
        print("\n" + "=" * 70)


def demo_csp():
    """Demonstrate CSP for job matching."""
    print("\n" + "=" * 70)
    print("CSP DEMONSTRATION - JOB-RESUME MATCHING")
    print("=" * 70)
    
    # Create CSP instance
    csp = JobResumeCSP()
    
    # Add job positions
    jobs = [
        JobPosition("J001", "Senior Data Scientist", 
                    {'python', 'machine learning', 'sql', 'tensorflow'}, 
                    min_experience=3, max_candidates=2),
        JobPosition("J002", "Backend Developer", 
                    {'java', 'spring', 'sql', 'aws'}, 
                    min_experience=2, max_candidates=2),
        JobPosition("J003", "ML Engineer", 
                    {'python', 'pytorch', 'docker', 'kubernetes'}, 
                    min_experience=2, max_candidates=1),
        JobPosition("J004", "Data Analyst", 
                    {'python', 'sql', 'tableau', 'excel'}, 
                    min_experience=1, max_candidates=2),
    ]
    
    for job in jobs:
        csp.add_job(job)
        print(f"📌 Added Job: {job.title}")
        print(f"   Required: {job.required_skills}")
        print(f"   Min Exp: {job.min_experience} years")
    
    # Add candidates
    candidates = [
        CandidateCSP("C001", "Alice", {'python', 'machine learning', 'sql', 'tensorflow', 'keras'}, 5),
        CandidateCSP("C002", "Bob", {'java', 'spring', 'sql', 'aws', 'docker'}, 4),
        CandidateCSP("C003", "Carol", {'python', 'pytorch', 'docker', 'kubernetes', 'aws'}, 3),
        CandidateCSP("C004", "David", {'python', 'sql', 'tableau', 'excel', 'power bi'}, 2),
        CandidateCSP("C005", "Eve", {'python', 'machine learning', 'sql', 'spark'}, 4),
        CandidateCSP("C006", "Frank", {'java', 'spring', 'microservices', 'aws'}, 3),
    ]
    
    print("\n" + "-" * 50)
    for cand in candidates:
        csp.add_candidate(cand)
        print(f"👤 Added Candidate: {cand.name}")
        print(f"   Skills: {cand.skills}")
        print(f"   Experience: {cand.experience_years} years")
    
    # Solve CSP
    print("\n🔄 Solving CSP with AC-3 + Backtracking...")
    assignments, stats = csp.solve(use_ac3=True)
    
    # Print solution
    csp.print_solution(assignments, stats)
    
    return csp, assignments, stats


if __name__ == "__main__":
    demo_csp()
