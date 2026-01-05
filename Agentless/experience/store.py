"""
Experience storage and management
"""
import json
import os
from typing import List, Optional, Dict, Any, Callable
from pathlib import Path
import re
from collections import Counter, defaultdict
import math

from .models import Experience


class TextSimilarity:
    """Text similarity calculation"""
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        tokens = re.findall(r'\w+', text)
        return tokens
    
    @staticmethod
    def compute_tf(tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency (TF)"""
        counter = Counter(tokens)
        total = len(tokens)
        return {word: count / total for word, count in counter.items()}
    
    @staticmethod
    def compute_idf(documents: List[List[str]]) -> Dict[str, float]:
        """Compute inverse document frequency (IDF)"""
        num_docs = len(documents)
        word_doc_count = Counter()
        
        for doc in documents:
            unique_words = set(doc)
            word_doc_count.update(unique_words)
        
        idf = {}
        for word, count in word_doc_count.items():
            idf[word] = math.log(num_docs / count)
        
        return idf
    
    @staticmethod
    def compute_tfidf(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
        """Compute TF-IDF"""
        tf = TextSimilarity.compute_tf(tokens)
        tfidf = {}
        for word, tf_value in tf.items():
            tfidf[word] = tf_value * idf.get(word, 0)
        return tfidf
    
    @staticmethod
    def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Compute cosine similarity"""
        common_words = set(vec1.keys()) & set(vec2.keys())
        
        if not common_words:
            return 0.0
        
        dot_product = sum(vec1[word] * vec2[word] for word in common_words)
        norm1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        norm2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def simple_similarity(text1: str, text2: str) -> float:
        """Simple bag-of-words similarity (no corpus needed)"""
        tokens1 = set(TextSimilarity.tokenize(text1))
        tokens2 = set(TextSimilarity.tokenize(text2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union)


class ExperienceStore:
    """Experience storage manager"""
    
    def __init__(self, storage_path: str = None):
        """
        Initialize storage manager
        
        Args:
            storage_path: Storage file path, defaults to experiences.jsonl in current directory
        """
        if storage_path is None:
            storage_path = os.path.join(
                os.path.dirname(__file__), 
                'experiences.jsonl'
            )
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.storage_path.exists():
            self.storage_path.touch()
    
    def add(self, experience: Experience) -> None:
        """
        Add an experience record
        
        Args:
            experience: Experience object
        """
        with open(self.storage_path, 'a', encoding='utf-8') as f:
            f.write(experience.to_json() + '\n')
    
    def add_batch(self, experiences: List[Experience]) -> None:
        """
        Add experience records in batch
        
        Args:
            experiences: List of experience objects
        """
        with open(self.storage_path, 'a', encoding='utf-8') as f:
            for exp in experiences:
                f.write(exp.to_json() + '\n')
    
    def delete(self, condition: Callable[[Experience], bool]) -> int:
        """
        Delete experience records matching the condition
        
        Args:
            condition: Delete condition function that takes Experience object and returns bool
            
        Returns:
            Number of deleted records
        """
        experiences = self.get_all()
        original_count = len(experiences)
        
        experiences = [exp for exp in experiences if not condition(exp)]
        
        deleted_count = original_count - len(experiences)
        
        if deleted_count > 0:
            self._write_all(experiences)
        
        return deleted_count
    
    def delete_by_issue_id(self, issue_id: str) -> int:
        """
        Delete all experience records for a specific issue_id
        
        Args:
            issue_id: Issue ID
            
        Returns:
            Number of deleted records
        """
        return self.delete(lambda exp: exp.issue_id == issue_id)
    
    def update(self, condition: Callable[[Experience], bool], 
               update_func: Callable[[Experience], Experience]) -> int:
        """
        Update experience records matching the condition
        
        Args:
            condition: Update condition function
            update_func: Update function that takes old Experience and returns new Experience
            
        Returns:
            Number of updated records
        """
        experiences = self.get_all()
        updated_count = 0
        
        for i, exp in enumerate(experiences):
            if condition(exp):
                experiences[i] = update_func(exp)
                updated_count += 1
        
        if updated_count > 0:
            self._write_all(experiences)
        
        return updated_count
    
    def get_by_issue_id(self, issue_id: str) -> List[Experience]:
        """
        Get all experience records for a specific issue_id
        
        Args:
            issue_id: Issue ID
            
        Returns:
            List of experience objects
        """
        return self.filter(lambda exp: exp.issue_id == issue_id)
    
    def get_all(self) -> List[Experience]:
        """
        Get all experience records
        
        Returns:
            List of experience objects
        """
        experiences = []
        if not self.storage_path.exists():
            return experiences
            
        with open(self.storage_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    exp = Experience.from_json(line)
                    experiences.append(exp)
                except json.JSONDecodeError:
                    continue
        return experiences
    
    def search_by_task_similarity(
        self, 
        query: str, 
        top_k: int = 5,
        min_similarity: float = 0.0,
        use_tfidf: bool = False
    ) -> List[tuple[Experience, float]]:
        """
        Search experience records by task description text similarity
        
        Args:
            query: Query text (describing the task type to find)
            top_k: Return top k most similar results
            min_similarity: Minimum similarity threshold
            use_tfidf: Whether to use TF-IDF (True) or simple Jaccard similarity (False)
            
        Returns:
            List of (experience object, similarity score) tuples, sorted by similarity descending
        """
        experiences = self.get_all()
        if not experiences:
            return []
        
        results = []
        
        if use_tfidf:
            query_tokens = TextSimilarity.tokenize(query)
            
            all_docs = []
            for exp in experiences:
                tokens = TextSimilarity.tokenize(exp.task_summary)
                all_docs.append(tokens)
            
            all_docs.append(query_tokens)
            
            idf = TextSimilarity.compute_idf(all_docs)
            
            query_tfidf = TextSimilarity.compute_tfidf(query_tokens, idf)
            
            for exp, doc_tokens in zip(experiences, all_docs[:-1]):
                doc_tfidf = TextSimilarity.compute_tfidf(doc_tokens, idf)
                similarity = TextSimilarity.cosine_similarity(query_tfidf, doc_tfidf)
                
                if similarity >= min_similarity:
                    results.append((exp, similarity))
        else:
            for exp in experiences:
                similarity = TextSimilarity.simple_similarity(query, exp.task_summary)
                
                if similarity >= min_similarity:
                    results.append((exp, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def search_by_issue_similarity(
        self, 
        query: str, 
        top_k: int = 5,
        min_similarity: float = 0.0,
        use_tfidf: bool = False
    ) -> List[tuple[Experience, float]]:
        """
        Search experience records by issue description text similarity
        
        Args:
            query: Query text
            top_k: Return top k most similar results
            min_similarity: Minimum similarity threshold
            use_tfidf: Whether to use TF-IDF
            
        Returns:
            List of (experience object, similarity score) tuples
        """
        experiences = self.get_all()
        if not experiences:
            return []
        
        results = []
        
        if use_tfidf:
            query_tokens = TextSimilarity.tokenize(query)
            all_docs = [TextSimilarity.tokenize(exp.issue_description) 
                       for exp in experiences]
            all_docs.append(query_tokens)
            
            idf = TextSimilarity.compute_idf(all_docs)
            query_tfidf = TextSimilarity.compute_tfidf(query_tokens, idf)
            
            for exp, doc_tokens in zip(experiences, all_docs[:-1]):
                doc_tfidf = TextSimilarity.compute_tfidf(doc_tokens, idf)
                similarity = TextSimilarity.cosine_similarity(query_tfidf, doc_tfidf)
                
                if similarity >= min_similarity:
                    results.append((exp, similarity))
        else:
            for exp in experiences:
                similarity = TextSimilarity.simple_similarity(query, exp.issue_description)
                
                if similarity >= min_similarity:
                    results.append((exp, similarity))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def group_by_task_similarity(
        self,
        similarity_threshold: float = 0.3,
        use_tfidf: bool = True
    ) -> List[List[Experience]]:
        """
        Group similar task steps together
        
        Args:
            similarity_threshold: Similarity threshold
            use_tfidf: Whether to use TF-IDF
            
        Returns:
            List of grouped experience lists
        """
        experiences = self.get_all()
        if not experiences:
            return []
        
        groups = []
        used = set()
        
        for i, exp in enumerate(experiences):
            if i in used:
                continue
            
            group = [exp]
            used.add(i)
            
            for j, other_exp in enumerate(experiences):
                if j in used:
                    continue
                
                if use_tfidf:
                    tokens1 = TextSimilarity.tokenize(exp.task_summary)
                    tokens2 = TextSimilarity.tokenize(other_exp.task_summary)
                    
                    all_docs = [tokens1, tokens2]
                    idf = TextSimilarity.compute_idf(all_docs)
                    
                    tfidf1 = TextSimilarity.compute_tfidf(tokens1, idf)
                    tfidf2 = TextSimilarity.compute_tfidf(tokens2, idf)
                    
                    similarity = TextSimilarity.cosine_similarity(tfidf1, tfidf2)
                else:
                    similarity = TextSimilarity.simple_similarity(
                        exp.task_summary, 
                        other_exp.task_summary
                    )
                
                if similarity >= similarity_threshold:
                    group.append(other_exp)
                    used.add(j)
            
            groups.append(group)
        
        groups.sort(key=len, reverse=True)
        return groups
    
    def get_issues(self) -> List[str]:
        """
        Get all unique Issue IDs
        
        Returns:
            List of Issue IDs
        """
        experiences = self.get_all()
        return sorted(list(set(exp.issue_id for exp in experiences)))
    
    def filter(self, condition: Callable[[Experience], bool]) -> List[Experience]:
        """
        Filter experience records by custom condition
        
        Args:
            condition: Filter function that takes Experience object and returns bool
            
        Returns:
            List of matching experience objects
        """
        experiences = self.get_all()
        return [exp for exp in experiences if condition(exp)]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics
        
        Returns:
            Statistics dictionary
        """
        experiences = self.get_all()
        
        if not experiences:
            return {
                'total_experiences': 0,
                'unique_issues': 0,
                'avg_confidence': 0,
                'min_confidence': 0,
                'max_confidence': 0,
                'steps_per_issue': {}
            }
        
        confidences = [exp.confidence for exp in experiences]
        issue_counts = Counter(exp.issue_id for exp in experiences)
        
        return {
            'total_experiences': len(experiences),
            'unique_issues': len(issue_counts),
            'avg_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'avg_steps_per_issue': sum(issue_counts.values()) / len(issue_counts),
            'steps_per_issue': dict(issue_counts)
        }
    
    def _write_all(self, experiences: List[Experience]) -> None:
        """
        Write all experience records to file
        
        Args:
            experiences: List of experience objects
        """
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            for exp in experiences:
                f.write(exp.to_json() + '\n')
    
    def clear(self) -> None:
        """Clear all experience records"""
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            f.write('')
    
    def export_to_json(self, output_path: str) -> None:
        """
        Export all experience records to JSON file
        
        Args:
            output_path: Output file path
        """
        experiences = self.get_all()
        data = [exp.to_dict() for exp in experiences]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def import_from_json(self, input_path: str, merge: bool = False) -> None:
        """
        Import experience records from JSON file
        
        Args:
            input_path: Input file path
            merge: Whether to merge with existing records (True) or replace (False)
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        new_experiences = [Experience.from_dict(item) for item in data]
        
        if merge:
            existing = self.get_all()
            existing.extend(new_experiences)
            self._write_all(existing)
        else:
            self._write_all(new_experiences)
