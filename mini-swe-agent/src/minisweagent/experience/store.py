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
    """经验存储管理器"""
    
    def __init__(self, storage_path: str = None):
        """
        初始化存储管理器
        
        Args:
            storage_path: 存储文件路径，默认为当前目录下的 experiences.jsonl
        """
        if storage_path is None:
            storage_path = os.path.join(
                os.path.dirname(__file__), 
                'experiences.jsonl'
            )
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 确保文件存在
        if not self.storage_path.exists():
            self.storage_path.touch()
    
    def add(self, experience: Experience) -> None:
        """
        添加一条经验记录
        
        Args:
            experience: 经验对象
        """
        # 追加到文件
        with open(self.storage_path, 'a', encoding='utf-8') as f:
            f.write(experience.to_json() + '\n')
    
    def add_batch(self, experiences: List[Experience]) -> None:
        """
        批量添加经验记录
        
        Args:
            experiences: 经验对象列表
        """
        with open(self.storage_path, 'a', encoding='utf-8') as f:
            for exp in experiences:
                f.write(exp.to_json() + '\n')
    
    def delete(self, condition: Callable[[Experience], bool]) -> int:
        """
        删除满足条件的经验记录
        
        Args:
            condition: 删除条件函数，接收 Experience 对象，返回 bool
            
        Returns:
            删除的记录数
        """
        experiences = self.get_all()
        original_count = len(experiences)
        
        # 过滤掉要删除的记录
        experiences = [exp for exp in experiences if not condition(exp)]
        
        deleted_count = original_count - len(experiences)
        
        if deleted_count > 0:
            # 重写文件
            self._write_all(experiences)
        
        return deleted_count
    
    def delete_by_issue_id(self, issue_id: str) -> int:
        """
        删除指定 issue_id 的所有经验记录
        
        Args:
            issue_id: Issue ID
            
        Returns:
            删除的记录数
        """
        return self.delete(lambda exp: exp.issue_id == issue_id)
    
    def update(self, condition: Callable[[Experience], bool], 
               update_func: Callable[[Experience], Experience]) -> int:
        """
        更新满足条件的经验记录
        
        Args:
            condition: 更新条件函数
            update_func: 更新函数，接收旧的 Experience，返回新的 Experience
            
        Returns:
            更新的记录数
        """
        experiences = self.get_all()
        updated_count = 0
        
        # 更新匹配的记录
        for i, exp in enumerate(experiences):
            if condition(exp):
                experiences[i] = update_func(exp)
                updated_count += 1
        
        if updated_count > 0:
            # 重写文件
            self._write_all(experiences)
        
        return updated_count
    
    def get_by_issue_id(self, issue_id: str) -> List[Experience]:
        """
        根据 issue_id 获取所有相关的经验记录
        
        Args:
            issue_id: Issue ID
            
        Returns:
            经验对象列表
        """
        return self.filter(lambda exp: exp.issue_id == issue_id)
    
    def get_all(self) -> List[Experience]:
        """
        获取所有经验记录
        
        Returns:
            经验对象列表
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
                    continue  # 跳过无效的行
        return experiences
    
    def search_by_task_similarity(
        self, 
        query: str, 
        top_k: int = 5,
        min_similarity: float = 0.0,
        use_tfidf: bool = False
    ) -> List[tuple[Experience, float]]:
        """
        基于任务描述的文本相似度搜索经验记录
        
        Args:
            query: 查询文本（描述要查找的任务类型）
            top_k: 返回前 k 个最相似的结果
            min_similarity: 最小相似度阈值
            use_tfidf: 是否使用 TF-IDF（True）或简单 Jaccard 相似度（False）
            
        Returns:
            (经验对象, 相似度分数) 的列表，按相似度降序排列
        """
        experiences = self.get_all()
        if not experiences:
            return []
        
        results = []
        
        if use_tfidf:
            # 使用 TF-IDF
            query_tokens = TextSimilarity.tokenize(query)
            
            # 构建所有文档的 tokens（使用 task_summary）
            all_docs = []
            for exp in experiences:
                # Handle task_summary as list or string
                task_text = exp.task_summary
                if isinstance(task_text, list):
                    task_text = " ".join(task_text)
                tokens = TextSimilarity.tokenize(task_text)
                all_docs.append(tokens)
            
            # 添加查询文档
            all_docs.append(query_tokens)
            
            # 计算 IDF
            idf = TextSimilarity.compute_idf(all_docs)
            
            # 计算查询的 TF-IDF
            query_tfidf = TextSimilarity.compute_tfidf(query_tokens, idf)
            
            # 计算每个文档的相似度
            for exp, doc_tokens in zip(experiences, all_docs[:-1]):
                doc_tfidf = TextSimilarity.compute_tfidf(doc_tokens, idf)
                similarity = TextSimilarity.cosine_similarity(query_tfidf, doc_tfidf)
                
                if similarity >= min_similarity:
                    results.append((exp, similarity))
        else:
            # 使用简单的 Jaccard 相似度
            for exp in experiences:
                # Handle task_summary as list or string
                task_text = exp.task_summary
                if isinstance(task_text, list):
                    task_text = " ".join(task_text)
                similarity = TextSimilarity.simple_similarity(query, task_text)
                
                if similarity >= min_similarity:
                    results.append((exp, similarity))
        
        # 按相似度降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前 k 个
        return results[:top_k]
    
    def search_by_issue_similarity(
        self, 
        query: str, 
        top_k: int = 5,
        min_similarity: float = 0.0,
        use_tfidf: bool = False
    ) -> List[tuple[Experience, float]]:
        """
        基于 Issue 描述的文本相似度搜索经验记录
        
        Args:
            query: 查询文本
            top_k: 返回前 k 个最相似的结果
            min_similarity: 最小相似度阈值
            use_tfidf: 是否使用 TF-IDF
            
        Returns:
            (经验对象, 相似度分数) 的列表
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
        将相似的任务步骤聚合在一起
        
        Args:
            similarity_threshold: 相似度阈值
            use_tfidf: 是否使用 TF-IDF
            
        Returns:
            聚合后的经验组列表
        """
        experiences = self.get_all()
        if not experiences:
            return []
        
        # 简单的聚类算法
        groups = []
        used = set()
        
        for i, exp in enumerate(experiences):
            if i in used:
                continue
            
            # 创建新组
            group = [exp]
            used.add(i)
            
            # 查找相似的经验
            for j, other_exp in enumerate(experiences):
                if j in used:
                    continue
                
                if use_tfidf:
                    # 使用 TF-IDF 计算相似度
                    # Handle task_summary as list or string
                    task_text1 = exp.task_summary
                    if isinstance(task_text1, list):
                        task_text1 = " ".join(task_text1)
                    task_text2 = other_exp.task_summary
                    if isinstance(task_text2, list):
                        task_text2 = " ".join(task_text2)
                    
                    tokens1 = TextSimilarity.tokenize(task_text1)
                    tokens2 = TextSimilarity.tokenize(task_text2)
                    
                    all_docs = [tokens1, tokens2]
                    idf = TextSimilarity.compute_idf(all_docs)
                    
                    tfidf1 = TextSimilarity.compute_tfidf(tokens1, idf)
                    tfidf2 = TextSimilarity.compute_tfidf(tokens2, idf)
                    
                    similarity = TextSimilarity.cosine_similarity(tfidf1, tfidf2)
                else:
                    # Handle task_summary as list or string
                    task_text1 = exp.task_summary
                    if isinstance(task_text1, list):
                        task_text1 = " ".join(task_text1)
                    task_text2 = other_exp.task_summary
                    if isinstance(task_text2, list):
                        task_text2 = " ".join(task_text2)
                    
                    similarity = TextSimilarity.simple_similarity(
                        task_text1, 
                        task_text2
                    )
                
                if similarity >= similarity_threshold:
                    group.append(other_exp)
                    used.add(j)
            
            groups.append(group)
        
        # 按组大小降序排序
        groups.sort(key=len, reverse=True)
        return groups
    
    def get_issues(self) -> List[str]:
        """
        获取所有唯一的 Issue ID
        
        Returns:
            Issue ID 列表
        """
        experiences = self.get_all()
        return sorted(list(set(exp.issue_id for exp in experiences)))
    
    def filter(self, condition: Callable[[Experience], bool]) -> List[Experience]:
        """
        根据自定义条件过滤经验记录
        
        Args:
            condition: 过滤函数，接收 Experience 对象，返回 bool
            
        Returns:
            符合条件的经验对象列表
        """
        experiences = self.get_all()
        return [exp for exp in experiences if condition(exp)]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
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
        重写所有经验记录到文件
        
        Args:
            experiences: 经验对象列表
        """
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            for exp in experiences:
                f.write(exp.to_json() + '\n')
    
    def clear(self) -> None:
        """清空所有经验记录"""
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            f.write('')
    
    def export_to_json(self, output_path: str) -> None:
        """
        导出所有经验记录到 JSON 文件
        
        Args:
            output_path: 输出文件路径
        """
        experiences = self.get_all()
        data = [exp.to_dict() for exp in experiences]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def import_from_json(self, input_path: str, merge: bool = False) -> None:
        """
        从 JSON 文件导入经验记录
        
        Args:
            input_path: 输入文件路径
            merge: 是否合并到现有记录（True）或替换（False）
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

