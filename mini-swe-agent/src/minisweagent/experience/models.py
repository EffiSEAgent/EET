"""
Data model definitions
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
import json


@dataclass
class Experience:
    """Agentless single-step execution experience record"""
    issue_id: str  # SWE-bench Issue ID
    issue_description: str  # Issue description
    task_summary: Union[str, List[str]]  # Task summary for a single step (can be string or list)
    confidence: int  # Confidence score (1-100)
    output: str = ""  # Step output result
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def __post_init__(self):
        """Validate data"""
        if not 1 <= self.confidence <= 100:
            raise ValueError(f"Confidence must be between 1-100, current value: {self.confidence}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        """Create object from dictionary"""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Experience':
        """Create object from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def __str__(self) -> str:
        """String representation"""
        task_str = self.task_summary
        if isinstance(task_str, list):
            task_str = " ".join(task_str)
        task_preview = task_str[:30] + "..." if len(task_str) > 30 else task_str
        return (f"Experience(issue_id={self.issue_id}, "
                f"task={task_preview}, "
                f"confidence={self.confidence})")

