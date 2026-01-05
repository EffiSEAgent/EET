"""
Data model definitions
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from datetime import datetime
import json


@dataclass
class Experience:
    """Agentless single-step execution experience record"""
    issue_id: str  # SWE-bench Issue ID
    issue_description: str  # Issue description
    task_summary: str  # Task summary for a single step
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
        data = data.copy()
        if isinstance(data.get('task_summary'), list):
            summmary = "\n".join([f"{i}. {step}" for i, step in enumerate(data['task_summary'], 1)])
            data['task_summary'] = summmary
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
        return (f"Experience(issue_id={self.issue_id}, "
                f"task={self.task_summary[:30]}..., "
                f"confidence={self.confidence})")

