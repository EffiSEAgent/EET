"""
Extract Experience from swe-bench-verified results
Filter out IDs in extracted_task_ids.txt
"""
import json
import os
import sys
from pathlib import Path
from typing import Set, List, Dict, Any

# Add parent directory to path to import experience module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experience import Experience, ExperienceStore
from datasets import load_dataset


class ExperienceExtractor:
    """Extract Experience from Agentless results"""
    
    # Task description mapping (from efficiency/*.py files)
    TASK_SUMMARIES = {
        "file_level": "You need to analyze a GitHub problem description and the repository structure, then identify and list (up to 5) the most relevant file paths that should be edited to fix the problem, ordered by importance and wrapped in a code block.",
        "file_level_combined": "You need to analyze a GitHub problem description and the repository structure, then identify and list (up to 5) the most relevant file paths that should be edited to fix the problem, ordered by importance and wrapped in a code block.",
        "file_level_irrelevant": "Identify and list all folders in the repository structure that are irrelevant to fixing the described GitHub problem.",
        "edit_location_samples": "Identify the exact classes, functions/methods in the given files that require editing to resolve the described GitHub problem, and list them in the required format.",
        "edit_location_individual": "Analyze the GitHub problem description and file skeleton, then list all relevant classes, functions, methods, or global variables that may require inspection or modification to fix the issue, formatted exactly as specified.",
        "related_elements": "Identify all relevant classes, functions, methods, or global variables in the provided file skeleton that may need inspection or modification to fix the described GitHub problem, and list them in the specified format.",
        "repair_sample_1": "Take an issue description and the corresponding source file content, identify and localize the root cause of the bug, and then generate precise editing instructions (in formats such as edit_file or search/replace blocks) with correct line ranges and indentation to modify the source code so that the issue is resolved.",
        "repair_sample_2": "Take an issue description and the corresponding source file content, identify and localize the root cause of the bug, and then generate precise editing instructions (in formats such as edit_file or search/replace blocks) with correct line ranges and indentation to modify the source code so that the issue is resolved.",
        "repair_sample_3": "Take an issue description and the corresponding source file content, identify and localize the root cause of the bug, and then generate precise editing instructions (in formats such as edit_file or search/replace blocks) with correct line ranges and indentation to modify the source code so that the issue is resolved.",
        "repair_sample_4": "Take an issue description and the corresponding source file content, identify and localize the root cause of the bug, and then generate precise editing instructions (in formats such as edit_file or search/replace blocks) with correct line ranges and indentation to modify the source code so that the issue is resolved.",
    }
    
    # Input key mapping - all stages now use "experience" field
    INPUT_KEYS = {
        "file_level": "experience",
        "file_level_combined": "experience",
        "file_level_irrelevant": "experience",
        "edit_location_samples": "experience",
        "edit_location_individual": "experience",
        "related_elements": "experience",
        "repair_sample_1": "experience",
        "repair_sample_2": "experience",
        "repair_sample_3": "experience",
        "repair_sample_4": "experience",
    }
    
    def __init__(
        self, 
        results_dir: str,
        extracted_ids_file: str,
        output_dir: str,
        dataset_name: str = "princeton-nlp/SWE-bench_Lite"
    ):
        """
        Initialize the extractor
        
        Args:
            results_dir: Results directory path (swe-bench-verified)
            extracted_ids_file: Extracted IDs file path
            output_dir: Output directory for Experience files (each stage has its own file)
            dataset_name: Dataset name
        """
        self.results_dir = Path(results_dir)
        self.extracted_ids_file = Path(extracted_ids_file)
        self.output_dir = Path(output_dir)
        self.dataset_name = dataset_name
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        print(f"Loading dataset: {dataset_name}")
        self.dataset = load_dataset(dataset_name, split="test")
        
        # Load extracted IDs
        self.extracted_ids = self._load_extracted_ids()
        print(f"Number of extracted IDs: {len(self.extracted_ids)}")
        
        # Store for each stage (created on demand)
        self.stores = {}
    
    def _get_store_for_stage(self, stage: str) -> ExperienceStore:
        """
        Get or create ExperienceStore for a specific stage
        
        Args:
            stage: Stage name
            
        Returns:
            ExperienceStore for this stage
        """
        if stage not in self.stores:
            output_file = self.output_dir / f"experiences_{stage}.jsonl"
            self.stores[stage] = ExperienceStore(str(output_file))
        return self.stores[stage]
    
    def _load_extracted_ids(self) -> Set[str]:
        """Load list of extracted IDs"""
        extracted_ids = set()
        if self.extracted_ids_file.exists():
            with open(self.extracted_ids_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        extracted_ids.add(line)
        print(f"Number of extracted IDs: {len(extracted_ids)}")
        return extracted_ids
    
    def _load_jsonl(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file"""
        data = []
        if not file_path.exists():
            return data
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return data
    
    def _get_problem_statement(self, instance_id: str) -> str:
        """Get problem statement"""
        for item in self.dataset:
            if item["instance_id"] == instance_id:
                return item["problem_statement"]
        return ""
    
    def _load_confidence_results(self, stage_dir: Path) -> Dict[str, int]:
        """Load confidence results"""
        confidence_file = stage_dir / "confidence_results.jsonl"
        confidence_map = {}
        
        if confidence_file.exists():
            data = self._load_jsonl(confidence_file)
            for item in data:
                instance_id = item.get("instance_id", "")
                confidence = item.get("confidence", 50)
                confidence_map[instance_id] = confidence
        
        return confidence_map
    
    def _format_output(self, output: Any, stage: str) -> str:
        """Format output"""
        if output is None:
            return ""
        
        if isinstance(output, str):
            return output
        
        if isinstance(output, list):
            if not output:
                return ""
            # For file lists, convert to string
            if stage in ["file_level", "file_level_combined", "file_level_irrelevant"]:
                return "\n".join(output)
            # For other lists, convert to JSON
            return json.dumps(output, ensure_ascii=False, indent=2)
        
        if isinstance(output, dict):
            return json.dumps(output, ensure_ascii=False, indent=2)
        
        return str(output)
    
    def extract_from_stage(self, stage: str, input_file: str = None) -> int:
        """
        Extract Experience from a stage
        
        Args:
            stage: Stage name (e.g. file_level, edit_location_samples, repair_sample_1, etc.)
            input_file: Input filename (default is experience_results.jsonl)
            
        Returns:
            Number of extracted experiences
        """
        stage_dir = self.results_dir / stage
        if not stage_dir.exists():
            print(f"Stage directory does not exist: {stage_dir}")
            return 0
        
        # Determine input file - now all stages use experience_results.jsonl
        if input_file is None:
            input_file = "experience_results.jsonl"
        
        input_path = stage_dir / input_file
        if not input_path.exists():
            print(f"Input file does not exist: {input_path}")
            return 0
        
        # Load confidence results
        confidence_map = self._load_confidence_results(stage_dir)
        
        # Load input data
        input_data = self._load_jsonl(input_path)
        print(f"\nProcessing stage: {stage}, total data: {len(input_data)}")
        
        # Get task description and input key
        task_summary = self.TASK_SUMMARIES.get(stage, f"{stage} stage task")
        input_key = self.INPUT_KEYS.get(stage, "output")
        
        # Extract Experience
        count = 0
        for item in input_data:
            instance_id = item.get("instance_id", "")
            if not instance_id:
                continue
            
            # Skip extracted IDs
            if instance_id in self.extracted_ids:
                continue
            
            # Get output
            output = item.get(input_key, "")
            if not output:
                continue
            
            # Get problem description
            issue_description = self._get_problem_statement(instance_id)
            if not issue_description:
                continue
            
            # Get confidence (if available)
            confidence = confidence_map.get(instance_id, 50)
            if confidence == 0:
                continue
            
            # Format output
            formatted_output = self._format_output(output, stage)
            
            # Create Experience
            exp = Experience(
                issue_id=instance_id,
                issue_description=issue_description,
                task_summary=task_summary,
                confidence=confidence,
                output=formatted_output,
                metadata={
                    "stage": stage,
                    "input_key": input_key,
                    "dataset": self.dataset_name
                }
            )
            
            # Add to stage-specific store
            store = self._get_store_for_stage(stage)
            store.add(exp)
            count += 1
        
        print(f"Extracted {count} experiences from {stage}")
        return count
    
    def extract_all(self) -> int:
        """Extract Experience from all stages"""
        total_count = 0
        
        # Define stages to process
        stages = [
            "file_level",
            "file_level_combined",
            "file_level_irrelevant",
            "related_elements",
            "edit_location_samples",
            "edit_location_individual",
            "repair_sample_1",
            # Add more repair samples if available
        ]
        
        # Check which stages actually exist
        available_stages = []
        for stage in stages:
            if (self.results_dir / stage).exists():
                available_stages.append(stage)
        
        # Check for repair_sample_2, 3, 4...
        for i in range(2, 10):
            stage = f"repair_sample_{i}"
            if (self.results_dir / stage).exists():
                available_stages.append(stage)
        
        print(f"\nAvailable stages: {available_stages}")
        
        # Extract from each stage
        for stage in available_stages:
            count = self.extract_from_stage(stage)
            total_count += count
        
        print(f"\nTotal extracted: {total_count} experiences")
        print(f"Output directory: {self.output_dir}")
        
        return total_count
    
    def get_statistics(self):
        """Get statistics for all stages"""
        print("\n=== Statistics ===")
        total_experiences = 0
        
        for stage, store in self.stores.items():
            stats = store.get_statistics()
            print(f"\n[{stage}]")
            print(f"  Total experiences: {stats['total_experiences']}")
            print(f"  Unique issues: {stats['unique_issues']}")
            if stats['total_experiences'] > 0:
                print(f"  Average confidence: {stats['avg_confidence']:.2f}")
                print(f"  Min confidence: {stats['min_confidence']}")
                print(f"  Max confidence: {stats['max_confidence']}")
                print(f"  Average steps per issue: {stats['avg_steps_per_issue']:.2f}")
            total_experiences += stats['total_experiences']
        
        print(f"\n=== Total across all stages ===")
        print(f"Total experiences: {total_experiences}")


def main():
    """Main function"""
    # Configure paths
    results_dir = "/home3/yaoqi/Agentless/results/swe-bench-lite"
    extracted_ids_file = "/home3/yaoqi/Agentless/verified_tasks.txt"
    output_dir = "/home3/yaoqi/Agentless-Exp/experience/extracted_experiences_lite"
    
    # Create extractor
    extractor = ExperienceExtractor(
        results_dir=results_dir,
        extracted_ids_file=extracted_ids_file,
        output_dir=output_dir
    )
    
    # Extract all experiences
    extractor.extract_all()
    
    # Show statistics
    extractor.get_statistics()


if __name__ == "__main__":
    main()

