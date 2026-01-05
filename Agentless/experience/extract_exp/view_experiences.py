"""
View and analyze extracted Experience
"""
import sys
import os
from pathlib import Path

# Add parent directory to path to import experience module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experience import ExperienceStore


def print_statistics(store: ExperienceStore):
    """Print statistics"""
    stats = store.get_statistics()
    
    print("\n" + "=" * 60)
    print("Experience Store Statistics")
    print("=" * 60)
    print(f"Total experiences:        {stats['total_experiences']}")
    print(f"Unique issues:            {stats['unique_issues']}")
    print(f"Average confidence:       {stats['avg_confidence']:.2f}")
    print(f"Confidence range:         {stats['min_confidence']} - {stats['max_confidence']}")
    print(f"Avg steps per issue:      {stats['avg_steps_per_issue']:.2f}")
    print("=" * 60)


def print_stage_distribution(store: ExperienceStore):
    """Print distribution by stage"""
    experiences = store.get_all()
    
    # Count by stage
    stage_counts = {}
    for exp in experiences:
        stage = exp.metadata.get("stage", "unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    
    print("\n" + "=" * 60)
    print("Experience Distribution by Stage")
    print("=" * 60)
    
    # Sort by count
    sorted_stages = sorted(stage_counts.items(), key=lambda x: x[1], reverse=True)
    for stage, count in sorted_stages:
        percentage = (count / len(experiences)) * 100
        print(f"{stage:<35} {count:>5} ({percentage:>5.1f}%)")
    
    print("=" * 60)


def print_confidence_distribution(store: ExperienceStore):
    """Print confidence distribution"""
    experiences = store.get_all()
    
    # Count by confidence range
    ranges = {
        "0-20": 0,
        "21-40": 0,
        "41-60": 0,
        "61-80": 0,
        "81-100": 0
    }
    
    for exp in experiences:
        conf = exp.confidence
        if conf <= 20:
            ranges["0-20"] += 1
        elif conf <= 40:
            ranges["21-40"] += 1
        elif conf <= 60:
            ranges["41-60"] += 1
        elif conf <= 80:
            ranges["61-80"] += 1
        else:
            ranges["81-100"] += 1
    
    print("\n" + "=" * 60)
    print("Confidence Distribution")
    print("=" * 60)
    
    for range_name, count in ranges.items():
        percentage = (count / len(experiences)) * 100 if len(experiences) > 0 else 0
        bar = "█" * int(percentage / 2)
        print(f"{range_name:>10}: {count:>5} ({percentage:>5.1f}%) {bar}")
    
    print("=" * 60)


def print_sample_experiences(store: ExperienceStore, n: int = 3):
    """Print sample experiences"""
    experiences = store.get_all()
    
    if not experiences:
        print("\nNo experiences found")
        return
    
    print("\n" + "=" * 60)
    print(f"Sample Experiences (first {n})")
    print("=" * 60)
    
    for i, exp in enumerate(experiences[:n], 1):
        print(f"\n--- Experience {i} ---")
        print(f"Issue ID:      {exp.issue_id}")
        print(f"Task:          {exp.task_summary[:80]}...")
        print(f"Confidence:    {exp.confidence}")
        print(f"Stage:         {exp.metadata.get('stage', 'unknown')}")
        print(f"Issue Desc:    {exp.issue_description[:100]}...")
        print(f"Output:        {exp.output[:200]}...")
        print("-" * 60)


def search_by_issue(store: ExperienceStore, issue_id: str):
    """Search by Issue ID"""
    experiences = store.get_by_issue_id(issue_id)
    
    if not experiences:
        print(f"\nNo experiences found for Issue ID: {issue_id}")
        return
    
    print(f"\nFound {len(experiences)} experiences for {issue_id}:")
    for i, exp in enumerate(experiences, 1):
        print(f"\n=== Step {i}: {exp.task_summary[:60]}... ===")
        print(f"Confidence: {exp.confidence}")
        print(f"Stage: {exp.metadata.get('stage', 'unknown')}")
        print(f"Output: {exp.output[:300]}...")
        print()


def list_all_issues(store: ExperienceStore):
    """List all Issues"""
    issues = store.get_issues()
    
    print(f"\nTotal {len(issues)} unique Issues:")
    print("=" * 60)
    
    for i, issue_id in enumerate(issues, 1):
        exp_count = len(store.get_by_issue_id(issue_id))
        print(f"{i:>4}. {issue_id:<40} ({exp_count} experiences)")


def export_to_json(store: ExperienceStore, output_file: str):
    """Export to JSON format"""
    output_path = Path(output_file)
    store.export_to_json(str(output_path))
    print(f"\nExported to: {output_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="View and analyze extracted Experience")
    parser.add_argument(
        "--file", 
        type=str, 
        default="/home3/yaoqi/Agentless-Exp/experience/extracted_experiences.jsonl",
        help="Experience file path"
    )
    parser.add_argument(
        "--action", 
        type=str, 
        default="stats",
        choices=["stats", "stages", "confidence", "samples", "search", "list", "export"],
        help="Action type"
    )
    parser.add_argument(
        "--issue-id", 
        type=str,
        help="Issue ID to search for"
    )
    parser.add_argument(
        "--output", 
        type=str,
        help="Output file path for export"
    )
    parser.add_argument(
        "--n", 
        type=int, 
        default=3,
        help="Number of samples to display"
    )
    
    args = parser.parse_args()
    
    # Create ExperienceStore
    store = ExperienceStore(args.file)
    
    # Execute action
    if args.action == "stats":
        print_statistics(store)
    elif args.action == "stages":
        print_stage_distribution(store)
    elif args.action == "confidence":
        print_confidence_distribution(store)
    elif args.action == "samples":
        print_sample_experiences(store, args.n)
    elif args.action == "search":
        if not args.issue_id:
            print("Error: --issue-id argument required")
            return
        search_by_issue(store, args.issue_id)
    elif args.action == "list":
        list_all_issues(store)
    elif args.action == "export":
        if not args.output:
            print("Error: --output argument required")
            return
        export_to_json(store, args.output)


if __name__ == "__main__":
    main()
