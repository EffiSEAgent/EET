"""
Usage examples
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experience import Experience, ExperienceStore


def main():
    store = ExperienceStore()
    
    store.clear()
    
    print("=== Agentless Experience Recording System Demo ===\n")
    
    # 1. Create and add experience records (each record is one step)
    print("1. Add experience records (each record represents one step)")
    
    # Django Issue - multiple steps
    django_steps = [
        Experience(
            issue_id="django__django-12345",
            issue_description="Fix Django ORM query performance issue with N+1 queries when using select_related",
            task_summary="Locate the problematic file in django/db/models/query.py",
            confidence=85,
            output="Successfully located file django/db/models/query.py:245, found select_related method"
        ),
        Experience(
            issue_id="django__django-12345",
            issue_description="Fix Django ORM query performance issue with N+1 queries when using select_related",
            task_summary="Analyze the implementation logic of select_related",
            confidence=75,
            output="Found that each related object triggers a separate query in the loop, causing N+1 problem"
        ),
        Experience(
            issue_id="django__django-12345",
            issue_description="Fix Django ORM query performance issue with N+1 queries when using select_related",
            task_summary="Modify the query optimizer code to fix the issue",
            confidence=90,
            output="Modified code to use JOIN statement to fetch all related objects at once, tests passed"
        ),
    ]
    
    # Flask Issue - multiple steps
    flask_steps = [
        Experience(
            issue_id="flask__flask-5678",
            issue_description="Flask routing system has security vulnerability when handling regex patterns",
            task_summary="Review the routing matching code implementation",
            confidence=70,
            output="Found in flask/routing.py that user input is directly used for regex compilation"
        ),
        Experience(
            issue_id="flask__flask-5678",
            issue_description="Flask routing system has security vulnerability when handling regex patterns",
            task_summary="Identify unescaped regex input vulnerability",
            confidence=80,
            output="Confirmed vulnerability: malicious users can inject regex causing ReDoS attacks"
        ),
        Experience(
            issue_id="flask__flask-5678",
            issue_description="Flask routing system has security vulnerability when handling regex patterns",
            task_summary="Implement input validation and escaping mechanism",
            confidence=85,
            output="Added re.escape() to handle user input and implemented whitelist validation"
        ),
    ]
    
    # Requests Issue - multiple steps
    requests_steps = [
        Experience(
            issue_id="requests__requests-9999",
            issue_description="Requests library has issues with HTTPS certificate verification",
            task_summary="Locate the certificate verification module",
            confidence=88,
            output="Found cert_verify method in requests/adapters.py"
        ),
        Experience(
            issue_id="requests__requests-9999",
            issue_description="Requests library has issues with HTTPS certificate verification",
            task_summary="Fix certificate chain verification logic",
            confidence=92,
            output="Updated certificate chain verification order, now correctly handles intermediate certificates"
        ),
    ]
    
    # Batch add all steps
    all_steps = django_steps + flask_steps + requests_steps
    store.add_batch(all_steps)
    print(f"✓ Successfully added {len(all_steps)} experience records")
    print()
    
    # 2. Query steps for a specific Issue
    print("2. Query all steps for a specific Issue")
    issue_id = "django__django-12345"
    django_experiences = store.get_by_issue_id(issue_id)
    print(f"Issue {issue_id} has {len(django_experiences)} steps:")
    for i, exp in enumerate(django_experiences, 1):
        print(f"  Step {i}: {exp.task_summary} (confidence: {exp.confidence})")
        print(f"          Output: {exp.output[:60]}...")
    print()
    
    # 3. Search by task similarity (find similar steps across Issues)
    print("3. Search by task similarity - Find 'locate' related steps (Jaccard)")
    query = "locate file find module"
    results = store.search_by_task_similarity(query, top_k=5, use_tfidf=False)
    print(f"Query: '{query}'")
    for exp, score in results:
        print(f"  Similarity {score:.4f}: [{exp.issue_id}] {exp.task_summary}")
    print()
    
    # 4. Search by task similarity (TF-IDF method)
    print("4. Search by task similarity - TF-IDF method")
    query = "modify code fix implementation"
    results = store.search_by_task_similarity(query, top_k=5, use_tfidf=True)
    print(f"Query: '{query}'")
    for exp, score in results:
        print(f"  Similarity {score:.4f}: [{exp.issue_id}] {exp.task_summary}")
    print()
    
    # 5. Search by Issue description similarity
    print("5. Search by Issue description - Find performance related problems")
    query = "performance optimization query speed"
    results = store.search_by_issue_similarity(query, top_k=3, use_tfidf=False)
    print(f"Query: '{query}'")
    for exp, score in results:
        print(f"  Similarity {score:.4f}: [{exp.issue_id}] {exp.issue_description[:60]}...")
    print()
    
    # 6. Group similar task steps
    print("6. Group similar task steps")
    groups = store.group_by_task_similarity(similarity_threshold=0.2, use_tfidf=False)
    print(f"Found {len(groups)} task groups:")
    for i, group in enumerate(groups, 1):
        if len(group) > 1:  # Only show groups with multiple steps
            print(f"\n  Group {i} ({len(group)} similar steps):")
            for exp in group:
                print(f"    - [{exp.issue_id}] {exp.task_summary}")
    print()
    
    # 7. Filter high confidence steps
    print("7. Filter high confidence steps (confidence >= 85)")
    high_confidence = store.filter(lambda exp: exp.confidence >= 85)
    print(f"Found {len(high_confidence)} high confidence records:")
    for exp in high_confidence:
        print(f"  - [{exp.issue_id}] {exp.task_summary} (confidence: {exp.confidence})")
        print(f"    Output: {exp.output[:60]}...")
    print()
    
    # 8. Get all unique Issue IDs
    print("8. Get all unique Issue IDs")
    issues = store.get_issues()
    print(f"Total {len(issues)} different Issues:")
    for issue_id in issues:
        count = len(store.get_by_issue_id(issue_id))
        print(f"  - {issue_id} ({count} steps)")
    print()
    
    # 9. Get statistics
    print("9. Statistics")
    stats = store.get_statistics()
    print(f"  Total experiences: {stats['total_experiences']}")
    print(f"  Unique issues: {stats['unique_issues']}")
    print(f"  Average confidence: {stats['avg_confidence']:.2f}")
    print(f"  Min confidence: {stats['min_confidence']}")
    print(f"  Max confidence: {stats['max_confidence']}")
    print(f"  Average steps per issue: {stats['avg_steps_per_issue']:.2f}")
    print()
    
    # 10. Update records
    print("10. Update specific record's confidence and output")
    updated = store.update(
        condition=lambda exp: exp.issue_id == "flask__flask-5678" and "Review" in exp.task_summary,
        update_func=lambda exp: Experience(
            issue_id=exp.issue_id,
            issue_description=exp.issue_description,
            task_summary=exp.task_summary,
            confidence=95,  # Update confidence
            output=exp.output + " [Verified and confirmed]",  # Update output
            created_at=exp.created_at,
            metadata=exp.metadata
        )
    )
    print(f"✓ Updated {updated} record(s)")
    print()
    
    # 11. Export to JSON
    print("11. Export to JSON file")
    export_path = os.path.join(os.path.dirname(__file__), "experiences_export.json")
    store.export_to_json(export_path)
    print(f"✓ Exported to: {export_path}")
    print()
    
    # 12. Delete all records of a specific Issue
    print("12. Delete all records of a specific Issue")
    issue_to_delete = "requests__requests-9999"
    deleted_count = store.delete_by_issue_id(issue_to_delete)
    print(f"✓ Deleted {deleted_count} record(s) (Issue: {issue_to_delete})")
    remaining = store.get_all()
    print(f"  Remaining {len(remaining)} records")
    print()
    
    print("=== Demo Completed ===")


if __name__ == "__main__":
    main()
