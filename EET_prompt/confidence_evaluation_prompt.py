"""Confidence Evaluation Prompt Template"""

CONFIDENCE_EVALUATION_PROMPT = """# Confidence Evaluation Prompt

## Task
Evaluate experience records extracted from agent trajectories. Assign a confidence score (1-100) and provide reasoning.

## Input Fields
- issue_id: Issue identifier
- issue_description: Problem description
- task_summary: List of solution steps
- created_at: Timestamp
- metadata: Additional context (original_count, extraction_method, model, etc.)

## Output Format
{{
    "confidence": 85,
    "confidence_reason": "Brief explanation for the score"
}}

## Confidence Score Ranges

### High (80-100)
- Complete solution path with logical progression
- Clear problem understanding and actionable steps
- Proper task flow (Find → Read → Analyze → Modify → Test → Submit)
- Evidence of successful completion

### Medium (60-79)
- Partial solution with some gaps
- Unclear progression or missing critical steps (e.g., testing)
- Generic descriptions lacking specificity
- Limited completion evidence

### Low (1-59)
- Incomplete or unclear information
- Vague/incorrect task summaries
- No coherent solution path
- Very few steps or major quality issues

## Evaluation Criteria
Your confidence_reason should address:
1. Completeness: Are all necessary steps present?
2. Quality: Are steps specific, actionable, and logical?
3. Relevance: Do steps directly address the issue?
4. Evidence: Are there indicators of successful completion?

## Example: High Confidence (85)
Input:
{{
    "issue_id": "pytest-dev__pytest-5495.traj",
    "issue_description": "Assertion rewriting shows confusing messages for bytes (treats as integer sequences)",
    "task_summary": [
        "Find Files: Locate assertion files under src/_pytest/assertion",
        "Read Code: Inspect src/_pytest/assertion/util.py for assertrepr_compare",
        "Modify Code: Treat bytes/bytearray as text for diffs and test",
        "Debug Issue: Fix AttributeError in attrs handling",
        "Modify Code: Make attrs field-check robust, run tests",
        "Debug Issue: Investigate plugin-installation test regression",
        "Modify Code / Submit: Stage changes and review"
    ],
    "metadata": {{"original_count": 7}}
}}

Output:
{{
    "confidence": 85,
    "confidence_reason": "Complete solution path with 7 logical steps: file location → inspection → modification → debugging → submission. Steps directly address root cause (bytes as integers). Includes edge case handling and testing. Submit step and original_count=7 indicate completion."
}}

## Guidelines
- Be consistent: Similar quality → similar scores
- Prioritize completeness: Full solution paths are more valuable
- Value specificity: Detailed steps indicate higher quality
- Be conservative: When uncertain, choose the lower score range
- Consider issue complexity when assessing sufficiency
"""

