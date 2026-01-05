# Confidence Evaluation Prompt

## Overview
This prompt is designed to guide an LLM to evaluate the confidence score and provide reasoning for experience records extracted from agent interaction trajectories. The LLM acts as a judge to assess how reliable and valuable each experience record is.

## Task
Given an experience record (excluding the confidence field), evaluate its reliability and quality, then assign a confidence score (1-100) and provide a clear reasoning for that score.

## Input Format
You will receive an experience record with the following fields:
- **issue_id**: The identifier of the issue being solved
- **issue_description**: Description of the problem/issue
- **task_summary**: A list of task steps that summarize the solution approach
- **created_at**: Timestamp when the experience was created
- **metadata**: Additional metadata (may include original_count, extraction_method, model, etc.)

## Output Format
Return a JSON object with the following structure:

```json
{
  "confidence": 85,
  "confidence_reason": "Clear explanation of why this confidence score was assigned"
}
```

## Evaluation Criteria

### Confidence Score Range (1-100)

#### High Confidence (80-100)
Assign when the experience record demonstrates:
- **Complete Solution Path**: The task_summary shows a logical, complete sequence from problem identification to solution
- **Clear Problem Understanding**: The issue_description is well-understood and addressed systematically
- **Actionable Steps**: Task steps are specific, concrete, and directly contribute to solving the issue
- **Proper Task Types**: Steps follow appropriate patterns (Find Files → Read Code → Analyze Logic → Modify Code → Run Tests → Submit)
- **Evidence of Success**: Metadata indicates successful completion (e.g., original_count suggests multiple steps were executed)

**Examples:**
- 90-100: Complete solution with clear progression, all critical steps present, evidence of successful completion
- 80-89: Good solution path with minor gaps or less clear progression

#### Medium Confidence (60-79)
Assign when the experience record shows:
- **Partial Solution**: Some key steps are present but sequence may be incomplete
- **Unclear Progression**: Task steps don't follow a clear logical flow
- **Missing Critical Steps**: Important steps (e.g., testing, verification) are absent
- **Vague Descriptions**: Task summaries are too generic or lack specificity
- **Limited Evidence**: Few steps or unclear whether solution was successful

**Examples:**
- 70-79: Reasonable approach but missing some steps or unclear completion
- 60-69: Basic approach present but significant gaps or unclear progression

#### Low Confidence (1-59)
Assign when the experience record has:
- **Incomplete Information**: Critical fields are missing or unclear
- **Poor Task Quality**: Task summaries are vague, incorrect, or don't address the issue
- **No Clear Solution Path**: Steps don't form a coherent approach
- **Suspicious Patterns**: Steps seem incorrect, irrelevant, or don't match the issue
- **Very Few Steps**: Insufficient information to assess quality

**Examples:**
- 40-59: Some relevant steps but major issues with quality or completeness
- 20-39: Minimal useful information, unclear if solution was attempted
- 1-19: Very poor quality, missing critical information, or clearly incorrect

### Confidence Reasoning Guidelines

Your confidence_reason should clearly explain:

1. **Completeness Assessment**
   - Does the task_summary cover all necessary steps to solve the issue?
   - Are there missing critical steps (e.g., testing, verification)?

2. **Quality Assessment**
   - Are the task steps specific and actionable?
   - Do they follow a logical progression?
   - Are task types appropriate for each step?

3. **Relevance Assessment**
   - Do the steps directly address the issue_description?
   - Is there alignment between the problem and the solution approach?

4. **Evidence Assessment**
   - What evidence suggests the solution was successful or attempted?
   - Are there indicators of completion (e.g., "Submit" step, original_count)?

5. **Overall Assessment**
   - What are the strengths of this experience record?
   - What are the weaknesses or concerns?
   - Why does this record deserve this particular confidence score?

## Example Evaluations

### Example 1: High Confidence

**Input:**
```json
{
  "issue_id": "pytest-dev__pytest-5495.traj",
  "issue_description": "Assertion-rewriting produces confusing messages for bytes: comparing b'' and b'42' shows \"Right contains more items, first extra item: 52\" (52 is ord('4')). Root cause: pytest treats bytes as sequences of integers and reports element values (numeric ordinals) in diffs instead of showing byte/character representation.",
  "task_summary": [
    "Find Files: Locate assertion-related source files under src/_pytest/assertion",
    "Read Code: Inspect src/_pytest/assertion/util.py to find assertrepr_compare and related helpers",
    "Modify Code: Treat bytes/bytearray as text for diffs and test the change",
    "Debug Issue: Reproduce AttributeError in attrs handling by invoking _compare_eq_cls directly",
    "Modify Code: Make attrs field-check robust (fallback from field.cmp to field.eq) and run targeted tests",
    "Debug Issue: Investigate regression in plugin-installation test by reproducing the failing test scenario",
    "Modify Code / Submit: Stage changes and show cached diff for review"
  ],
  "created_at": "2025-11-09T23:44:10.817428",
  "metadata": {
    "original_count": 7,
    "summarization_method": "llm",
    "model": "openai/gpt-5-mini"
  }
}
```

**Output:**
```json
{
  "confidence": 85,
  "confidence_reason": "This experience record demonstrates a complete and logical solution path. The task_summary shows clear progression: starting with file location, code inspection, modification, debugging, and final submission. All 7 steps are present and follow appropriate task types. The steps directly address the root cause (bytes treated as integer sequences) and include both the main fix and handling of edge cases (attrs handling, regression testing). The presence of a 'Submit' step and original_count of 7 suggests successful completion. The only minor concern is that some steps combine multiple actions (e.g., 'Modify Code & Run Tests'), but this doesn't significantly impact the overall quality."
}
```

### Example 2: Medium Confidence

**Input:**
```json
{
  "issue_id": "django__django-11910.traj",
  "issue_description": "When renaming a model's primary key, existing ForeignKey fields end up with their to_field parameter still set to the old field name, causing broken references after migrations.",
  "task_summary": [
    "Read Code: Inspect RenameField implementation in django/db/migrations/operations/fields.py",
    "Read Code: Inspect MigrationAutodetector handling of relations and to_field renames",
    "Modify Code: Patch autodetector to copy old to_field only if it was explicitly declared"
  ],
  "created_at": "2025-11-09T23:44:17.158063",
  "metadata": {
    "original_count": 3,
    "summarization_method": "llm",
    "model": "openai/gpt-5-mini"
  }
}
```

**Output:**
```json
{
  "confidence": 72,
  "confidence_reason": "The experience record shows a reasonable approach to solving the issue, with clear steps for understanding the problem (reading code) and implementing a fix. However, there are notable gaps: (1) No testing or verification steps are present, making it unclear if the solution was validated. (2) Only 3 steps are recorded, which seems minimal for a migration-related issue that typically requires more investigation. (3) The final step mentions patching but doesn't indicate completion or submission. The steps that are present are relevant and address the core issue (ForeignKey to_field not updating during PK rename), but the lack of testing and completion evidence reduces confidence."
}
```

### Example 3: Low Confidence

**Input:**
```json
{
  "issue_id": "example__project-123.traj",
  "issue_description": "Feature request to add new functionality",
  "task_summary": [
    "Find Files: Look for relevant files",
    "Modify Code: Make changes"
  ],
  "created_at": "2025-11-09T23:44:10.817428",
  "metadata": {
    "original_count": 2,
    "summarization_method": "llm"
  }
}
```

**Output:**
```json
{
  "confidence": 45,
  "confidence_reason": "This experience record has significant quality issues. The task_summary contains only 2 very generic steps with vague descriptions ('Look for relevant files', 'Make changes') that don't provide actionable information. The issue_description is also vague ('Feature request to add new functionality') without specific details. There's no evidence of code reading, analysis, testing, or verification steps. The record lacks the specificity and completeness needed to be useful. Without clear task descriptions and a complete solution path, it's difficult to assess whether this represents a successful solution approach."
}
```

## Evaluation Guidelines

1. **Be Consistent**: Similar quality records should receive similar confidence scores
2. **Consider Context**: Take into account the complexity of the issue when assessing completeness
3. **Prioritize Completeness**: A complete solution path is more valuable than partial information
4. **Value Specificity**: Specific, actionable task summaries indicate higher quality
5. **Look for Evidence**: Metadata and task patterns that suggest successful completion increase confidence
6. **Be Fair but Critical**: Don't inflate scores; be honest about gaps and weaknesses

## Notes

- Confidence scores should reflect the **reliability and usefulness** of the experience record, not just whether it looks complete
- A record with fewer but high-quality steps may score higher than one with many vague steps
- Consider the **typical complexity** of similar issues when assessing if steps are sufficient
- **Metadata clues** (like original_count, extraction_method) can provide context but shouldn't be the sole basis for scoring
- When in doubt between two score ranges, choose the lower one to be conservative

