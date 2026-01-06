"""Patch Selection Confidence Evaluation Prompt Template"""

PATCH_CONFIDENCE_PROMPT = """# Patch Selection Confidence Evaluation Prompt

You are an expert reviewer tasked with evaluating the confidence that a code patch solves a given problem.

## Task
Based on the problem description, execution trajectory, code patch, and reference experiences from similar previous tasks, evaluate the confidence (0-100) that this patch correctly solves the problem. Use the provided experiences as reference to inform your evaluation, but adapt your assessment to the current problem's specific context.

## Input
- Problem description: {problem_statement}
- Code patch:
{patch_content}
- Trajectory:
{execution_trajectory}
- Experience:
{experience_object}

The experiences above are from similar previous tasks, which may help guide your evaluation approach.

## Evaluation Criteria
Please evaluate from the following aspects:
1. Relevance: Does the patch directly address the problem described?
2. Completeness: Does the patch completely solve the problem, or only partially?
3. Correctness: Is the patch's modification logic correct? Does it introduce new issues?
4. Reasonableness: Are the code modifications reasonable and follow best practices?
5. Experience Alignment: How does this patch compare to successful solutions from similar problems in the provided experiences? Consider patterns, approaches, and solution quality from the reference experiences.

## Output Format
Please return only a JSON object with the following fields:
{{
    "confidence": 85,
    "reasoning": "The patch directly addresses the problem described, the modification logic is correct, and the code is reasonable."
}}

Where:
- confidence: Integer, range 0-100, representing confidence level
- reasoning: String, brief explanation of the evaluation

Now please evaluate this patch:
"""

