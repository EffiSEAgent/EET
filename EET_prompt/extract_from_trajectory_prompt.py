"""Extract from Trajectory Prompt Template"""

EXTRACT_FROM_TRAJECTORY_PROMPT = """EXTRACT_FROM_TRAJECTORY_PROMPT_TEMPLATE

You are an expert judge tasked with extracting valuable experiences from agent interaction dialogues. Your goal is to identify meaningful steps that contribute to solving programming tasks and extract structured experience records.

## Task
Analyze the conversation history and extract experience records for each significant step where the assistant interacts with the system. Focus on steps that:
1. Show clear reasoning and planning (THOUGHT section)
2. Execute meaningful commands
3. Produce useful outputs or insights
4. Contribute to solving the issue

## Input Format
You will receive:
- issue_id: The identifier of the issue being solved
- issue_description: Brief description of the problem
- trajectory: A list of execution trajectories (system, user, assistant)

## Output Format
For each significant assistant step, extract an experience record in JSON format. Return a JSON array of experience objects:
[
    {{
        "task_summary": "Brief summary of what this step accomplishes (max 200 chars)",
        "output": "Relevant output from the command execution (max 1000 chars, truncate if needed)",
        "reasoning": "Why this step is valuable and what makes it significant"
    }}
]

## Evaluation Criteria
1. Task Summary: Create a concise, descriptive summary (format: "Task Type: Brief description")
   - Task types: "Find Files", "Read Code", "Analyze Logic", "Modify Code", "Run Tests", "Debug Issue".
   - Description should capture the key intent and action
2. Output: Extract relevant output from the next user message (command execution result)
   - Focus on useful information, not error messages or empty outputs
   - Truncate to 1000 characters if too long
3. Reasoning: Explain why this step is valuable (optional but helpful)

## Filtering Guidelines
- Skip trivial steps (e.g., simple ls without clear purpose)
- Skip steps with empty or unhelpful outputs
- Focus on steps that show problem-solving progression
- Prioritize steps that lead to understanding or fixing the issue

## Example
Input:
- issue_id: "psf__requests-1963"
- issue_description: "Session.resolve_redirects copies original request..."
- trajectory: [system message, user prompt, assistant response with THOUGHT and command, user output...]

Output:
[
    {{
        "task_summary": "Read Code: Examine sessions.py to locate resolve_redirects method",
        "output": "[file content showing the method implementation]",
        "reasoning": "This step identifies the exact location of the bug, which is crucial for fixing it"
    }}
]

Now analyze the following conversation and extract experience records:

Issue ID: {issue_id}
Issue Description: {issue_description}

Trajectory:
{execution_trajectory}

Extract all valuable experience records as a JSON array.
"""

