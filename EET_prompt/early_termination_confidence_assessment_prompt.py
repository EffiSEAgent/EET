"""Early Termination Confidence Assessment Prompt Template"""

EARLY_TERMINATION_CONFIDENCE_ASSESSMENT_PROMPT = """# Early Termination Confidence Assessment Prompt

You've reached a key milestone in your task. Please take a moment to assess your progress based on the complete context of your work so far, including:

- The issue description and requirements
- All the commands you've executed and their results
- Code changes you've made (if any)
- Tests or verification you've run (if any)
- Your understanding of the problem and solution

**Please provide your confidence assessment:**

**Confidence Score (0-100):**
Based on your understanding of the entire task context, how confident are you that the task is complete?

Consider:
- Have you addressed the core issue described in the PR?
- Are your changes appropriate and complete?
- Have you verified that your solution works correctly?
- Is the solution robust and handles edge cases?
- Have you checked that your fix logic is correct? (e.g., function parameters, return types, logic flow)

**Please respond with your confidence score using this exact format:**
CONFIDENCE_SCORE: <number>
Where <number> is an integer between 0 and 100.

Example: CONFIDENCE_SCORE: 85

After providing your score, continue with your next command as usual. The system will evaluate your score and provide guidance accordingly.
"""

