instruction = """
You are now required to solve a SAT (Boolean Satisfiability) problem. The problem is provided in JSON format, containing the following fields:

- "n_sat": The number of variables in each clause (n-SAT problem).
- "k": The total number of distinct variables in the problem.
- "clause": A string representation of the SAT formula, where clauses are separated by " & " (representing logical AND). Within each clause, variables are combined using concatenation (representing logical OR). A negation is indicated by "!" before a variable.

Your task is to provide a valid solution. The answer is a number of length k representing the truth values of the variables in order (1 for true, 0 for false). If there are multiple solutions, provide any one of them.
Please reason step by step, and put your final answer within \boxed{}.

**Example**
{"n_sat": 3, "k": 4, "clause": "!B!C!D & A!B!D & AB!D"}
**Final Answer**
\boxed{1101}
"""

problem = """
Below is the SAT problem you need to solve:
{{"n_sat": {n_sat}, "k": {k}, "clause": "{clause}"}}
"""


def format_prompt(sample: dict) -> str:
    clause = sample["clause"]
    n_sat = sample["n_sat"]
    k = sample["k"]
    
    prompt = f"{instruction.strip()}\n{problem.format(n_sat=n_sat, k=k, clause=clause).strip()}"
    return prompt