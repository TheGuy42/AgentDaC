from src.configs.markers import Markers

def format_prompt(sample: dict) -> str:
    # base model doesn't work well with this instruction
    # instruction = (
    #     "The final answer should be written as valid LaTeX equation, starting with $ and ending with $. "
    #     "It should contain only the final result, without any additional text or explanation. "
    #     "Final answer format examples: $42$, $1,2,3,4$, $(1,2)$, $x^2$, $y=1$, $\\frac{1}{2}$, $\\sqrt{2} \\pi$, $\\text{Michael}$, $\\text{no}$, and so on."
    # )

    instruction = ""

    # TODO: find a unified instruction that works for both base model and task agents.
    
    # TODO: maybe we should simplify the system prompt of the task agents, just let them
    # know they can delegate tasks via <task> </task> and that it should contain the full info
    # we'll let the model figure the rest by itself. 
    # and also that it may ask for clarifications if needed.
    # and that the part containing the actual answer must be within <answer> <\answer>
    
    # official Qwen instruction for math problems: https://huggingface.co/Qwen/Qwen3-8B
    # increase by 0.75% the accuracy over no instruction
    # instruction = "Please reason step by step, and put your final answer within \\boxed{}."

    # for some reason leads to 5% accuracy drop compared to the official Qwen instruction.
    # instruction = f"Please reason step by step, and put your final answer within {Markers.ANSWER_START} $ LaTeX-here $ {Markers.ANSWER_END}."

    problem = sample["instruct_prompt"]
    content = f"{problem}"
    return content


test_code_template = """
# 2. The function to run tests and calculate the ratio
def run_tests_and_get_ratio():
    # Create a TestLoader instance
    loader = unittest.TestLoader()

    # Create a TestSuite
    suite = unittest.TestSuite()
    
    # Add tests from the test class to the suite
    suite.addTests(loader.loadTestsFromTestCase(TestCases))

    # Create a TestRunner
    # We use a TextTestRunner with verbosity=0 to suppress the default output,
    # and stream=sys.stderr to prevent cluttering stdout. You can adjust this.
    runner = unittest.TextTestRunner(verbosity=0, stream=sys.stderr)
    
    # Run the tests. The run() method returns a TestResult object.
    # print("Running tests...")
    result = runner.run(suite)
    # print("Tests finished.")
    
    # --- The Calculation Part ---
    
    # Total number of tests that were run
    total_run = result.testsRun
    
    # Number of failures and errors
    failures = len(result.failures)
    errors = len(result.errors)
    
    # Calculate the number of passed tests
    passed = total_run - (failures + errors)

    # Calculate the ratio
    if total_run > 0:
        pass_ratio = passed / total_run
    else:
        pass_ratio = 0.0

    return pass_ratio, result.wasSuccessful()


# 3. Main execution block, replacing the simple unittest.main()
if __name__ == '__main__':
    import unittest
    import sys

    ratio, was_successful = run_tests_and_get_ratio()
    sys.exit(int(ratio * 100))
"""

def create_test_code(sample, code: str) -> str:
    # code = sample['complete_prompt'] + sample['canonical_solution']

    # delimeter = "class TestCases(unittest.TestCase):"
    # test_code = sample['test'].split(delimeter, maxsplit=1)

    # test_code = f"{test_code[0]}\n{code}\n{delimeter}\n{test_code[1]}\n{test_code_template}"
    test_code = f"{sample['test']}\n{code}\n{test_code_template}"
    return test_code
