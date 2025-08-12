from src.configs.markers import Markers

def format_prompt(sample: dict) -> str:

    instruction = "Write a function to solve the following problem:\n"
    instruction = ""

    problem = sample["instruct_prompt"]
    content = f"{instruction}{problem}"
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
