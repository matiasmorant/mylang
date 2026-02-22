import pytest
import json
from langLark2 import parser, MyTransformer, ProgramState, Number, Symbol

def load_tests():
    with open("test_cases.json", "r") as f:
        return json.load(f)

def unwrap(val):
    """Helper to convert internal objects to python primitives for testing."""
    if isinstance(val, (Number, float)): return float(val)
    if isinstance(val, (Symbol, str)): return str(val)
    return val

@pytest.mark.parametrize("test_case", load_tests())
def test_language_execution(test_case):
    code = test_case["code"]
    
    # 1. Parse and Transform
    tree = parser.parse(code)
    transformed = MyTransformer().transform(tree)
    
    # 2. Initialize State
    ps = ProgramState(queue=transformed.children)
    
    # 3. Execute
    try:
        ps.finish()
    except Exception as e:
        pytest.fail(f"Execution failed for '{code}': {e}")

    # 4. Validate (if expected output is provided in JSON)
    if "expected" in test_case:
        result_stack = [unwrap(x) for x in ps.stack]
        assert result_stack == [test_case["expected"]]