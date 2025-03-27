import re

def math_agent_compute(expression: str) -> float:
    try:
        expression = expression.replace("MATH:", "").strip()
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*\*\s*(\d+(?:\.\d+)?)", expression)
        if match:
            num1 = float(match.group(1))
            num2 = float(match.group(2))
            return num1 * num2
        else:
            return None
    except Exception as e:
        print(f"Math Agent error: {e}")
        return None

def process_math_expressions(answer: str) -> str:
    def repl(match):
        expr = match.group(1)
        result = math_agent_compute("MATH: " + expr)
        if result is not None:
            return str(int(result)) if result.is_integer() else f"{result:.1f}"
        else:
            return match.group(0)
    return re.sub(r"MATH:\s*([0-9\.]+\s*\*\s*[0-9\.]+)", repl, answer)
