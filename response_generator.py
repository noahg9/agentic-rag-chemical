import re

# Definitions for specific industry terms
definitions = {
    "msds": "Material Safety Data Sheet",
    "fryma": "A system used for process cooling and reagent mixing",
    "maa/rs": "Ratio related to acid index and residue levels",
}

def replace_definitions(text: str) -> str:
    """
    Replaces industry terms with their definitions.
    """
    for term, definition in definitions.items():
        text = re.sub(rf"\b{term}\b", definition, text, flags=re.IGNORECASE)
    return text

def process_math_expressions(answer: str) -> str:
    """
    Replaces math expressions with computed results.
    """
    def repl(match):
        expr = match.group(1)
        result = eval(expr)
        return str(int(result)) if result.is_integer() else f"{result:.1f}"

    return re.sub(r"MATH:\s*([0-9\.]+\s*\*\s*[0-9\.]+)", repl, answer)
