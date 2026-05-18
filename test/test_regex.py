import re

def evaluate_delimited_formulas(text: str, para_dict: dict) -> str:
    pattern = r'<<(.*?)>>'
    
    def replace_match(match):
        formula = match.group(1).strip()
        try:
            expr = formula
            # Replace variable names with their values - order by length to prevent partial matches
            for key in sorted(para_dict.keys(), key=len, reverse=True):
                val = para_dict[key]
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    pass
                if isinstance(val, (int, float)):
                    # Use regex with word boundaries
                    expr = re.sub(rf'\b{re.escape(key)}\b', str(val), expr)
            
            print("Expr after replacement:", expr)
            result = eval(expr, {"__builtins__": {}})
            
            if isinstance(result, float):
                if abs(result) < 0.001:
                    return f"{result:.2e}"
                elif result > 100:
                    return f"{result:.1f}"
                else:
                    return f"{result:.4f}"
            return str(result)
        except Exception as e:
            print(f"[FORMULA ERROR] '{formula}' - {e}")
            return f"<<{formula}>>"
    
    return re.sub(pattern, replace_match, text)

print(evaluate_delimited_formulas("Value is << accuracy / duration >>", {"accuracy": "0.95", "duration": 0.05}))
