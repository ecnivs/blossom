def math_handler(a: float, b: float, op: str):
    if op == "add":
        result = a + b
    elif op == "sub":
        result = a - b
    elif op == "mul":
        result = a * b
    elif op == "div":
        result = a / b if b != 0 else None
    else:
        raise ValueError("Invalid operation.")
    return {"operation": op, "a": a, "b": b, "result": result}


def register():
    return {
        "name": "math_tool",
        "description": "Performs simple arithmetic operations.",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
                "op": {"type": "string", "enum": ["add", "sub", "mul", "div"]},
            },
            "required": ["a", "b", "op"],
        },
        "handler": math_handler,
    }
