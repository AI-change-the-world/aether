import sys

sys.path.append("../src")
from aether.utils.object_match import validate

if __name__ == "__main__":
    schema = """
    {
        "temperature": {
            "type": "float",
            "default": 0.7
        },
        "history": {
            "type": "list",
            "default": "...",
            "items": {
                "type": "object",
                "fields": {
                    "role": {
                        "type": "string",
                        "default": "user"
                    },
                    "content": {
                        "type": "string",
                        "default": "..."
                    }
                }
            }
        }
    }
"""
    input = {
        "temperature": 0.5,
        "history": [{"role": "system", "content": "you are a helpful assistant"}],
    }

    assert validate(input, schema)
