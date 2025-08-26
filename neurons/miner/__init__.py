import sys
import json

from neurons.miner.generate_variations import generate_variations_from_template
from neurons.miner.parse_query import parse_query


if __name__ == "__main__":
    query_file = "default_query.json" if len(sys.argv) == 1 else sys.argv[2]
    names = []
    query_template = ""
    with open(query_file, "r") as f:
        query_data = json.load(f)
        names = query_data["names"]
        query_template = query_data["query_template"]
    variations = generate_variations_from_template(names, query_template)
    print(variations)

__all__ = [
    "generate_variations_from_template",
    "parse_query",
]
