import os
import json
import asyncio

from neurons.miner.generate_name_variations import generate_name_variations
from neurons.miner.parse_query import parse_query


async def generate_variations_from_query_params(names: list, query_params: dict, run_dir: str = None) -> list:
    total_count=int(query_params.get("variation_count") or 0)
    rule_percentage=float(query_params.get("rule_percentage") or 0.0)
    selected_rules=list(query_params.get("selected_rules") or [])
    phonetic_similarity=query_params.get("phonetic_similarity")
    orthographic_similarity=query_params.get("orthographic_similarity")
    # Clamp and compute counts
    if total_count < 0:
        total_count = 0
    if rule_percentage < 0.0:
        rule_percentage = 0.0
    if rule_percentage > 1.0:
        rule_percentage = 1.0

    tasks = []
    for i, name in enumerate(names):
        task = generate_name_variations(
            original_name=name,
            total_count=total_count,
            rule_percentage=rule_percentage,
            selected_rules=selected_rules,
            phonetic_similarity=phonetic_similarity,
            orthographic_similarity=orthographic_similarity,
            logfile=os.path.join(run_dir, f"{i}_{name}.log") if run_dir is not None else None
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks)
    # Collect results
    name_results = {}
    metrics = {}
    metric_file = None
    for name, name_variations, metric in results:
        name_results[name] = name_variations
        metrics[name] = metric
    if metric_file:
        with open(metric_file, "w") as f:
            json.dump(metrics, f, indent=4)
    return name_results, metrics

async def generate_variations_from_template(names: list, query_template: str, run_dir: str = None) -> list:
    # parse query and save to query.json
    query_file = None
    if run_dir is not None:
        query_file = os.path.join(run_dir, "query.json")
        metric_file = os.path.join(run_dir, "metric.json")
    else:
        query_file = "query.json"
        metric_file = "metric.json"
        
    query_params = await parse_query(query_template, max_retries=1)
    
    name_results, metrics = await generate_variations_from_query_params(names, query_params)
    if query_file:
        output = {}
        output["names"] = names
        output["query_template"] = query_template
        output["parsed_result"] = query_params
        with open(query_file, "w") as f:
            json.dump(output, f, indent=4)
    return name_results, metrics

if __name__ == "__main__":
    generate_variations_from_template()