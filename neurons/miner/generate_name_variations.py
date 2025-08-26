from typing import List, Tuple, Iterable, Dict, Set, Any
import bittensor as bt
try:
    from typing import SimpleNamespace
except Exception as e:
    from types import SimpleNamespace
import asyncio
from tabulate import tabulate
import json
import random
import numpy as np

from neurons.miner.custom_logger import CustomLogger
from neurons.miner.generate_possible_count_pairs import generate_possible_count_pairs, generate_all_possible_count_pairs
from neurons.miner.rule_based_transformations import RULE_BASED_TRANSFORMATIONS
from neurons.miner.rule_based_transformations import RULE_BASED_TRANSFORMATIONS_COMBINED
from MIID.validator.reward import calculate_variation_quality, calculate_orthographic_similarity
from MIID.validator.rule_evaluator import evaluate_rule_compliance
from MIID.validator.rule_evaluator import RULE_EVALUATORS


def calculate_orthographic_level(name: str, variation: str) -> int:
    """Calculate the orthographic similarity level of a variation"""
    orthographic_boundaries = {
                0: (0.70, 1.00),    # High orthographic similarity
                1: (0.50, 0.69),   # Medium orthographic similarity  
                2: (0.20, 0.49),       # Low orthographic similarity
                3: (0.0, 0.19)       # Too low orthographic similarity
            }
    sim = calculate_orthographic_similarity(name, variation)
    for level, (min_val, max_val) in orthographic_boundaries.items(): 
        if min_val <= sim <= max_val:
            return level
    return 3 # Too low orthographic similarity

def get_effective_rules(name: str, selected_rules: List[str]) -> List[str]:
    compliant_variations, _ = evaluate_rule_compliance(name, [""], selected_rules)
    return list(compliant_variations.keys())

def get_name_parts(name: str) -> Tuple[str, str]:
    name_parts = name.split()
    if len(name_parts) < 2:
        first_name = name
        last_name = None
    else:
        i = name.find(" ")
        first_name = name[:i]
        last_name = name[i+1:]

    return first_name, last_name

def generate_one_rule_based_variation(rules, name, max_attempts = 1000):
    attempts = 0
    while attempts < max_attempts:
        try:
            attempts +=1
            key = "_".join(sorted(rules))
            variation = RULE_BASED_TRANSFORMATIONS_COMBINED[key](name)
            if not variation.strip() or variation == name:
                continue
            is_valid = True
            for r in rules:
                if not RULE_EVALUATORS[r](name, variation):
                    is_valid = False
                    break
            if is_valid:
                return variation
        except Exception:
            pass
    return None

def get_effective_rules_variations(name: str, effective_rules: List[str]) -> List[str]:
    effective_rules_variations = set()
    checked_rules = set()
    from itertools import combinations
    for r in range(len(effective_rules), 0, -1):
        for comb in combinations(effective_rules, r):
            key = "_".join(sorted(comb))
            if not key in RULE_BASED_TRANSFORMATIONS_COMBINED or any([cr in checked_rules for cr in comb]):
                continue
            variation = generate_one_rule_based_variation(comb, name)
            if variation:
                effective_rules_variations.add(variation)
                checked_rules.update(comb)
    return effective_rules_variations

def variation_matches_any_rule(name: str, variation: str, rules: List[str]) -> bool:
    """Check if a variation matches any of the specified rule-based transformations"""
    if not rules:
        return False
    
    for rule in rules:
        if rule in RULE_EVALUATORS:
            try:
                if RULE_EVALUATORS[rule](name, variation):
                    return True
            except Exception:
                # If rule evaluation fails, assume it doesn't match
                continue
    return False

async def gen_pool(name, effective_rules, logger):
    if name is None:
        return
    max_l_level = len(name)
    # Request variant pool from nvgen service
    vpools_from_generator = None
    try:
        import httpx
        
        client = None
        try:
            retry_count = 0
            max_retries = 5
            response = None
            while retry_count < max_retries:
                retry_count += 1
                try:
                    client = httpx.AsyncClient(timeout=300.0)
                    response = await client.get(f"http://localhost:8001/pool?original_name={name}&timeout={100}")
                    if response.status_code == 200:
                        break
                    else:
                        logger.debug(f"Failed to get variants from nvgen service for {name}: {response.status_code}")
                except Exception as e:
                    logger.debug(f"Failed to get variants from nvgen service for {name}: {e}")
                logger.info(f"Retrying to get variants from nvgen service for {name} (attempt {retry_count}/{max_retries})")                
                await asyncio.sleep(3)
                continue
            if response and response.status_code == 200:
                nvgen_data = response.json()
                # logger.debug(f"nvgen response for {name}: {nvgen_data}")
                pools = nvgen_data.get("pools", [])
                # logger.debug(f"pools structure for {name}: {pools}")
                # Update max_l_level based on actual pool data
                if pools:
                    # Extend vpools if needed
                    vpools_from_generator = [[[set() for _ in range(8)] for _ in range(4)] for _ in range(max_l_level)]
                    
                    # Populate vpools from nvgen service response
                    for l_level, o_levels in enumerate(pools):
                        if l_level < max_l_level:
                            all_zero = True
                            for o_level, p_classes in enumerate(o_levels):
                                if o_level < 4:
                                    for p_level, variants in enumerate(p_classes):
                                        if p_level < 8:
                                            for variant in variants:
                                                # check rule-based transformations
                                                if effective_rules and variation_matches_any_rule(name, variant, effective_rules):
                                                    continue
                                                vpools_from_generator[l_level][o_level][p_level].add(variant)
                                                all_zero = False
                            if all_zero:
                                break
        finally:
            if client:
                try:
                    await client.aclose()
                except RuntimeError:
                    # Event loop is closed, ignore cleanup error
                    pass
    except Exception as e:
        logger.warning(f"Failed to get variants from nvgen service for {name}: {e}")
        # Fall back to local generation
    
    # Ensure vpools_from_generator is properly initialized even if service call failed
    if not vpools_from_generator:
        # Initialize with default structure
        vpools_from_generator = [[[set() for _ in range(8)] for _ in range(4)] for _ in range(max_l_level)]
        
    # initialize vpools as the 1(l) * 4(o) * 8(p) matrix of variable pools
    # Print vpools_from_generator in tabular format
    try:
        
        # Create table data for vpools_from_generator
        for l_level in range(len(vpools_from_generator)):
            logger.info(f"Variant pools for {name} at L{l_level}:")
            
            # Prepare headers (G0, G1, G2, ..., G7)
            headers = [""] + [f"G{p}" for p in range(8)]
            
            # Prepare table data
            table_data = []
            for o_level in range(4):
                row = [f"O{o_level}"]
                for p_level in range(8):
                    if l_level < len(vpools_from_generator) and o_level < len(vpools_from_generator[l_level]) and p_level < len(vpools_from_generator[l_level][o_level]):
                        count = len(vpools_from_generator[l_level][o_level][p_level])
                    else:
                        count = 0
                    row.append(str(count))
                table_data.append(row)
            
            # Print the table
            table_str = tabulate(table_data, headers=headers, tablefmt="grid")
            logger.info(f"\n{table_str}")
            
    except ImportError:
        logger.warning("tabulate library not available, skipping formatted table output")
    except Exception as e:
        logger.warning(f"Error creating table for vpools: {e}")
    # fulfil pool by using uppercased variations
    logger.debug(f"max_l_level: {len(vpools_from_generator)}")

    vpools_per_name_new = [[[set() for _ in range(8)] for _ in range(4)] for _ in range(len(vpools_from_generator))]
    for l_level in range(len(vpools_from_generator)):
        for o_level in range(4):
            for p_level in range(8):
                for variation in vpools_from_generator[l_level][o_level][p_level]:
                    # no check duplication as variation never duplicated in the pool from generator
                    # no check used variants as the pool never sends the used variants for the given problem.
                    # add original variation to the pool
                    for o in range(o_level, 4):
                        if len(vpools_per_name_new[l_level][o][p_level]) < 20:
                            break
                    if o == 3:
                        continue
                    vpools_per_name_new[l_level][o_level][p_level].add(variation)
                    # add uppercased variations to the pool
                    def toggle_case(s:str, idxs:Iterable[int])->str:
                        arr=list(s)
                        for i in idxs:
                            ch=arr[i]
                            arr[i]= ch.upper() if ch.islower() else ch.lower()
                        return ''.join(arr)
                    maxm = 1 << len(variation)
                    for mask in range(1, maxm):
                        idxs=[i for i in range(len(variation)) if (mask >> i) & 1]
                        uppercased = toggle_case(variation, idxs)
                        if effective_rules and variation_matches_any_rule(name, uppercased, effective_rules):
                            continue
                        uppercased_o_level = calculate_orthographic_level(name, uppercased)
                        if len(vpools_per_name_new[l_level][uppercased_o_level][p_level]) < 20:
                            vpools_per_name_new[l_level][uppercased_o_level][p_level].add(uppercased)
    
    # Debug: print vpools table using tabulate if available
    try:
        logger.debug("Vpools table after case transformations:")
        
        for l_level in range(len(vpools_per_name_new)):
            logger.debug(f"L{l_level} level:")
            
            # Create headers
            headers = ["O\\P"]
            for p_level in range(8):
                headers.append(f"P{p_level}")
            
            # Create table data
            table_data = []
            for o_level in range(4):
                row = [f"O{o_level}"]
                for p_level in range(8):
                    count = len(vpools_per_name_new[l_level][o_level][p_level])
                    row.append(str(count))
                table_data.append(row)
            
            # Print the table
            table_str = tabulate(table_data, headers=headers, tablefmt="grid")
            logger.debug(f"\n{table_str}")
            
    except ImportError:
        logger.warning("tabulate library not available, skipping vpools table output")
    except Exception as e:
        logger.warning(f"Error creating vpools table: {e}")

    return vpools_per_name_new

def generate_rule_variations(
    original_name: str,
    effective_rules: Set[str],
    effective_rules_variations: Set[str],
    rule_count: int,
    logger: CustomLogger) -> Set[str]:
    attempts = 0
    max_attempts = 1000
    rule_variations = effective_rules_variations.copy()
    while len(rule_variations) < rule_count:
        if not effective_rules:
            break
        rule = random.choice(effective_rules)
        attempts += 1
        if attempts >= max_attempts:
            break
        try:
            variation = RULE_BASED_TRANSFORMATIONS[rule](original_name)
            is_valid = RULE_EVALUATORS[rule](original_name, variation)
            if (is_valid and
                variation != original_name and 
                len(variation.strip()) > 0 and
                variation not in rule_variations):
                rule_variations.add(variation)
        except Exception:
            continue
    
    return rule_variations

def counts_in_matrix48(
    name: str,
    pool: List[str],
    O: List[int],
    P: List[int],
    nonrule_count: int,
    logger: CustomLogger
) -> Dict[str, int]:
    if name is None:
        return None
    counts_in_matrix48 = {}
    # initilize max_l_level = len(name) and update from variant pool
    max_l_level = len(pool)
    # select variations from each pool according to
    # orthographic and phonetic similarity using max flow algorithm
    O = O.copy()
    P = P.copy()

    x = []
    pidx_levels = [
        [0],
        [1, 2, 3],
        [4, 5, 6],
        [7]
    ]
    # start from the lowest level of length
    max_flow = []
    for l_level in range(max_l_level):
        # calc matrix flow in level l_level
        # construct MaxU matrix 4(p) * 4(o)
        MaxU = [[None]*4 for _ in range(4)]
        for o_level in range(4):
            for p_level in range(4):
                MaxU[o_level][p_level] = sum(
                    len(pool[l_level][o_level][pidx])
                    for pidx in pidx_levels[p_level]
                    )

        # solve max flow problem
        from neurons.miner.matrix_flow import solve_maxflow, max_transport44
        # mf, flow = solve_maxflow(MaxU, P, O)
        mf, flow = max_transport44(MaxU, P, O)
        

        # let's update the remains of O and P for the next length level.
        for o_level in range(4):
            O[o_level] = O[o_level] - sum(mf[o_level])
        for p_level in range(4):
            P[p_level] = P[p_level] - sum(mf[o_level][p_level] for o_level in range(4))
        # Check if all O values are zero
        # append last column to x: 4(p) * 4(o) -> 4(p) * 4(o)
        for o_level in range(4):
            mf[o_level].append(0)
        max_flow.append(mf)

    # if othographic(sum of row) not solved successfully,
    # complete the orthographic count of matrix_flow with the variants in the order of m, f, l, t-f
    for o_level in range(4):
        if O[o_level] > 0:
            logger.warning(f"Complete {O[o_level]} for othographic level {o_level}")
            for l_level in range(max_l_level):
                for p_level in (3, 1, 2, 0):
                    remains = sum(len(pool[l_level][o_level][pidx]) for pidx in pidx_levels[p_level]) - max_flow[l_level][o_level][p_level]
                    if remains >= O[o_level]:
                        max_flow[l_level][o_level][p_level] += O[o_level]
                        O[o_level] = 0
                        break
                    else:
                        max_flow[l_level][o_level][p_level] += remains
                        O[o_level] -= remains

    # if still not completed nonrule count,
    # complete the remains with ortho_level = 3 row to ensure nonrule count
    remains = nonrule_count - sum(
        max_flow[l_level][o_level][p_level]
        for p_level in range(4)
        for o_level in range(4)
        for l_level in range(max_l_level)
    )
    if remains > 0:
        for l_level in range(max_l_level):
            # for p_level in [1, 2, 0, 3]:
            for p_level in [3, 1, 2, 0]:
                count = sum(
                    len(pool[l_level][3][pidx])
                    for pidx in pidx_levels[p_level]
                ) - max_flow[l_level][3][p_level]
                if count >= remains:
                    max_flow[l_level][3][p_level] += remains
                    remains = 0
                    break # need for double break
                else:
                    max_flow[l_level][3][p_level] += count
                    remains -= count
    logger.warning(f"Remains: {remains}")
    max_flow2 = [[[0 for _ in range(8)] for _ in range(4)] for _ in range(max_l_level)]
    for l_level in range(len(max_flow)):
        for p_level in [1, 2, 0, 3]:
            all_zero = True
            O = [0] * 4
            for o_level in range(4):
                O[o_level] = max_flow[l_level][o_level][p_level]
                if O[o_level] != 0:
                    all_zero = False
            if all_zero:
                continue
            MaxU = [[0 for _ in range(len(pidx_levels[p_level]))] for _ in range(4)]
            for o_level in range(4):
                for idx, pidx in enumerate(pidx_levels[p_level]):
                    MaxU[o_level][idx] = len(pool[l_level][o_level][pidx])
            logger.debug(f"p_level: {p_level}, O: {O}")
            for row in MaxU:
                logger.debug(f"{row}")
            from neurons.miner.matrix_flow import solve_integer_diverse
            xx, _ = solve_integer_diverse(MaxU, O)
            for o_level in range(4):
                logger.debug(f"{xx[o_level]}")
            for o_level in range(4):
                for idx, pidx in enumerate(pidx_levels[p_level]):
                    max_flow2[l_level][o_level][pidx] = xx[o_level][idx]
    logger.debug(f"final:")
    for l_level in range(max_l_level):
        logger.debug(f"name: {name}, l_level: {l_level}")
        logger.debug(f"-" * 50)
        for o_level in range(4):
            logger.debug(f"{max_flow2[l_level][o_level]}")
    return max_flow2

def generate_nonrule_variations(
    original_name: str,
    first_name: str,
    last_name: str,
    name_pools: Dict[str, List[str]],
    nonrule_count: int,
    phonetic_similarity: float,
    orthographic_similarity: float,
    effective_rules: Set[str],
    rule_variations: Set[str],
    logger: CustomLogger) -> Set[str]:
    if nonrule_count <= 0:
        return []

    nonrule_variations = set()
    # Use provided similarity distributions or defaults
    phonetic_dist = phonetic_similarity if phonetic_similarity else {"Light": 0.33, "Medium": 0.34, "Far": 0.33}
    orthographic_dist = orthographic_similarity if orthographic_similarity else {"Light": 0.33, "Medium": 0.34, "Far": 0.33}

    # calculate the number of variations for each orthographic similarity level
    # to complete the nonrule_count, the rest will be assigned to the too-far level
    O = [0] * 4
    for key in orthographic_dist.keys():
        if key == "Light":
            O[0] = int(orthographic_dist["Light"] * nonrule_count)
        elif key == "Medium":
            O[1] = int(orthographic_dist["Medium"] * nonrule_count)
        elif key == "Far":
            O[2] = int(orthographic_dist["Far"] * nonrule_count)
    O[3] = nonrule_count - sum(O)
    # calculate the number of variations for each phonetic similarity level
    P = [0] * 4
    # if phonetic_dist["Light"] < 0.4 and phonetic_dist["Medium"] < 0.4 and phonetic_dist["Far"] < 0.4:
    #     phonetic_dist["Light"] = 0.3
    #     phonetic_dist["Medium"] = 0.3
    #     phonetic_dist["Far"] = 0.3
    for key in phonetic_dist.keys():
        if key == "Light":
            P[0] = int(phonetic_dist["Light"] * nonrule_count)
        elif key == "Medium":
            P[1] = int(phonetic_dist["Medium"] * nonrule_count)
        elif key == "Far":
            P[2] = int(phonetic_dist["Far"] * nonrule_count)
    # to complete the nonrule_count, the rest will be assigned to the medium level
    P[3] += nonrule_count - sum(P)
    logger.debug(f"O: {O}")
    logger.debug(f"P: {P}")

    # select variations from each pool according to
    # orthographic and phonetic similarity using max flow algorithm
    max_flow2 = {}
    for name in [first_name, last_name]:
        if name:
            max_flow2[name] = counts_in_matrix48(
                name,
                name_pools[name],
                O,
                P,
                nonrule_count,
                logger
            )
        # for l_level in range(len(max_flow2)):
        #     for o_level in range(4):
        #         for p_level in range(8):
        #             if max_flow2[l_level][o_level][p_level] > 0:
        #                 nonrule_variations.add(name_pools[name][l_level][o_level][p_level])
    if not last_name:
        for l_level in range(len(max_flow2[first_name])):
             for o_level in range(4):
                for pidx in range(8):
                    count = max_flow2[first_name][l_level][o_level][pidx]
                    max_count = len(name_pools[first_name][l_level][o_level][pidx])
                    select_variants = random.sample(name_pools[first_name][l_level][o_level][pidx], min(count, max_count))
                    nonrule_variations.update(select_variants)
    else:
        # Counters
        first_used_count = [[[0 for _ in range(8)] for _ in range(4)] for _ in range(len(max_flow2[first_name]))]
        last_used_count  = [[[0 for _ in range(8)] for _ in range(4)] for _ in range(len(max_flow2[last_name]))]

        first_used, last_used = set(), set()

        # Flatten cells into (l_level, o_level, pidx, variant) lists
        def collect_candidates(name):
            candidates = []
            for l_level in range(len(max_flow2[name])):
                for o_level in range(4):
                    for pidx in range(8):
                        target = max_flow2[name][l_level][o_level][pidx]
                        if target <= 0:
                            continue
                        pool = list(name_pools[name][l_level][o_level][pidx])
                        random.shuffle(pool)
                        for v in pool:
                            candidates.append((l_level, o_level, pidx, v, target))
            random.shuffle(candidates)
            return candidates

        first_candidates = collect_candidates(first_name)
        last_candidates  = collect_candidates(last_name)

        # Try to pair
        for fl, fo, fp, fvar, ftarget in first_candidates:
            if first_used_count[fl][fo][fp] >= ftarget or fvar in first_used:
                continue

            for ll, lo, lp, lvar, ltarget in last_candidates:
                if last_used_count[ll][lo][lp] >= ltarget or lvar in last_used:
                    continue

                variation = fvar + " " + lvar
                if variation in rule_variations:
                    continue
                if effective_rules and variation_matches_any_rule(original_name, variation, effective_rules):
                    continue

                # Accept this pair
                nonrule_variations.add(variation)
                first_used.add(fvar)
                last_used.add(lvar)
                first_used_count[fl][fo][fp] += 1
                last_used_count[ll][lo][lp] += 1
                break  # move to next first_variant

    return nonrule_variations


def get_name_variation_rewards_exclude_phonetic(
    seed_names: List[str],
    responses: List[Dict[str, List[str]]],
    uids: List[int],
    variation_count: int = 10,
    phonetic_similarity: Dict[str, float] = None,
    orthographic_similarity: Dict[str, float] = None,
    rule_based: Dict[str, Any] = None
) -> Tuple[np.ndarray, List[Dict]]:

    from MIID.validator.reward import get_name_variation_rewards
    from MIID.validator.reward import get_name_part_weights
    name_part_weights = get_name_part_weights(seed_names[0])

    rewards, detailed_metrics =  get_name_variation_rewards(
        None, seed_names, responses, uids, variation_count, phonetic_similarity, orthographic_similarity, rule_based)
    if not detailed_metrics:
        return 0, None
    if not detailed_metrics[0]['name_metrics'].values():
        return 0, None
    name_metrics = list(detailed_metrics[0]['name_metrics'].values())[0]
    
    try:
        if 'last_name' in name_metrics:
            rewards -= name_metrics['first_name']['metrics']['similarity']['phonetic'] * 0.24 * name_part_weights['first_name_weight'] + name_metrics['last_name']['metrics']['similarity']['phonetic'] * 0.24 * name_part_weights['last_name_weight']
        else:
            rewards -= name_metrics['first_name']['metrics']['similarity']['phonetic'] * 0.24 * name_part_weights['first_name_weight']
    except Exception as e:
        # print(json.dumps(name_metrics, indent=4))
        return 0, None
    return rewards, detailed_metrics

def try_once(
    original_name: str,
    first_name: str,
    last_name: str,
    name_pools: Dict[str, List[str]],
    total_count: int,
    rule_count: int,
    nonrule_count: int,
    rule_percentage: float,
    selected_rules: List[str],
    effective_rules: Set[str],
    effective_rules_variations: Set[str],
    phonetic_similarity: float,
    orthographic_similarity: float,
    logger: CustomLogger) -> Tuple[Set[str], Set[str]]:

    # 1) Rule-based portion
    rule_variations = generate_rule_variations(
        original_name,
        effective_rules,
        effective_rules_variations,
        rule_count,
        logger
    )
    # Adjust rule count based on actual generated variations
    if len(rule_variations) < rule_count:
        rule_count = len(rule_variations)
        nonrule_count = total_count - rule_count
        logger.debug(f"Adjusted rule_count: {rule_count}, nonrule_count: {nonrule_count}")
    rule_variations = set(list(rule_variations)[:rule_count])

    # 2) Generate non-rule variations
    nonrule_variations =  generate_nonrule_variations(
            original_name,
            first_name,
            last_name,
            name_pools,
            nonrule_count,
            phonetic_similarity,
            orthographic_similarity,
            effective_rules,
            rule_variations,
            logger
        )
    # 3) Select the best variants
    all_variants = list(rule_variations) + list(nonrule_variations)
    responses = {}
    responses = [SimpleNamespace(
        variations={original_name: all_variants}
    )]
    # Calculate rule-based metadata
    rule_based = {"selected_rules": selected_rules, "rule_percentage": rule_percentage * 100}
    debug_level = bt.logging.get_level()
    bt.logging.setLevel('CRITICAL')
    scores, metric = get_name_variation_rewards_exclude_phonetic(
        seed_names=[original_name], 
        responses=responses,
        uids=[0],
        variation_count=total_count,
        phonetic_similarity=phonetic_similarity,
        orthographic_similarity=orthographic_similarity,
        rule_based=rule_based,
    )
    bt.logging.setLevel(debug_level)
    return (scores[0], metric[0]) if scores else (0.0, None), all_variants, rule_variations, nonrule_variations


async def generate_name_variations(
    original_name: str,
    total_count: int,
    rule_percentage: float,
    selected_rules: List[str],
    phonetic_similarity: float,
    orthographic_similarity: float,
    logfile: str = None
) -> List[str]:
    logger = CustomLogger(name=original_name, output_file=logfile, use_stdout=False)
    logger.info(f"Generating variations for {original_name}")
    logger.info(f"Total count: {total_count}")
    logger.info(f"Rule percentage: {rule_percentage}")
    logger.info(f"Selected rules: {selected_rules}")
    logger.info(f"Phonetic similarity: {phonetic_similarity}")
    logger.info(f"Orthographic similarity: {orthographic_similarity}")
    logger.info(f"-" * 100)

    effective_rules = get_effective_rules(original_name, selected_rules)
    logger.info(f"Effective rules: {effective_rules}")
    logger.info(f"-" * 100)

    # generate rule-based variations at most one per each effective rule
    effective_rules_variations = get_effective_rules_variations(original_name, effective_rules)
    logger.info(f"Effective rules variations: {effective_rules_variations}")
    logger.info(f"-" * 100)

    # get name parts
    (first_name, last_name) = get_name_parts(original_name)
    logger.info(f"First name: {first_name}")
    logger.info(f"Last name: {last_name}")
    logger.info(f"-" * 100)

    # get pool of nonrule variations for first and last name
    try:
        tasks = [gen_pool(name, effective_rules, logger) for name in [first_name, last_name]]
        name_pools = await asyncio.gather(*tasks)
        name_pools = {first_name: name_pools[0], last_name: name_pools[1]}
    except Exception as e:
        logger.warning(f"Failed to generate variant pools for {original_name}: {e}")
        name_pools = None
        return []

    # # Get all possible rule count pairs
    # rule_count_pairs = generate_possible_count_pairs(
    #     total_count,
    #     rule_percentage,
    #     orthographic_similarity,
    #     len(effective_rules)
    # )
    rule_count_pairs = generate_all_possible_count_pairs(
        total_count
    )
    logger.info(f"Rule count pairs: {rule_count_pairs}")
    
    best_score = 0.0
    best_variations = []
    best_rule_variations = []
    best_nonrule_variations = []
    best_metric = {}

    for i, (rule_count, nonrule_count) in enumerate(rule_count_pairs):
        logger.info(
            f"-" * 30 +
            f"Trying rule_count_pair {i+1} / {len(rule_count_pairs)}: rule_count/nonrule_count: {rule_count}/{nonrule_count}" +
            "-" * 30
        )
        (score, metric), all_variants, rule_variations, nonrule_variations = try_once(
            original_name,
            first_name,
            last_name,
            name_pools,
            total_count,
            rule_count,
            nonrule_count,
            rule_percentage,
            selected_rules,
            effective_rules,
            effective_rules_variations,
            phonetic_similarity,
            orthographic_similarity,
            logger
        )
        if score > best_score:
            if abs(score - best_score) <= 0.00001:
                if len(best_variations) > len(all_variants):
                    best_variations = all_variants
                    best_rule_variations = rule_variations
                    best_nonrule_variations = nonrule_variations
                    best_metric = metric
                else:
                    continue
            else:
                best_score = score
                best_variations = all_variants
                best_rule_variations = rule_variations
                best_nonrule_variations = nonrule_variations
                best_metric = metric
    logger.debug(f"-" * 100)
    logger.debug(f"Best score achieved: {best_score}")
    logger.debug(f"Best selected: {rule_count}/{nonrule_count}")
    logger.debug(f"Rule variations: {best_rule_variations}")
    logger.debug(f"Nonrule variations: {best_nonrule_variations}")
    logger.debug(f"-" * 100)

    logger.flush()
    
    return original_name, list(best_variations), best_metric