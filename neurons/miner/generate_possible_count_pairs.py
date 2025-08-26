import math
from typing import Dict, Tuple

def generate_all_possible_count_pairs(C: int) -> list[Tuple[int, int]]:
    """
    Generate all possible (rule_count, nonrule_count) pairs where the sum is in range [int(C/3), int(3*C)].
    
    INPUT:
    C: target count, int
    Pr: Route percent, 0~1 float (not used in this function)
    
    OUTPUT:
    List of (rule_count, nonrule_count) tuples
    """
    
    min_sum = int(C / 5)
    max_sum = int(2 * C)
    
    pairs = []
    
    for total in range(min_sum, max_sum + 1):
        for rule_count in range(0, total + 1):
            nonrule_count = total - rule_count
            pairs.append((rule_count, nonrule_count))
    
    return pairs
    

def generate_possible_count_pairs(C: int, Pr: float, O: Dict[str, float], Ce: int) -> list[Tuple[int, int]]:
    """
    Calculate top 10 optimal rule count (Cr) and none rule count (Cn) based on constraints and optimization criteria.
    
    INPUT:
    C: target count, int
    Pr: Route percent, 0~1 float
    O: distribution dict, dictionary {'Light': 0.7, 'Medium': 0.2, 'Far': 0.1}
    Ce: effective rule count, int
    
    OUTPUT:
    List of (Cr, Cn) tuples, sorted by score descending, max 10 items
    Where:
    Cr: rule count, int
    Cn: none rule count, int
    
    Constraints:
    R1: int(0.5 * C) < Cr + Cn <= int(1.5 * C)
    
    Optimization criteria (when R1 satisfied):
    - Weight 0.4: min(O values, except 0.0) * Cn >= 1 (for orthograph)
    - Weight 0.3: Cr = int((Cr + Cn) * Pr) (for rule)
    - Weight 0.2: Cr > Ce (for rule)
    - Weight 0.1: Cr + Cn close to range <C-R, C+R>
      here R = max(1, C*(1-Pr)*0.2)
    """
    
    # Calculate bounds for R1
    min_total = int(0.5 * C) + 1  # > int(0.5 * C)
    max_total = int(3 * C)      # <= int(1.5 * C)
    
    # Get minimum non-zero value from O
    non_zero_values = [v for v in O.values() if v > 0.0]
    min_o_value = min(non_zero_values) if non_zero_values else 0.1
    
    # Calculate range R for the fourth criterion
    R = max(1, int(C * (1 - Pr) * 0.2))
    target_min = C - R
    target_max = C + R
    
    candidates = []
    
    # Try all possible combinations within the valid range
    for total in range(min_total, max_total + 1):
        for cr_candidate in range(0, total + 1):
            cn_candidate = total - cr_candidate
            
            # Calculate optimization score
            score = 0.0
            
            # Criterion 1 (weight 0.4): min(O values, except 0.0) * Cn >= 1 (for orthograph)
            if min_o_value * cn_candidate >= 1:
                score += 0.4
                
            # Criterion 2 (weight 0.3): Cr = int((Cr + Cn) * Pr) (for rule)
            expected_cr = int(total * Pr)
            if cr_candidate == expected_cr:
                score += 0.3
                
            # Criterion 3 (weight 0.2): Cr > Ce (for rule)
            if cr_candidate > Ce:
                score += 0.2
                
            # Criterion 4 (weight 0.1): Cr + Cn close to range <C-R, C+R>
            if target_min <= total <= target_max:
                score += 0.1
            
            # Store detailed scoring for filtering
            criteria_scores = (
                1 if min_o_value * cn_candidate >= 1 else 0,  # orthograph
                1 if cr_candidate == expected_cr else 0,      # rule match
                1 if cr_candidate > Ce else 0,                # rule count
                1 if target_min <= total <= target_max else 0 # count score (binary)
            )
            
            candidates.append((score, cr_candidate, cn_candidate, criteria_scores))
    
    # Group candidates by their first 3 criteria (orthograph, rule match, rule count)
    # For each group, keep the 3 with best count score
    groups = {}
    for score, cr, cn, (orth, rule_match, rule_count, count_score) in candidates:
        key = (orth, rule_match, rule_count)
        if key not in groups:
            groups[key] = []
        groups[key].append((score, cr, cn, (orth, rule_match, rule_count, count_score)))
    
    # For each group, keep only the top 3 by score
    filtered_candidates = []
    for key, group_candidates in groups.items():
        # Get all items where at least 2 out of the first 3 criteria are 1
        orth, rule_match, rule_count = key
        criteria_sum = orth + rule_match + rule_count
        
        if criteria_sum >= 2:
            # Sort by score descending and take all items
            group_candidates.sort(reverse=True)
            filtered_candidates.extend(group_candidates)
        else:
            # For other cases, sort by score descending and take only top 5
            group_candidates.sort(reverse=True)
            filtered_candidates.extend(group_candidates[:5])
    # Remove records where cr or cn is 0
    filtered_candidates = [c for c in filtered_candidates if c[1] > 0 and c[2] > 0]
    filtered_candidates.sort(reverse=True)
    return [(cr, cn) for _, cr, cn, _ in filtered_candidates[:27]]


def validate_solution(C: int, Pr: float, O: Dict[str, float], Ce: int, Cr: int, Cn: int) -> Dict[str, any]:
    """
    Validate if the solution satisfies all constraints and criteria.
    """
    results = {}
    
    # Check R1: int(0.5 * C) < Cr + Cn <= int(1.5 * C)
    min_total = int(0.5 * C)
    max_total = int(1.5 * C)
    results['R1'] = min_total < (Cr + Cn) <= max_total
    
    # Check optimization criteria
    non_zero_values = [v for v in O.values() if v > 0.0]
    min_o_value = min(non_zero_values) if non_zero_values else 0.1
    
    # Calculate range R for the fourth criterion
    R = max(1, int(C * (1 - Pr) * 0.2))
    target_min = C - R
    target_max = C + R
    
    # Criterion 1 (weight 0.4): min(O values, except 0.0) * Cn >= 1 (for orthograph)
    results['criterion_1_orthograph'] = min_o_value * Cn >= 1
    
    # Criterion 2 (weight 0.3): Cr = int((Cr + Cn) * Pr) (for rule)
    expected_cr = int((Cr + Cn) * Pr)
    results['criterion_2_rule_match'] = Cr == expected_cr
    
    # Criterion 3 (weight 0.2): Cr > Ce (for rule)
    results['criterion_3_rule_count'] = Cr > Ce
    
    # Criterion 4 (weight 0.1): Cr + Cn close to range <C-R, C+R>
    results['criterion_4_in_range'] = target_min <= (Cr + Cn) <= target_max
    results['criterion_4_range'] = f"[{target_min}, {target_max}]"
    
    # Calculate total score
    score = 0.0
    if results['criterion_1_orthograph']:
        score += 0.4
    if results['criterion_2_rule_match']:
        score += 0.3
    if results['criterion_3_rule_count']:
        score += 0.2
    if results['criterion_4_in_range']:
        score += 0.1
    
    results['total_score'] = score
    
    return results


# Example usage and test
if __name__ == "__main__":
    # Test case 1
    # Test case 2
    C = 15
    Pr = 0.51
    O = {'Light': 0.5, 'Medium': 0.5}
    Ce = 1
    
    
    solutions = calculate_rule_counts(C, Pr, O, Ce)
    
    print(f"Test Case 1:")
    print(f"Input: C={C}, Pr={Pr}, O={O}, Ce={Ce}")
    print("\nAll Solutions (Cr, Cn):")
    for i, (cr, cn) in enumerate(solutions, 1):
        validation = validate_solution(C, Pr, O, Ce, cr, cn)
        print(f"\nSolution {i}:")
        print(f"Cr={cr}, Cn={cn}")
        print(f"Validation: {validation}")
    print()