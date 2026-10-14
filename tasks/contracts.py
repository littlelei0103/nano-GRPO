"""Executable input contracts for every built-in BugHunter problem.

These checks validate the input domain independently of the reference output.
The registry is trusted code; model completions cannot supply validators.
"""
import ast
import inspect
from collections import Counter
from functools import lru_cache


def integer(x):
    return type(x) is int


def ints(xs, minimum=None, nonempty=False):
    return (type(xs) is list and (not nonempty or bool(xs))
            and all(integer(x) and (minimum is None or x >= minimum) for x in xs))


def ordered(xs, unique=False):
    return ints(xs) and all(a < b if unique else a <= b for a, b in zip(xs, xs[1:]))


def binary(xs):
    return ints(xs) and all(x in (0, 1) for x in xs)


def strings(xs):
    return type(xs) is list and all(type(x) is str for x in xs)


def matrix(xs, square=False):
    return (type(xs) is list and all(ints(row) for row in xs)
            and (not xs or all(len(row) == len(xs[0]) for row in xs))
            and (not square or all(len(row) == len(xs) for row in xs)))


def intervals(xs):
    return type(xs) is list and all(ints(x) and len(x) == 2 and x[0] <= x[1] for x in xs)


def rotated(xs):
    if not ints(xs, nonempty=True) or len(set(xs)) != len(xs):
        return False
    return sum(xs[i] > xs[(i + 1) % len(xs)] for i in range(len(xs))) <= 1


def paired(xs, sorted_input=False):
    if not ints(xs, nonempty=True) or (sorted_input and not ordered(xs)):
        return False
    counts = Counter(xs).values()
    return sum(n == 1 for n in counts) == 1 and all(n in (1, 2) for n in counts)


def rpn(tokens):
    if not strings(tokens):
        return False
    stack = []
    for token in tokens:
        if token in ("+", "-", "*", "/"):
            if len(stack) < 2:
                return False
            b, a = stack.pop(), stack.pop()
            if token == "/" and b == 0:
                return False
            if token == "+": value = a + b
            elif token == "-": value = a - b
            elif token == "*": value = a * b
            else: value = (abs(a) // abs(b)) * (-1 if (a < 0) != (b < 0) else 1)
            if value.bit_length() > 63:
                return False
            stack.append(value)
        else:
            try:
                value = int(token)
            except ValueError:
                return False
            if value.bit_length() > 63:
                return False
            stack.append(value)
    return len(stack) == 1


def stack_ops(ops):
    if type(ops) is not list:
        return False
    depth = 0
    for op in ops:
        if type(op) not in (list, tuple) or len(op) != 2:
            return False
        name, value = op
        if name == "push" and integer(value): depth += 1
        elif name == "pop" and depth and value is None: depth -= 1
        elif name == "get_min" and depth and value is None: pass
        else: return False
    return True


def parentheses(s):
    if type(s) is not str or any(c not in "()" for c in s):
        return False
    depth = 0
    for c in s:
        depth += 1 if c == "(" else -1
        if depth < 0: return False
    return depth == 0


CONTRACTS = {}


def register(ids, predicate):
    for problem_id in ids.split():
        CONTRACTS[problem_id] = predicate


register("bs_standard_right_edge", lambda v: ordered(v["arr"]) and integer(v["target"]))
register("bs_insert_position bs_search_range", lambda v: ordered(v["nums"]) and integer(v["target"]))
register("bs_sqrt_floor", lambda v: integer(v["x"]) and v["x"] >= 0)
register("bs_peak_element", lambda v: ints(v["nums"], nonempty=True) and all(a != b for a, b in zip(v["nums"], v["nums"][1:])))
register("bs_find_min_rotated", lambda v: rotated(v["nums"]))
register("two_sum_unique_pair", lambda v: ints(v["nums"]) and integer(v["target"]) and sum(1 for i, a in enumerate(v["nums"]) for b in v["nums"][i+1:] if a+b == v["target"]) == 1)
register("two_pointers_remove_duplicates_length", lambda v: ordered(v["nums"]))
register("two_pointers_move_zeroes two_pointers_three_sum longest_increasing_contiguous product_except_self_zeros prefix_find_pivot_index prefix_running_sum monotonic_array_equality majority_element_verify prefix_min_value_step_sum", lambda v: ints(v["nums"]))
register("max_subarray_all_negative prefix_max_product_subarray", lambda v: ints(v["nums"], nonempty=True))
register("maximum_product_three", lambda v: ints(v["nums"]) and len(v["nums"]) >= 3)
register("two_pointers_container_water", lambda v: ints(v["height"], 0) and len(v["height"]) >= 2)
register("two_pointers_reverse_vowels sw_longest_non_repeating valid_palindrome_alphanumeric length_of_last_word first_unique_char two_pointers_valid_palindrome_ii atoi_clamping stack_backspace_str_len", lambda v: type(v["s"]) is str)
register("sw_min_size_subarray_sum", lambda v: ints(v["nums"], 1) and integer(v["target"]) and v["target"] > 0)
register("sw_contains_duplicate_ii", lambda v: ints(v["nums"]) and integer(v["k"]) and v["k"] >= 0)
register("sw_max_average_subarray_i", lambda v: ints(v["nums"]) and integer(v["k"]) and 1 <= v["k"] <= len(v["nums"]))
register("sw_subarray_product_less_than_k", lambda v: ints(v["nums"], 1) and integer(v["k"]))
register("sw_max_consecutive_ones_iii", lambda v: binary(v["nums"]) and integer(v["k"]) and v["k"] >= 0)
register("subarray_sum_equals_k_zero", lambda v: ints(v["nums"]) and integer(v["k"]))
register("intervals_merge_adjacent intervals_non_overlapping intervals_meeting_rooms", lambda v: intervals(v["intervals"]))
register("intervals_insert_interval", lambda v: intervals(v["intervals"]) and intervals([v["new_interval"]]) and all(a[1] < b[0] for a, b in zip(v["intervals"], v["intervals"][1:])))
register("can_jump_single_zero", lambda v: ints(v["nums"], 0, nonempty=True))
register("summary_ranges_single", lambda v: ordered(v["nums"], unique=True))
register("isomorphic_strings", lambda v: type(v["s"]) is str and type(v["t"]) is str and len(v["s"]) == len(v["t"]))
register("ransom_note_counts", lambda v: type(v["ransomNote"]) is str and type(v["magazine"]) is str)
register("word_pattern_bijection", lambda v: type(v["pattern"]) is str and type(v["s"]) is str)
register("group_anagrams_empty longest_common_prefix_empty", lambda v: strings(v["strs"]))
register("valid_parentheses_unclosed", lambda v: type(v["s"]) is str and all(c in "()[]{}" for c in v["s"]))
register("daily_temperatures_stack", lambda v: ints(v["temperatures"]))
register("eval_rpn_negative_division", lambda v: rpn(v["tokens"]))
register("stack_min_stack_operations", lambda v: stack_ops(v["ops"]))
register("stack_simplify_path", lambda v: type(v["path"]) is str and v["path"].startswith("/"))
register("stack_asteroid_collision", lambda v: ints(v["asteroids"]) and all(x != 0 for x in v["asteroids"]))
register("climbing_stairs_zero power_of_two_negative fizz_buzz_ordering", lambda v: integer(v["n"]))
register("palindrome_number_negative", lambda v: integer(v["x"]))
register("coin_change_unreachable", lambda v: ints(v["coins"], 1) and integer(v["amount"]) and v["amount"] >= 0)
register("house_robber_single", lambda v: ints(v["nums"], 0))
register("unique_paths_single_cell", lambda v: integer(v["m"]) and integer(v["n"]) and v["m"] > 0 and v["n"] > 0)
register("min_cost_climbing_stairs", lambda v: ints(v["cost"], 0) and len(v["cost"]) >= 2)
register("dp_tribonacci_number counting_bits_length dp_count_sorted_vowel_strings", lambda v: integer(v["n"]) and v["n"] >= 0)
register("single_number_find", lambda v: paired(v["nums"]))
register("add_binary_carry", lambda v: all(type(v[k]) is str and v[k] and all(c in "01" for c in v[k]) for k in ("a", "b")))
register("reverse_bits_32bit", lambda v: integer(v["n"]) and 0 <= v["n"] < 2**32)
register("missing_number_in_range", lambda v: ints(v["nums"], 0) and len(set(v["nums"])) == len(v["nums"]) and all(x <= len(v["nums"]) for x in v["nums"]))
register("chunk_list_zero", lambda v: type(v["lst"]) is list and integer(v["size"]))
register("find_all_duplicates", lambda v: ints(v["nums"], 1) and all(x <= len(v["nums"]) for x in v["nums"]) and all(n <= 2 for n in Counter(v["nums"]).values()))
register("bs_first_bad_version", lambda v: integer(v["n"]) and integer(v["b"]) and 1 <= v["b"] <= v["n"])
register("bs_search_matrix_flattened", lambda v: matrix(v["matrix"]) and integer(v["target"]) and ordered([x for row in v["matrix"] for x in row]))
register("two_pointers_sort_colors", lambda v: ints(v["nums"]) and all(x in (0, 1, 2) for x in v["nums"]))
register("sw_permutation_in_string", lambda v: type(v["s1"]) is str and type(v["s2"]) is str)
register("sw_find_all_anagrams", lambda v: type(v["s"]) is str and type(v["p"]) is str and bool(v["p"]))
register("prefix_contiguous_array_binary sw_longest_subarray_after_deleting_one", lambda v: binary(v["nums"]) and bool(v["nums"]))
register("prefix_range_sum_query", lambda v: ints(v["nums"]) and integer(v["left"]) and integer(v["right"]) and 0 <= v["left"] <= v["right"] < len(v["nums"]))
register("greedy_lemonade_change", lambda v: ints(v["bills"]) and all(x in (5, 10, 20) for x in v["bills"]))
register("greedy_assign_cookies", lambda v: ints(v["g"], 1) and ints(v["s"], 1))
register("str_str_empty_needle", lambda v: type(v["haystack"]) is str and type(v["needle"]) is str)
register("stack_make_good_string", lambda v: type(v["s"]) is str and all(c in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" for c in v["s"]))
register("stack_remove_outermost_parentheses", lambda v: parentheses(v["s"]))
register("dp_pascals_triangle_row", lambda v: integer(v["rowIndex"]) and v["rowIndex"] >= 0)
register("dp_divisor_game", lambda v: integer(v["n"]) and v["n"] > 0)
register("plus_one_digits", lambda v: ints(v["digits"]) and bool(v["digits"]) and all(0 <= x <= 9 for x in v["digits"]) and (len(v["digits"]) == 1 or v["digits"][0] != 0))
register("rotate_array_large", lambda v: ints(v["nums"]) and integer(v["k"]) and v["k"] >= 0)
register("matrix_transpose matrix_set_zeroes", lambda v: matrix(v["matrix"]))
register("matrix_diagonal_sum", lambda v: matrix(v["mat"], square=True))
register("bs_guess_number_higher_lower", lambda v: integer(v["n"]) and integer(v["pick"]) and 1 <= v["pick"] <= v["n"])
register("bs_single_non_duplicate", lambda v: paired(v["nums"], sorted_input=True))
register("two_pointers_backspace_compare is_subsequence_empty", lambda v: type(v["s"]) is str and type(v["t"]) is str)
register("two_pointers_remove_element", lambda v: ints(v["nums"]) and integer(v["val"]))
register("sw_num_subarrays_with_sum_k", lambda v: binary(v["nums"]) and integer(v["goal"]))
register("greedy_largest_number", lambda v: ints(v["nums"], 0, nonempty=True))
register("stack_final_prices_discount", lambda v: ints(v["prices"], 0))
register("dp_maximum_subarray_len_k", lambda v: ints(v["nums"], 1, nonempty=True))
register("gcd_both_zero", lambda v: integer(v["a"]) and integer(v["b"]) and v["a"] >= 0 and v["b"] >= 0)
register("hamming_distance_bits", lambda v: integer(v["x"]) and integer(v["y"]) and v["x"] >= 0 and v["y"] >= 0)


@lru_cache(maxsize=256)
def signature(code):
    func = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == "target_func")
    # Built-in problems have ordinary positional-or-keyword parameters, no defaults.
    return inspect.Signature([inspect.Parameter(a.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD) for a in func.args.args])


def validate_input(problem_id, code, args, kwargs):
    if problem_id not in CONTRACTS:
        raise ValueError(f"Missing input contract: {problem_id}")
    try:
        values = signature(code).bind(*args, **kwargs).arguments
        return bool(CONTRACTS[problem_id](values))
    except (TypeError, ValueError, KeyError, IndexError, OverflowError):
        return False
