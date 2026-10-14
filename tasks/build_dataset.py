"""
tasks/build_dataset.py - 自动化构建与全量双向沙盒验证 100 道 BugHunter 工业级缺陷题库
"""

import os
import sys
import json
import subprocess
from typing import List, Dict, Any

# 引入已验证的双向沙盒判定逻辑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tasks.bug_hunter import CURATED_BUG_HUNTER_PROBLEMS, verify_bug_exposure


def get_curated_100_problems() -> List[Dict[str, Any]]:
    problems = []

    # 1. 纳入已有的 15 道精选题，并附带验证反例
    existing_counter_examples = {
        "binary_search_right_edge": "assert target_func([1, 3, 5], 5) == 2",
        "longest_increasing_empty_list": "assert target_func([]) == 0",
        "merge_intervals_adjacent": "assert target_func([[1, 4], [4, 5]]) == [[1, 5]]",
        "max_subarray_all_negative": "assert target_func([-5, -2, -8]) == -2",
        "two_sum_same_element_reuse": "assert target_func([3, 2, 4], 6) == [1, 2]",
        "reverse_vowels_case_sensitivity": "assert target_func('Apple') == 'epplA'",
        "rotate_array_large_k": "assert target_func([1, 2, 3], 4) == [3, 1, 2]",
        "is_subsequence_empty_pattern": "assert target_func('', 'abc') == True",
        "is_prime_strict_inequality": "assert target_func(4) == False",
        "string_compression_negative_gain": "assert target_func('abcdef') == 'abcdef'",
        "climbing_stairs_zero_steps": "assert target_func(0) == 0",
        "first_unique_char_none_exists": "assert target_func('aabb') == -1",
        "chunk_list_zero_step": "assert target_func([1, 2, 3], 0) == []",
        "fizz_buzz_ordering_trap": "assert target_func(15) == ['1', '2', 'Fizz', '4', 'Buzz', 'Fizz', '7', '8', 'Fizz', 'Buzz', '11', 'Fizz', '13', '14', 'FizzBuzz']",
        "product_except_self_multiple_zeros": "assert target_func([0, 0, 2]) == [0, 0, 0]"
    }

    # 如果已有题库少于 15，先补齐已有的
    for p in CURATED_BUG_HUNTER_PROBLEMS:
        p_copy = dict(p)
        if p["id"] in existing_counter_examples:
            p_copy["counter_example"] = existing_counter_examples[p["id"]]
            problems.append(p_copy)

    # 2. 追加新增的高质量算法与软件工程缺陷题目 (按类别系统化扩充)
    new_raw_problems = [
        # --- [类别 1: 二分搜索与极值边界 (Binary Search & Bounds)] ---
        {
            "id": "bs_insert_position_missing_left",
            "spec": "Given a sorted array `nums` and a `target`, return the index if target is found. If not, return the index where it would be if inserted in order.",
            "buggy_code": (
                "def target_func(nums: list[int], target: int) -> int:\n"
                "    left, right = 0, len(nums) - 1\n"
                "    while left <= right:\n"
                "        mid = (left + right) // 2\n"
                "        if nums[mid] == target: return mid\n"
                "        elif nums[mid] < target: left = mid + 1\n"
                "        else: right = mid - 1\n"
                "    return mid  # BUG: should return left, not mid\n"
            ),
            "golden_code": (
                "def target_func(nums: list[int], target: int) -> int:\n"
                "    left, right = 0, len(nums) - 1\n"
                "    while left <= right:\n"
                "        mid = (left + right) // 2\n"
                "        if nums[mid] == target: return mid\n"
                "        elif nums[mid] < target: left = mid + 1\n"
                "        else: right = mid - 1\n"
                "    return left\n"
            ),
            "counter_example": "assert target_func([1, 3, 5, 6], 2) == 1"
        },
        {
            "id": "sqrt_floor_boundary_equality",
            "spec": "Given a non-negative integer x, compute and return the integer square root of x (floor of sqrt(x)). For x=0 return 0.",
            "buggy_code": (
                "def target_func(x: int) -> int:\n"
                "    if x <= 1: return x\n"
                "    l, r, ans = 1, x, 0\n"
                "    while l <= r:\n"
                "        m = (l + r) // 2\n"
                "        if m * m < x:  # BUG: should be m * m <= x, misses exact perfect squares\n"
                "            ans = m\n"
                "            l = m + 1\n"
                "        else:\n"
                "            r = m - 1\n"
                "    return ans\n"
            ),
            "golden_code": (
                "def target_func(x: int) -> int:\n"
                "    if x <= 1: return x\n"
                "    l, r, ans = 1, x, 0\n"
                "    while l <= r:\n"
                "        m = (l + r) // 2\n"
                "        if m * m <= x:\n"
                "            ans = m\n"
                "            l = m + 1\n"
                "        else:\n"
                "            r = m - 1\n"
                "    return ans\n"
            ),
            "counter_example": "assert target_func(4) == 2"
        },
        {
            "id": "peak_element_single_item",
            "spec": "An element is a peak if it is strictly greater than its neighbors. Given an integer array `nums`, return the index of any peak. For array of length 1, return 0.",
            "buggy_code": (
                "def target_func(nums: list[int]) -> int:\n"
                "    # BUG: directly accesses index 1 without checking len(nums) == 1\n"
                "    if nums[0] > nums[1]: return 0\n"
                "    for i in range(1, len(nums) - 1):\n"
                "        if nums[i] > nums[i-1] and nums[i] > nums[i+1]:\n"
                "            return i\n"
                "    return len(nums) - 1\n"
            ),
            "golden_code": (
                "def target_func(nums: list[int]) -> int:\n"
                "    if len(nums) == 1: return 0\n"
                "    if nums[0] > nums[1]: return 0\n"
                "    for i in range(1, len(nums) - 1):\n"
                "        if nums[i] > nums[i-1] and nums[i] > nums[i+1]:\n"
                "            return i\n"
                "    return len(nums) - 1\n"
            ),
            "counter_example": "assert target_func([5]) == 0"
        },
        {
            "id": "search_range_single_occurrence",
            "spec": "Given a sorted array of integers `nums` and a `target`, return the starting and ending position of `target` as [start, end]. If not found, return [-1, -1].",
            "buggy_code": (
                "def target_func(nums: list[int], target: int) -> list[int]:\n"
                "    if not nums: return [-1, -1]\n"
                "    first, last = -1, -1\n"
                "    for i, x in enumerate(nums):\n"
                "        if x == target:\n"
                "            if first == -1: first = i\n"
                "            # BUG: forgets to update last on each match, so last remains -1 if only one match\n"
                "        elif first != -1:\n"
                "            last = i - 1\n"
                "            break\n"
                "    return [first, last]\n"
            ),
            "golden_code": (
                "def target_func(nums: list[int], target: int) -> list[int]:\n"
                "    if not nums: return [-1, -1]\n"
                "    first, last = -1, -1\n"
                "    for i, x in enumerate(nums):\n"
                "        if x == target:\n"
                "            if first == -1: first = i\n"
                "            last = i\n"
                "    return [first, last]\n"
            ),
            "counter_example": "assert target_func([5, 7, 8], 8) == [2, 2]"
        },
        {
            "id": "find_min_rotated_sorted_array_no_rotation",
            "spec": "Given a sorted rotated array of unique elements `nums`, return the minimum element. The array might not be rotated at all.",
            "buggy_code": (
                "def target_func(nums: list[int]) -> int:\n"
                "    l, r = 0, len(nums) - 1\n"
                "    # BUG: assumes array is always rotated, returns nums[r] when nums[l] < nums[r]\n"
                "    if nums[l] < nums[r]: return nums[r]\n"
                "    while l < r:\n"
                "        m = (l + r) // 2\n"
                "        if nums[m] > nums[r]: l = m + 1\n"
                "        else: r = m\n"
                "    return nums[l]\n"
            ),
            "golden_code": (
                "def target_func(nums: list[int]) -> int:\n"
                "    l, r = 0, len(nums) - 1\n"
                "    while l < r:\n"
                "        m = (l + r) // 2\n"
                "        if nums[m] > nums[r]: l = m + 1\n"
                "        else: r = m\n"
                "    return nums[l]\n"
            ),
            "counter_example": "assert target_func([1, 2, 3]) == 1"
        },

        # --- [类别 2: 数组操作与前缀和 (Arrays, Subarrays & Prefix Sum)] ---
        {
            "id": "can_jump_single_zero_reach",
            "spec": "Given an integer array `nums` where each element represents your max jump length at that position, return True if you can reach the last index, False otherwise.",
            "buggy_code": (
                "def target_func(nums: list[int]) -> bool:\n"
                "    max_reach = 0\n"
                "    for i in range(len(nums)):\n"
                "        # BUG: does not check if current index i is unreachable before updating max_reach\n"
                "        max_reach = max(max_reach, i + nums[i])\n"
                "        if max_reach >= len(nums) - 1: return True\n"
                "    return False\n"
            ),
            "golden_code": (
                "def target_func(nums: list[int]) -> bool:\n"
                "    max_reach = 0\n"
                "    for i in range(len(nums)):\n"
                "        if i > max_reach: return False\n"
                "        max_reach = max(max_reach, i + nums[i])\n"
                "        if max_reach >= len(nums) - 1: return True\n"
                "    return False\n"
            ),
            "counter_example": "assert target_func([0, 2, 3]) == False"
        },
        {
            "id": "subarray_sum_equals_k_zero_prefix",
            "spec": "Given an array of integers `nums` and an integer `k`, return the total number of subarrays whose sum equals `k`.",
            "buggy_code": (
                "def target_func(nums: list[int], k: int) -> int:\n"
                "    # BUG: forgets to initialize count for prefix sum 0 with 1, missing subarrays starting at index 0\n"
                "    prefix = {}\n"
                "    cur, ans = 0, 0\n"
                "    for x in nums:\n"
                "        cur += x\n"
                "        if cur - k in prefix:\n"
                "            ans += prefix[cur - k]\n"
                "        prefix[cur] = prefix.get(cur, 0) + 1\n"
                "    return ans\n"
            ),
            "golden_code": (
                "def target_func(nums: list[int], k: int) -> int:\n"
                "    prefix = {0: 1}\n"
                "    cur, ans = 0, 0\n"
                "    for x in nums:\n"
                "        cur += x\n"
                "        if cur - k in prefix:\n"
                "            ans += prefix[cur - k]\n"
                "        prefix[cur] = prefix.get(cur, 0) + 1\n"
                "    return ans\n"
            ),
            "counter_example": "assert target_func([1, 1, 1], 2) == 2"
        },
        {
            "id": "remove_duplicates_inplace_length",
            "spec": "Given an integer array `nums` sorted in non-decreasing order, remove duplicates in-place such that each unique element appears only once. Return the number of unique elements `k`.",
            "buggy_code": (
                "def target_func(nums: list[int]) -> int:\n"
                "    # BUG: misses empty list check, crashes on nums[0]\n"
                "    slow = 0\n"
                "    for fast in range(1, len(nums)):\n"
                "        if nums[fast] != nums[slow]:\n"
                "            slow += 1\n"
                "            nums[slow] = nums[fast]\n"
                "    return slow + 1\n"
            ),
            "golden_code": (
                "def target_func(nums: list[int]) -> int:\n"
                "    if not nums: return 0\n"
                "    slow = 0\n"
                "    for fast in range(1, len(nums)):\n"
                "        if nums[fast] != nums[slow]:\n"
                "            slow += 1\n"
                "            nums[slow] = nums[fast]\n"
                "    return slow + 1\n"
            ),
            "counter_example": "assert target_func([]) == 0"
        },
        {
            "id": "move_zeroes_relative_order",
            "spec": "Given an integer array `nums`, move all 0's to the end of it while maintaining the relative order of the non-zero elements. Return the modified list.",
            "buggy_code": (
                "def target_func(nums: list[int]) -> list[int]:\n"
                "    # BUG: uses two pointers from both ends, reversing non-zero relative order\n"
                "    l, r = 0, len(nums) - 1\n"
                "    while l < r:\n"
                "        if nums[l] == 0:\n"
                "            nums[l], nums[r] = nums[r], nums[l]\n"
                "            r -= 1\n"
                "        else:\n"
                "            l += 1\n"
                "    return nums\n"
            ),
            "golden_code": (
                "def target_func(nums: list[int]) -> list[int]:\n"
                "    non_zeros = [x for x in nums if x != 0]\n"
                "    zeros = [0] * (len(nums) - len(non_zeros))\n"
                "    return non_zeros + zeros\n"
            ),
            "counter_example": "assert target_func([0, 1, 0, 3, 12]) == [1, 3, 12, 0, 0]"
        },
        {
            "id": "contains_duplicate_ii_distance_strict",
            "spec": "Given an integer array `nums` and an integer `k`, return True if there are two distinct indices i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.",
            "buggy_code": (
                "def target_func(nums: list[int], k: int) -> bool:\n"
                "    seen = {}\n"
                "    for i, x in enumerate(nums):\n"
                "        # BUG: uses strictly less than instead of <= k\n"
                "        if x in seen and i - seen[x] < k:\n"
                "            return True\n"
                "        seen[x] = i\n"
                "    return False\n"
            ),
            "golden_code": (
                "def target_func(nums: list[int], k: int) -> bool:\n"
                "    seen = {}\n"
                "    for i, x in enumerate(nums):\n"
                "        if x in seen and i - seen[x] <= k:\n"
                "            return True\n"
                "        seen[x] = i\n"
                "    return False\n"
            ),
            "counter_example": "assert target_func([1, 2, 3, 1], 3) == True"
        },
        {
            "id": "majority_element_verify_strict_half",
            "spec": "Given an array `nums` of size n, return the majority element (element appearing strictly more than n // 2 times). If no such element exists, return None.",
            "buggy_code": (
                "def target_func(nums: list[int]) -> int | None:\n"
                "    # BUG: Boyer-Moore without verification, returns candidate even when no majority exists\n"
                "    count, cand = 0, None\n"
                "    for x in nums:\n"
                "        if count == 0: cand = x\n"
                "        count += (1 if x == cand else -1)\n"
                "    return cand\n"
            ),
            "golden_code": (
                "def target_func(nums: list[int]) -> int | None:\n"
                "    count, cand = 0, None\n"
                "    for x in nums:\n"
                "        if count == 0: cand = x\n"
                "        count += (1 if x == cand else -1)\n"
                "    if nums.count(cand) > len(nums) // 2: return cand\n"
                "    return None\n"
            ),
            "counter_example": "assert target_func([1, 2, 3]) == None"
        },
        {
            "id": "find_all_duplicates_one_missing",
            "spec": "Given an integer array `nums` of length n where all integers are in range [1, n] and each integer appears once or twice, return an array of all integers that appear twice.",
            "buggy_code": (
                "def target_func(nums: list[int]) -> list[int]:\n"
                "    from collections import Counter\n"
                "    c = Counter(nums)\n"
                "    # BUG: returns items appearing >= 1 time instead of == 2\n"
                "    return [k for k, v in c.items() if v >= 1]\n"
            ),
            "golden_code": (
                "def target_func(nums: list[int]) -> list[int]:\n"
                "    from collections import Counter\n"
                "    c = Counter(nums)\n"
                "    return sorted([k for k, v in c.items() if v == 2])\n"
            ),
            "counter_example": "assert target_func([4, 3, 2, 7, 8, 2, 3, 1]) == [2, 3]"
        },
        {
            "id": "maximum_product_three_numbers_negative",
            "spec": "Given an integer array `nums`, find three numbers whose product is maximum and return the maximum product.",
            "buggy_code": (
                "def target_func(nums: list[int]) -> int:\n"
                "    nums.sort()\n"
                "    # BUG: only considers the three largest positive numbers, misses product of two large negatives and one positive\n"
                "    return nums[-1] * nums[-2] * nums[-3]\n"
            ),
            "golden_code": (
                "def target_func(nums: list[int]) -> int:\n"
                "    nums.sort()\n"
                "    return max(nums[-1] * nums[-2] * nums[-3], nums[0] * nums[1] * nums[-1])\n"
            ),
            "counter_example": "assert target_func([-10, -10, 5, 2]) == 500"
        },
        {
            "id": "monotonic_array_all_equal",
            "spec": "An array is monotonic if it is either monotone increasing or monotone decreasing. Given an integer array `nums`, return True if monotonic, False otherwise.",
            "buggy_code": (
                "def target_func(nums: list[int]) -> bool:\n"
                "    # BUG: checks strict inequality, claims array with equal elements is not monotonic\n"
                "    inc = all(nums[i] < nums[i+1] for i in range(len(nums)-1))\n"
                "    dec = all(nums[i] > nums[i+1] for i in range(len(nums)-1))\n"
                "    return inc or dec\n"
            ),
            "golden_code": (
                "def target_func(nums: list[int]) -> bool:\n"
                "    inc = all(nums[i] <= nums[i+1] for i in range(len(nums)-1))\n"
                "    dec = all(nums[i] >= nums[i+1] for i in range(len(nums)-1))\n"
                "    return inc or dec\n"
            ),
            "counter_example": "assert target_func([1, 1, 1]) == True"
        },
        {
            "id": "summary_ranges_single_element",
            "spec": "Return the smallest sorted list of ranges that cover all numbers in the sorted array `nums`. Format single element as 'x' and range as 'a->b'.",
            "buggy_code": (
                "def target_func(nums: list[int]) -> list[str]:\n"
                "    if not nums: return []\n"
                "    res = []\n"
                "    start = nums[0]\n"
                "    for i in range(1, len(nums)):\n"
                "        if nums[i] != nums[i-1] + 1:\n"
                "            # BUG: always formats as 'a->b' even for single element\n"
                "            res.append(f'{start}->{nums[i-1]}')\n"
                "            start = nums[i]\n"
                "    res.append(f'{start}->{nums[-1]}')\n"
                "    return res\n"
            ),
            "golden_code": (
                "def target_func(nums: list[int]) -> list[str]:\n"
                "    if not nums: return []\n"
                "    res = []\n"
                "    start = nums[0]\n"
                "    for i in range(1, len(nums)):\n"
                "        if nums[i] != nums[i-1] + 1:\n"
                "            res.append(str(start) if start == nums[i-1] else f'{start}->{nums[i-1]}')\n"
                "            start = nums[i]\n"
                "    res.append(str(start) if start == nums[-1] else f'{start}->{nums[-1]}')\n"
                "    return res\n"
            ),
            "counter_example": "assert target_func([0, 1, 2, 4, 5, 7]) == ['0->2', '4->5', '7']"
        },

        # --- [类别 3: 字符串与文本格式解析 (Strings & Parsing)] ---
        {
            "id": "valid_parentheses_unclosed_stack",
            "spec": "Given a string `s` containing '(', ')', '{', '}', '[' and ']', determine if the input string is valid.",
            "buggy_code": (
                "def target_func(s: str) -> bool:\n"
                "    stack = []\n"
                "    mapping = {')': '(', '}': '{', ']': '['}\n"
                "    for char in s:\n"
                "        if char in mapping:\n"
                "            top = stack.pop() if stack else '#'\n"
                "            if mapping[char] != top: return False\n"
                "        else:\n"
                "            stack.append(char)\n"
                "    # BUG: forgets to check if stack is empty, returns True for unclosed brackets like '('\n"
                "    return True\n"
            ),
            "golden_code": (
                "def target_func(s: str) -> bool:\n"
                "    stack = []\n"
                "    mapping = {')': '(', '}': '{', ']': '['}\n"
                "    for char in s:\n"
                "        if char in mapping:\n"
                "            top = stack.pop() if stack else '#'\n"
                "            if mapping[char] != top: return False\n"
                "        else:\n"
                "            stack.append(char)\n"
                "    return len(stack) == 0\n"
            ),
            "counter_example": "assert target_func('(') == False"
        },
        {
            "id": "longest_common_prefix_empty_list",
            "spec": "Write a function to find the longest common prefix string amongst an array of strings. If there is no common prefix, return an empty string ''.",
            "buggy_code": (
                "def target_func(strs: list[str]) -> str:\n"
                "    # BUG: directly accesses strs[0] on empty list without checking\n"
                "    prefix = strs[0]\n"
                "    for s in strs[1:]:\n"
                "        while not s.startswith(prefix):\n"
                "            prefix = prefix[:-1]\n"
                "            if not prefix: return ''\n"
                "    return prefix\n"
            ),
            "golden_code": (
                "def target_func(strs: list[str]) -> str:\n"
                "    if not strs: return ''\n"
                "    prefix = strs[0]\n"
                "    for s in strs[1:]:\n"
                "        while not s.startswith(prefix):\n"
                "            prefix = prefix[:-1]\n"
                "            if not prefix: return ''\n"
                "    return prefix\n"
            ),
            "counter_example": "assert target_func([]) == ''"
        },
        {
            "id": "valid_palindrome_alphanumeric_lower",
            "spec": "A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward.",
            "buggy_code": (
                "def target_func(s: str) -> bool:\n"
                "    # BUG: forgets lower(), fails on case difference\n"
                "    cleaned = [c for c in s if c.isalnum()]\n"
                "    return cleaned == cleaned[::-1]\n"
            ),
            "golden_code": (
                "def target_func(s: str) -> bool:\n"
                "    cleaned = [c.lower() for c in s if c.isalnum()]\n"
                "    return cleaned == cleaned[::-1]\n"
            ),
            "counter_example": "assert target_func('A man, a plan, a canal: Panama') == True"
        },
        {
            "id": "length_of_last_word_trailing_spaces",
            "spec": "Given a string `s` consisting of words and spaces, return the length of the last word in the string. A word is a maximal substring consisting of non-space characters only.",
            "buggy_code": (
                "def target_func(s: str) -> int:\n"
                "    # BUG: uses s.split(' ') without stripping trailing spaces, so last element is ''\n"
                "    parts = s.split(' ')\n"
                "    return len(parts[-1])\n"
            ),
            "golden_code": (
                "def target_func(s: str) -> int:\n"
                "    parts = s.strip().split()\n"
                "    return len(parts[-1]) if parts else 0\n"
            ),
            "counter_example": "assert target_func('   fly me   to   the moon  ') == 4"
        },
        {
            "id": "isomorphic_strings_two_way_mapping",
            "spec": "Two strings `s` and `t` are isomorphic if the characters in `s` can be replaced to get `t`. No two characters may map to the same character.",
            "buggy_code": (
                "def target_func(s: str, t: int | str) -> bool:\n"
                "    # BUG: only checks one-way mapping, fails when two distinct chars in s map to same char in t\n"
                "    if len(s) != len(t): return False\n"
                "    m = {}\n"
                "    for c1, c2 in zip(s, t):\n"
                "        if c1 in m and m[c1] != c2: return False\n"
                "        m[c1] = c2\n"
                "    return True\n"
            ),
            "golden_code": (
                "def target_func(s: str, t: str) -> bool:\n"
                "    if len(s) != len(t): return False\n"
                "    return len(set(zip(s, t))) == len(set(s)) == len(set(t))\n"
            ),
            "counter_example": "assert target_func('badc', 'baba') == False"
        },
        {
            "id": "ransom_note_char_counts",
            "spec": "Given two strings `ransomNote` and `magazine`, return True if ransomNote can be constructed by using the letters from magazine, False otherwise. Each letter in magazine can only be used once.",
            "buggy_code": (
                "def target_func(ransomNote: str, magazine: str) -> bool:\n"
                "    # BUG: uses 'in' instead of tracking counts, allowing reuse of characters\n"
                "    return all(c in magazine for c in ransomNote)\n"
            ),
            "golden_code": (
                "def target_func(ransomNote: str, magazine: str) -> bool:\n"
                "    from collections import Counter\n"
                "    c1, c2 = Counter(ransomNote), Counter(magazine)\n"
                "    return all(c2[k] >= v for k, v in c1.items())\n"
            ),
            "counter_example": "assert target_func('aa', 'ab') == False"
        },
        {
            "id": "word_pattern_bijective_check",
            "spec": "Given a `pattern` and a string `s`, find if `s` follows the same pattern (bijection between pattern letter and non-empty word in s).",
            "buggy_code": (
                "def target_func(pattern: str, s: str) -> bool:\n"
                "    words = s.split()\n"
                "    if len(pattern) != len(words): return False\n"
                "    p_to_w = {}\n"
                "    for p, w in zip(pattern, words):\n"
                "        # BUG: one-way mapping, fails when 'a' and 'b' map to same word 'dog'\n"
                "        if p in p_to_w and p_to_w[p] != w: return False\n"
                "        p_to_w[p] = w\n"
                "    return True\n"
            ),
            "golden_code": (
                "def target_func(pattern: str, s: str) -> bool:\n"
                "    words = s.split()\n"
                "    if len(pattern) != len(words): return False\n"
                "    return len(set(zip(pattern, words))) == len(set(pattern)) == len(set(words))\n"
            ),
            "counter_example": "assert target_func('abba', 'dog dog dog dog') == False"
        },
        {
            "id": "group_anagrams_empty_string",
            "spec": "Given an array of strings `strs`, group the anagrams together in any order. Return list of lists of strings.",
            "buggy_code": (
                "def target_func(strs: list[str]) -> list[list[str]]:\n"
                "    from collections import defaultdict\n"
                "    groups = defaultdict(list)\n"
                "    for s in strs:\n"
                "        # BUG: filters out empty strings\n"
                "        if s:\n"
                "            groups[''.join(sorted(s))].append(s)\n"
                "    return sorted(list(groups.values()), key=len)\n"
            ),
            "golden_code": (
                "def target_func(strs: list[str]) -> list[list[str]]:\n"
                "    from collections import defaultdict\n"
                "    groups = defaultdict(list)\n"
                "    for s in strs:\n"
                "        groups[''.join(sorted(s))].append(s)\n"
                "    return sorted(list(groups.values()), key=len)\n"
            ),
            "counter_example": "assert target_func(['']) == [['']]"
        },
        {
            "id": "str_str_empty_needle",
            "spec": "Given two strings `needle` and `haystack`, return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack. If needle is empty, return 0.",
            "buggy_code": (
                "def target_func(haystack: str, needle: str) -> int:\n"
                "    # BUG: returns -1 for empty needle\n"
                "    if not needle: return -1\n"
                "    return haystack.find(needle)\n"
            ),
            "golden_code": (
                "def target_func(haystack: str, needle: str) -> int:\n"
                "    if not needle: return 0\n"
                "    return haystack.find(needle)\n"
            ),
            "counter_example": "assert target_func('hello', '') == 0"
        },
        {
            "id": "atoi_integer_clamping",
            "spec": "Implement myAtoi which converts string to a 32-bit signed integer clamped within [-2**31, 2**31 - 1].",
            "buggy_code": (
                "def target_func(s: str) -> int:\n"
                "    s = s.strip()\n"
                "    if not s: return 0\n"
                "    sign = -1 if s[0] == '-' else 1\n"
                "    if s[0] in ['+', '-']: s = s[1:]\n"
                "    digits = []\n"
                "    for c in s:\n"
                "        if c.isdigit(): digits.append(c)\n"
                "        else: break\n"
                "    # BUG: forgets 32-bit clamping bounds [-2**31, 2**31 - 1]\n"
                "    return sign * int(''.join(digits)) if digits else 0\n"
            ),
            "golden_code": (
                "def target_func(s: str) -> int:\n"
                "    s = s.strip()\n"
                "    if not s: return 0\n"
                "    sign = -1 if s[0] == '-' else 1\n"
                "    if s[0] in ['+', '-']: s = s[1:]\n"
                "    digits = []\n"
                "    for c in s:\n"
                "        if c.isdigit(): digits.append(c)\n"
                "        else: break\n"
                "    val = sign * int(''.join(digits)) if digits else 0\n"
                "    return max(-2**31, min(2**31 - 1, val))\n"
            ),
            "counter_example": "assert target_func('91283472332') == 2147483647"
        },

        # --- [类别 4: 双指针与滑动窗口 (Two Pointers & Sliding Window)] ---
        {
            "id": "container_water_equal_height_decrement",
            "spec": "Given n non-negative integers `height` where each represents a point (i, ai). Find two lines that together with the x-axis form a container containing the most water.",
            "buggy_code": (
                "def target_func(height: list[int]) -> int:\n"
                "    # BUG: uses max instead of min to calculate container capacity\n"
                "    l, r, ans = 0, len(height) - 1, 0\n"
                "    while l < r:\n"
                "        ans = max(ans, max(height[l], height[r]) * (r - l))\n"
                "        if height[l] < height[r]: l += 1\n"
                "        else: r -= 1\n"
                "    return ans\n"
            ),
            "golden_code": (
                "def target_func(height: list[int]) -> int:\n"
                "    l, r, ans = 0, len(height) - 1, 0\n"
                "    while l < r:\n"
                "        ans = max(ans, min(height[l], height[r]) * (r - l))\n"
                "        if height[l] < height[r]: l += 1\n"
                "        else: r -= 1\n"
                "    return ans\n"
            ),
            "counter_example": "assert target_func([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49"
        },
        {
            "id": "longest_substring_without_repeating_index_jump",
            "spec": "Given a string `s`, find the length of the longest substring without duplicate characters.",
            "buggy_code": (
                "def target_func(s: str) -> int:\n"
                "    used = {}\n"
                "    start = 0\n"
                "    max_len = 0\n"
                "    for i, c in enumerate(s):\n"
                "        if c in used:\n"
                "            # BUG: sets start = used[c] + 1 without checking if used[c] < start, moving start backwards!\n"
                "            start = used[c] + 1\n"
                "        used[c] = i\n"
                "        max_len = max(max_len, i - start + 1)\n"
                "    return max_len\n"
            ),
            "golden_code": (
                "def target_func(s: str) -> int:\n"
                "    used = {}\n"
                "    start = 0\n"
                "    max_len = 0\n"
                "    for i, c in enumerate(s):\n"
                "        if c in used and used[c] >= start:\n"
                "            start = used[c] + 1\n"
                "        used[c] = i\n"
                "        max_len = max(max_len, i - start + 1)\n"
                "    return max_len\n"
            ),
            "counter_example": "assert target_func('abba') == 2"
        },
        {
            "id": "three_sum_duplicate_skip_infinite_loop",
            "spec": "Given an integer array `nums`, return all triplets [nums[i], nums[j], nums[k]] such that i != j != k and nums[i] + nums[j] + nums[k] == 0. No duplicate triplets.",
            "buggy_code": (
                "def target_func(nums: list[int]) -> list[list[int]]:\n"
                "    nums.sort()\n"
                "    res = []\n"
                "    for i in range(len(nums) - 2):\n"
                "        if i > 0 and nums[i] == nums[i-1]: continue\n"
                "        l, r = i + 1, len(nums) - 1\n"
                "        while l < r:\n"
                "            s = nums[i] + nums[l] + nums[r]\n"
                "            if s == 0:\n"
                "                res.append([nums[i], nums[l], nums[r]])\n"
                "                # BUG: forgets to increment l or decrement r after finding match, or skips duplicates incorrectly\n"
                "                l += 1\n"
                "                # missing duplicate skipping logic for l and r\n"
                "            elif s < 0: l += 1\n"
                "            else: r -= 1\n"
                "    return sorted(res)\n"
            ),
            "golden_code": (
                "def target_func(nums: list[int]) -> list[list[int]]:\n"
                "    nums.sort()\n"
                "    res = set()\n"
                "    for i in range(len(nums) - 2):\n"
                "        l, r = i + 1, len(nums) - 1\n"
                "        while l < r:\n"
                "            s = nums[i] + nums[l] + nums[r]\n"
                "            if s == 0:\n"
                "                res.add((nums[i], nums[l], nums[r]))\n"
                "                l += 1\n"
                "                r -= 1\n"
                "            elif s < 0: l += 1\n"
                "            else: r -= 1\n"
                "    return sorted([list(t) for t in res])\n"
            ),
            "counter_example": "assert target_func([-2, 0, 0, 2, 2]) == [[-2, 0, 2]]"
        },
        {
            "id": "minimum_size_subarray_sum_impossible",
            "spec": "Given an array of positive integers `nums` and a positive integer `target`, return the minimal length of a contiguous subarray of which the sum is >= target. If no such subarray, return 0.",
            "buggy_code": (
                "def target_func(target: int, nums: list[int]) -> int:\n"
                "    l, cur, ans = 0, 0, len(nums) + 1\n"
                "    for r in range(len(nums)):\n"
                "        cur += nums[r]\n"
                "        while cur >= target:\n"
                "            ans = min(ans, r - l + 1)\n"
                "            cur -= nums[l]\n"
                "            l += 1\n"
                "    # BUG: returns -1 instead of 0 when impossible\n"
                "    return ans if ans <= len(nums) else -1\n"
            ),
            "golden_code": (
                "def target_func(target: int, nums: list[int]) -> int:\n"
                "    l, cur, ans = 0, 0, len(nums) + 1\n"
                "    for r in range(len(nums)):\n"
                "        cur += nums[r]\n"
                "        while cur >= target:\n"
                "            ans = min(ans, r - l + 1)\n"
                "            cur -= nums[l]\n"
                "            l += 1\n"
                "    return ans if ans <= len(nums) else 0\n"
            ),
            "counter_example": "assert target_func(11, [1, 1, 1, 1]) == 0"
        },

        # --- [类别 5: 动态规划与递归基 (Dynamic Programming Edge Cases)] ---
        {
            "id": "coin_change_unreachable_amount",
            "spec": "Return the fewest number of coins that you need to make up `amount`. If that amount cannot be made up by any combination of coins, return -1. For amount 0, return 0.",
            "buggy_code": (
                "def target_func(coins: list[int], amount: int) -> int:\n"
                "    dp = [float('inf')] * (amount + 1)\n"
                "    dp[0] = 0\n"
                "    for c in coins:\n"
                "        for x in range(c, amount + 1):\n"
                "            dp[x] = min(dp[x], dp[x - c] + 1)\n"
                "    # BUG: returns float('inf') instead of -1 when unreachable\n"
                "    return dp[amount]\n"
            ),
            "golden_code": (
                "def target_func(coins: list[int], amount: int) -> int:\n"
                "    dp = [float('inf')] * (amount + 1)\n"
                "    dp[0] = 0\n"
                "    for c in coins:\n"
                "        for x in range(c, amount + 1):\n"
                "            dp[x] = min(dp[x], dp[x - c] + 1)\n"
                "    return int(dp[amount]) if dp[amount] != float('inf') else -1\n"
            ),
            "counter_example": "assert target_func([2], 3) == -1"
        },
        {
            "id": "house_robber_single_house",
            "spec": "Determine the maximum amount of money you can rob tonight without alerting the police (cannot rob adjacent houses).",
            "buggy_code": (
                "def target_func(nums: list[int]) -> int:\n"
                "    # BUG: directly accesses nums[1] on length 1 list\n"
                "    dp = [0] * len(nums)\n"
                "    dp[0] = nums[0]\n"
                "    dp[1] = max(nums[0], nums[1])\n"
                "    for i in range(2, len(nums)):\n"
                "        dp[i] = max(dp[i-1], dp[i-2] + nums[i])\n"
                "    return dp[-1]\n"
            ),
            "golden_code": (
                "def target_func(nums: list[int]) -> int:\n"
                "    if not nums: return 0\n"
                "    if len(nums) == 1: return nums[0]\n"
                "    prev2, prev1 = nums[0], max(nums[0], nums[1])\n"
                "    for i in range(2, len(nums)):\n"
                "        prev2, prev1 = prev1, max(prev1, prev2 + nums[i])\n"
                "    return prev1\n"
            ),
            "counter_example": "assert target_func([5]) == 5"
        },
        {
            "id": "unique_paths_single_cell_grid",
            "spec": "A robot is located at the top-left corner of a m x n grid. The robot can only move either down or right. Return number of unique paths to the bottom-right.",
            "buggy_code": (
                "def target_func(m: int, n: int) -> int:\n"
                "    # BUG: returns 0 for 1x1 grid\n"
                "    if m == 1 and n == 1: return 0\n"
                "    import math\n"
                "    return math.comb(m + n - 2, m - 1)\n"
            ),
            "golden_code": (
                "def target_func(m: int, n: int) -> int:\n"
                "    import math\n"
                "    return math.comb(m + n - 2, m - 1)\n"
            ),
            "counter_example": "assert target_func(1, 1) == 1"
        },
        {
            "id": "min_cost_climbing_stairs_two_steps",
            "spec": "Given an integer array `cost` where cost[i] is the cost of ith step, return the minimum cost to reach the top. You can start from index 0 or 1.",
            "buggy_code": (
                "def target_func(cost: list[int]) -> int:\n"
                "    # BUG: assumes you must start at index 0\n"
                "    dp = [0] * (len(cost) + 1)\n"
                "    dp[0] = cost[0]\n"
                "    dp[1] = cost[0] + cost[1]\n"
                "    for i in range(2, len(cost)):\n"
                "        dp[i] = cost[i] + min(dp[i-1], dp[i-2])\n"
                "    dp[-1] = min(dp[-2], dp[-3])\n"
                "    return dp[-1]\n"
            ),
            "golden_code": (
                "def target_func(cost: list[int]) -> int:\n"
                "    a, b = 0, 0\n"
                "    for c in cost:\n"
                "        a, b = b, c + min(a, b)\n"
                "    return min(a, b)\n"
            ),
            "counter_example": "assert target_func([10, 15, 20]) == 15"
        },

        # --- [类别 6: 栈与括号单调结构 (Stack & Monotonic)] ---
        {
            "id": "daily_temperatures_monotonic_stack_zero",
            "spec": "Given an array of integers `temperatures`, return an array `answer` such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature. If no future day is warmer, keep answer[i] == 0.",
            "buggy_code": (
                "def target_func(temperatures: list[int]) -> list[int]:\n"
                "    res = [0] * len(temperatures)\n"
                "    stack = []\n"
                "    for i, t in enumerate(temperatures):\n"
                "        # BUG: uses strictly <= instead of <, popping equal temperatures and claiming immediate warmer day\n"
                "        while stack and temperatures[stack[-1]] <= t:\n"
                "            prev = stack.pop()\n"
                "            res[prev] = i - prev\n"
                "        stack.append(i)\n"
                "    return res\n"
            ),
            "golden_code": (
                "def target_func(temperatures: list[int]) -> list[int]:\n"
                "    res = [0] * len(temperatures)\n"
                "    stack = []\n"
                "    for i, t in enumerate(temperatures):\n"
                "        while stack and temperatures[stack[-1]] < t:\n"
                "            prev = stack.pop()\n"
                "            res[prev] = i - prev\n"
                "        stack.append(i)\n"
                "    return res\n"
            ),
            "counter_example": "assert target_func([30, 30, 30]) == [0, 0, 0]"
        },
        {
            "id": "eval_rpn_negative_division_truncate",
            "spec": "Evaluate the value of an arithmetic expression in Reverse Polish Notation. Valid operators are +, -, *, /. Division between two integers should truncate toward zero.",
            "buggy_code": (
                "def target_func(tokens: list[str]) -> int:\n"
                "    stack = []\n"
                "    for t in tokens:\n"
                "        if t in ['+', '-', '*', '/']:\n"
                "            b, a = stack.pop(), stack.pop()\n"
                "            if t == '+': stack.append(a + b)\n"
                "            elif t == '-': stack.append(a - b)\n"
                "            elif t == '*': stack.append(a * b)\n"
                "            # BUG: uses python floor division '//' which rounds toward negative infinity for negative numbers!\n"
                "            elif t == '/': stack.append(a // b)\n"
                "        else:\n"
                "            stack.append(int(t))\n"
                "    return stack[0]\n"
            ),
            "golden_code": (
                "def target_func(tokens: list[str]) -> int:\n"
                "    stack = []\n"
                "    for t in tokens:\n"
                "        if t in ['+', '-', '*', '/']:\n"
                "            b, a = stack.pop(), stack.pop()\n"
                "            if t == '+': stack.append(a + b)\n"
                "            elif t == '-': stack.append(a - b)\n"
                "            elif t == '*': stack.append(a * b)\n"
                "            elif t == '/': stack.append(int(a / b))\n"
                "        else:\n"
                "            stack.append(int(t))\n"
                "    return stack[0]\n"
            ),
            "counter_example": "assert target_func(['4', '-3', '/']) == -1"
        },

        # --- [类别 7: 数学、位运算与进位 (Math, Bits & Overflow)] ---
        {
            "id": "power_of_two_zero_and_negative_input",
            "spec": "Given an integer n, return True if it is a power of two. Otherwise, return False.",
            "buggy_code": (
                "def target_func(n: int) -> bool:\n"
                "    # BUG: n & (n - 1) == 0 returns True for n = 0\n"
                "    return (n & (n - 1)) == 0\n"
            ),
            "golden_code": (
                "def target_func(n: int) -> bool:\n"
                "    return n > 0 and (n & (n - 1)) == 0\n"
            ),
            "counter_example": "assert target_func(0) == False"
        },
        {
            "id": "palindrome_number_negative_check",
            "spec": "Given an integer x, return True if x is a palindrome, and False otherwise. Negative numbers are not palindromes.",
            "buggy_code": (
                "def target_func(x: int) -> bool:\n"
                "    # BUG: converts directly to string without negative check, so -121 reversed is 121- which fails, but doesn't handle -0\n"
                "    # Better bug: strips '-' before checking\n"
                "    s = str(abs(x))\n"
                "    return s == s[::-1]\n"
            ),
            "golden_code": (
                "def target_func(x: int) -> bool:\n"
                "    if x < 0: return False\n"
                "    s = str(x)\n"
                "    return s == s[::-1]\n"
            ),
            "counter_example": "assert target_func(-121) == False"
        },
        {
            "id": "single_number_all_pairs_with_zero",
            "spec": "Given a non-empty array of integers `nums`, every element appears twice except for one. Find that single one.",
            "buggy_code": (
                "def target_func(nums: list[int]) -> int:\n"
                "    from collections import Counter\n"
                "    c = Counter(nums)\n"
                "    # BUG: returns first key without checking value == 1\n"
                "    return nums[0]\n"
            ),
            "golden_code": (
                "def target_func(nums: list[int]) -> int:\n"
                "    from collections import Counter\n"
                "    c = Counter(nums)\n"
                "    for k, v in c.items():\n"
                "        if v == 1: return k\n"
                "    return 0\n"
            ),
            "counter_example": "assert target_func([1, 1, 2, 2, 4]) == 4"
        },
        {
            "id": "add_binary_carry_overflow_at_end",
            "spec": "Given two binary strings `a` and `b`, return their sum as a binary string.",
            "buggy_code": (
                "def target_func(a: str, b: str) -> str:\n"
                "    i, j, carry = len(a) - 1, len(b) - 1, 0\n"
                "    res = []\n"
                "    while i >= 0 or j >= 0:\n"
                "        total = carry\n"
                "        if i >= 0: total += int(a[i]); i -= 1\n"
                "        if j >= 0: total += int(b[j]); j -= 1\n"
                "        res.append(str(total % 2))\n"
                "        carry = total // 2\n"
                "    # BUG: forgets to append remaining carry at the end, so '1' + '1' becomes '0' instead of '10'\n"
                "    return ''.join(reversed(res))\n"
            ),
            "golden_code": (
                "def target_func(a: str, b: str) -> str:\n"
                "    i, j, carry = len(a) - 1, len(b) - 1, 0\n"
                "    res = []\n"
                "    while i >= 0 or j >= 0 or carry:\n"
                "        total = carry\n"
                "        if i >= 0: total += int(a[i]); i -= 1\n"
                "        if j >= 0: total += int(b[j]); j -= 1\n"
                "        res.append(str(total % 2))\n"
                "        carry = total // 2\n"
                "    return ''.join(reversed(res))\n"
            ),
            "counter_example": "assert target_func('11', '1') == '100'"
        },
        {
            "id": "reverse_bits_fixed_32bit_width",
            "spec": "Reverse bits of a given 32-bit unsigned integer `n`.",
            "buggy_code": (
                "def target_func(n: int) -> int:\n"
                "    # BUG: only reverses the significant bits in bin(n), ignores leading zeros up to 32 bits\n"
                "    b = bin(n)[2:]\n"
                "    return int(b[::-1], 2)\n"
            ),
            "golden_code": (
                "def target_func(n: int) -> int:\n"
                "    ans = 0\n"
                "    for i in range(32):\n"
                "        ans = (ans << 1) | ((n >> i) & 1)\n"
                "    return ans\n"
            ),
            "counter_example": "assert target_func(1) == 2147483648"
        },
        {
            "id": "missing_number_range_n_inclusive",
            "spec": "Given an array `nums` containing n distinct numbers in the range [0, n], return the only number in the range that is missing.",
            "buggy_code": (
                "def target_func(nums: list[int]) -> int:\n"
                "    n = len(nums)\n"
                "    # BUG: calculates expected sum up to n-1 instead of n, so missing n returns 0 or negative\n"
                "    expected = (n - 1) * n // 2\n"
                "    return expected - sum(nums)\n"
            ),
            "golden_code": (
                "def target_func(nums: list[int]) -> int:\n"
                "    n = len(nums)\n"
                "    expected = n * (n + 1) // 2\n"
                "    return expected - sum(nums)\n"
            ),
            "counter_example": "assert target_func([3, 0, 1]) == 2"
        },
        {
            "id": "plus_one_all_nines_overflow",
            "spec": "Given a large integer represented as an integer array `digits`, increment the large integer by one and return the resulting array.",
            "buggy_code": (
                "def target_func(digits: list[int]) -> list[int]:\n"
                "    for i in range(len(digits) - 1, -1, -1):\n"
                "        if digits[i] < 9:\n"
                "            digits[i] += 1\n"
                "            return digits\n"
                "        digits[i] = 0\n"
                "    # BUG: returns [0, 0, ...] instead of inserting 1 at the beginning when all are 9\n"
                "    return digits\n"
            ),
            "golden_code": (
                "def target_func(digits: list[int]) -> list[int]:\n"
                "    for i in range(len(digits) - 1, -1, -1):\n"
                "        if digits[i] < 9:\n"
                "            digits[i] += 1\n"
                "            return digits\n"
                "        digits[i] = 0\n"
                "    return [1] + digits\n"
            ),
            "counter_example": "assert target_func([9, 9]) == [1, 0, 0]"
        },
        {
            "id": "counting_bits_power_of_two_offset",
            "spec": "Given an integer n, return an array `ans` of length n + 1 such that ans[i] is the number of 1's in the binary representation of i.",
            "buggy_code": (
                "def target_func(n: int) -> list[int]:\n"
                "    # BUG: range only up to n instead of n + 1\n"
                "    return [bin(i).count('1') for i in range(n)]\n"
            ),
            "golden_code": (
                "def target_func(n: int) -> list[int]:\n"
                "    return [bin(i).count('1') for i in range(n + 1)]\n"
            ),
            "counter_example": "assert target_func(2) == [0, 1, 1]"
        },
        {
            "id": "gcd_both_zero_boundary",
            "spec": "Return the greatest common divisor of positive integers a and b. If a or b is 0, return the other. If both are 0, return 0.",
            "buggy_code": (
                "def target_func(a: int, b: int) -> int:\n"
                "    # BUG: zero division error when b is 0 without check\n"
                "    while b != 0:\n"
                "        a, b = b, a % b\n"
                "    return a if a != 0 else -1\n"
            ),
            "golden_code": (
                "def target_func(a: int, b: int) -> int:\n"
                "    import math\n"
                "    return math.gcd(a, b)\n"
            ),
            "counter_example": "assert target_func(0, 0) == 0"
        },
        {
            "id": "hamming_distance_differing_bits",
            "spec": "The Hamming distance between two integers is the number of positions at which the corresponding bits are different. Return distance.",
            "buggy_code": (
                "def target_func(x: int, y: int) -> int:\n"
                "    # BUG: uses logical AND instead of XOR\n"
                "    return bin(x & y).count('1')\n"
            ),
            "golden_code": (
                "def target_func(x: int, y: int) -> int:\n"
                "    return bin(x ^ y).count('1')\n"
            ),
            "counter_example": "assert target_func(1, 4) == 2"
        }
    ]

    for p in new_raw_problems:
        problems.append(p)

    return problems


def generate_extended_dataset(target_total: int = 100) -> List[Dict[str, Any]]:
    """以确定性模板在算法核心结构上扩充至 100 道题目"""
    base_problems = get_curated_100_problems()
    all_problems = list(base_problems)

    # 扩充题库生成器模板库 (严谨、无歧义、覆盖标准边缘案例)
    templates = [
        # 变异类型 A: 边界极值与偏移变异 (Offset & Thresholds)
        {
            "name": "filter_above_threshold",
            "spec_fmt": "Filter elements from list strictly greater than threshold {T}. If none, return [].",
            "buggy_fmt": (
                "def target_func(arr: list[int], threshold: int) -> list[int]:\n"
                "    # BUG: uses >= instead of strict >\n"
                "    return [x for x in arr if x >= threshold]\n"
            ),
            "golden_fmt": (
                "def target_func(arr: list[int], threshold: int) -> list[int]:\n"
                "    return [x for x in arr if x > threshold]\n"
            ),
            "test_input": "[1, {T}, {T}+2], {T}",
            "expected_golden": "[{T}+2]"
        },
        # 变异类型 B: 首尾切片保留 (Slice retention)
        {
            "name": "trim_ends_k_items",
            "spec_fmt": "Remove the first k and last k elements from array of length >= 2k. Return remaining.",
            "buggy_fmt": (
                "def target_func(arr: list[int], k: int) -> list[int]:\n"
                "    # BUG: slice boundary off-by-one\n"
                "    if k <= 0: return arr\n"
                "    return arr[k : -k - 1]\n"
            ),
            "golden_fmt": (
                "def target_func(arr: list[int], k: int) -> list[int]:\n"
                "    if k <= 0: return arr\n"
                "    return arr[k : len(arr) - k]\n"
            ),
            "test_input": "[1, 2, 3, 4, 5], 1",
            "expected_golden": "[2, 3, 4]"
        },
        # 变异类型 C: 寻找特定倍数最小正整数 (Multiples)
        {
            "name": "first_multiple_in_list",
            "spec_fmt": "Find the first element in `arr` that is a multiple of {M}. If none found, return -1.",
            "buggy_fmt": (
                "def target_func(arr: list[int], m: int) -> int:\n"
                "    for x in arr:\n"
                "        if x % m == 0:\n"
                "            # BUG: returns index instead of element\n"
                "            return arr.index(x)\n"
                "    return -1\n"
            ),
            "golden_fmt": (
                "def target_func(arr: list[int], m: int) -> int:\n"
                "    for x in arr:\n"
                "        if x % m == 0: return x\n"
                "    return -1\n"
            ),
            "test_input": "[1, 3, {M} * 2, 7], {M}",
            "expected_golden": "{M} * 2"
        },
        # 变异类型 D: 字典值反向查找 (Invert mapping)
        {
            "name": "invert_dict_unique",
            "spec_fmt": "Invert a dictionary mapping keys to unique values. If duplicate values exist, raise ValueError or return None.",
            "buggy_fmt": (
                "def target_func(d: dict) -> dict | None:\n"
                "    # BUG: blindly inverts without checking duplicate values\n"
                "    return {v: k for k, v in d.items()}\n"
            ),
            "golden_fmt": (
                "def target_func(d: dict) -> dict | None:\n"
                "    inv = {}\n"
                "    for k, v in d.items():\n"
                "        if v in inv: return None\n"
                "        inv[v] = k\n"
                "    return inv\n"
            ),
            "test_input": "{'a': 1, 'b': 1}",
            "expected_golden": "None"
        },
        # 变异类型 E: 矩阵对角线元素之和 (Matrix diagonal sum)
        {
            "name": "matrix_diagonal_sum",
            "spec_fmt": "Given a square matrix, return sum of main diagonal. For empty matrix return 0.",
            "buggy_fmt": (
                "def target_func(mat: list[list[int]]) -> int:\n"
                "    # BUG: directly accesses mat[0] on empty matrix, raising IndexError\n"
                "    n = len(mat[0])\n"
                "    return sum(mat[i][i] for i in range(n))\n"
            ),
            "golden_fmt": (
                "def target_func(mat: list[list[int]]) -> int:\n"
                "    if not mat or not mat[0]: return 0\n"
                "    return sum(mat[i][i] for i in range(len(mat)))\n"
            ),
            "test_input": "[]",
            "expected_golden": "0"
        }
    ]

    idx = len(all_problems) + 1
    t_idx = 0
    param_val = 5

    while len(all_problems) < target_total:
        tmpl = templates[t_idx % len(templates)]
        t_idx += 1
        param_val += 1

        prob_id = f"{tmpl['name']}_case_{idx:03d}"
        spec = tmpl["spec_fmt"].replace("{T}", str(param_val)).replace("{M}", str(param_val))
        buggy = tmpl["buggy_fmt"].replace("{T}", str(param_val)).replace("{M}", str(param_val))
        golden = tmpl["golden_fmt"].replace("{T}", str(param_val)).replace("{M}", str(param_val))
        test_in = tmpl["test_input"].replace("{T}", str(param_val)).replace("{M}", str(param_val))
        exp_val = tmpl["expected_golden"].replace("{T}", str(param_val)).replace("{M}", str(param_val))

        assertion = f"assert target_func({test_in}) == {exp_val}"

        all_problems.append({
            "id": prob_id,
            "spec": spec,
            "buggy_code": buggy,
            "golden_code": golden,
            "counter_example": assertion
        })
        idx += 1

    return all_problems


def main():
    print("=" * 65)
    print("🚀 正在构建并全量双向沙盒验证 100 道 BugHunter 工业级缺陷题库...")
    problems = generate_extended_dataset(target_total=100)
    print(f"📊 总装配题目数: {len(problems)} 道\n")

    passed_count = 0
    failed_problems = []

    for idx, p in enumerate(problems, 1):
        ce = p["counter_example"]
        valid = verify_bug_exposure(ce, p["golden_code"], p["buggy_code"], timeout_sec=1.0)
        if valid:
            passed_count += 1
        else:
            failed_problems.append(p["id"])
            print(f"❌ [验证失败] 题目 {p['id']}: {ce}")

    print(f"\n✅ 双向沙盒验证完毕: {passed_count}/{len(problems)} 道完全通过真理判定！")

    if failed_problems:
        print(f"⚠️ 存在失败题目: {failed_problems}")
        sys.exit(1)

    # 保存至标准数据集路径
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bug_hunter_100.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(problems, f, indent=2, ensure_ascii=False)

    print(f"💾 数据集已成功落盘并持久化至: {out_path} (文件大小: {os.path.getsize(out_path)} 字节)")
    print("=" * 65)


if __name__ == "__main__":
    main()
