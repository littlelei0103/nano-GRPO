"""Typed JSON transport and comparison for builtin differential-test results.

Types are strict at every depth. Float tolerance is 1e-7 (relative and absolute),
including in containers; NaNs compare equal. Dict/set order is immaterial.
Custom objects and cyclic/deep containers are outside this result contract.
"""
import json
import math


def _encode(value, depth=0):
    if depth > 64:
        raise ValueError("Result exceeds maximum nesting depth")
    kind = type(value)
    if kind in (type(None), bool, int, str):
        return [kind.__name__, value]
    if kind is float:
        return ["float", value.hex()]
    if kind is bytes:
        return ["bytes", value.hex()]
    if kind in (list, tuple, set, frozenset):
        return [kind.__name__, [_encode(item, depth + 1) for item in value]]
    if kind is dict:
        return ["dict", [[_encode(k, depth + 1), _encode(v, depth + 1)]
                         for k, v in value.items()]]
    raise ValueError(f"Unsupported result type: {kind.__name__}")


def _decode(node, depth=0):
    if depth > 64 or type(node) is not list or len(node) != 2:
        raise ValueError("Invalid result payload")
    tag, payload = node
    scalar_types = {"NoneType": type(None), "bool": bool, "int": int, "str": str}
    if tag in scalar_types:
        if type(payload) is not scalar_types[tag]:
            raise ValueError("Invalid scalar payload")
        return payload
    if tag == "float" and type(payload) is str:
        return float.fromhex(payload)
    if tag == "bytes" and type(payload) is str:
        return bytes.fromhex(payload)
    if type(payload) is not list:
        raise ValueError("Invalid container payload")
    if tag in ("list", "tuple", "set", "frozenset"):
        factory = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}[tag]
        return factory(_decode(item, depth + 1) for item in payload)
    if tag == "dict":
        return {_decode(k, depth + 1): _decode(v, depth + 1) for k, v in payload}
    raise ValueError("Unknown result type tag")


def dumps_value(value):
    return json.dumps(_encode(value), allow_nan=False)


def loads_value(payload):
    return _decode(json.loads(payload))


def _unordered_equal(left, right, equal):
    """Find a bijection, including when float tolerance admits several matches."""
    if len(left) != len(right):
        return False
    matches = {}

    def augment(i, seen):
        for j, value in enumerate(right):
            if j not in seen and equal(left[i], value):
                seen.add(j)
                if j not in matches or augment(matches[j], seen):
                    matches[j] = i
                    return True
        return False

    return all(augment(i, set()) for i in range(len(left)))


def are_semantic_values_equal(left, right):
    if type(left) is not type(right):
        return False
    if type(left) is float:
        if math.isnan(left) or math.isnan(right):
            return math.isnan(left) and math.isnan(right)
        return math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-7)
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(
            are_semantic_values_equal(a, b) for a, b in zip(left, right))
    if type(left) is dict:
        return _unordered_equal(list(left.items()), list(right.items()),
                                are_semantic_values_equal)
    if type(left) in (set, frozenset):
        return _unordered_equal(list(left), list(right), are_semantic_values_equal)
    if type(left) in (type(None), bool, int, str, bytes):
        return left == right
    raise ValueError(f"Unsupported comparison type: {type(left).__name__}")
