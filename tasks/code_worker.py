"""Restricted function worker. Test assertions and expected values stay in the parent.

This supports algorithmic Python functions, not arbitrary Python applications.
Imports, builtins and reflective attributes are restricted before execution.
"""
import ast
import builtins
import contextlib
import importlib
import json
import resource
import sys
import types


MODULES = {
    "math": None,
    "collections": {"Counter", "defaultdict", "deque", "OrderedDict"},
    "itertools": {"accumulate", "chain", "combinations", "permutations", "product", "groupby", "islice", "zip_longest"},
    "functools": {"lru_cache", "cache", "reduce", "cmp_to_key"},
    "bisect": {"bisect", "bisect_left", "bisect_right", "insort", "insort_left", "insort_right"},
    "heapq": {"heappush", "heappop", "heapify", "heappushpop", "heapreplace", "nlargest", "nsmallest", "merge"},
    "re": {"match", "fullmatch", "search", "findall", "finditer", "split", "sub", "compile", "escape", "IGNORECASE"},
}
BUILTINS = {
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
    "frozenset", "int", "isinstance", "issubclass", "len", "list", "map", "max", "min",
    "next", "ord", "chr", "pow", "print", "range", "reversed", "round", "set", "slice",
    "sorted", "str", "sum", "tuple", "zip", "Exception", "ValueError", "TypeError",
    "IndexError", "KeyError", "RuntimeError", "ZeroDivisionError", "StopIteration",
}
FORBIDDEN_ATTRIBUTES = {
    "format", "format_map", "mro", "gi_frame", "gi_code", "cr_frame", "cr_code",
    "ag_frame", "ag_code", "f_back", "f_globals", "f_locals", "f_builtins", "f_code",
    "tb_frame", "tb_next", "co_consts", "co_code",
}


def validate_source(source):
    tree = ast.parse(source)
    if len(source) > 100_000 or sum(1 for _ in ast.walk(tree)) > 10_000:
        raise ValueError("Source too large")
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.YieldFrom)):
            raise ValueError("Unsupported syntax")
        if isinstance(node, ast.Name) and ((node.id.startswith("_") and node.id != "_") or node.id in {
            "exec", "eval", "compile", "open", "input", "globals", "locals", "vars",
            "getattr", "setattr", "delattr", "hasattr", "type", "object", "breakpoint",
            "help", "exit", "quit", "SystemExit", "BaseException", "memoryview",
        }):
            raise ValueError("Reflective or system name")
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRIBUTES):
            raise ValueError("Reflective attribute")
        if isinstance(node, ast.Import):
            if any(alias.name not in MODULES for alias in node.names):
                raise ValueError("Unsupported import")
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in MODULES:
                raise ValueError("Unsupported import")
            exports = MODULES[node.module]
            if any(a.name.startswith("_") or a.name == "*" or (exports is not None and a.name not in exports) for a in node.names):
                raise ValueError("Unsupported export")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if any(a.asname and a.asname.startswith("_") for a in node.names):
                raise ValueError("Private import alias")
    return tree


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in MODULES:
        raise ImportError(name)
    module = importlib.import_module(name)
    exports = MODULES[name]
    names = exports if exports is not None else {n for n in dir(module) if not n.startswith("_")}
    return types.SimpleNamespace(**{n: getattr(module, n) for n in names})


def plain_value(value, depth=0):
    if depth > 40:
        raise ValueError("Result too deep")
    if type(value) in (type(None), bool, int, float, str):
        return
    if type(value) in (list, tuple, set):
        for item in value:
            plain_value(item, depth + 1)
    elif type(value) is dict:
        for key, item in value.items():
            plain_value(key, depth + 1)
            plain_value(item, depth + 1)
    else:
        raise ValueError("Return only builtin literal values")


def main():
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    payload = json.loads(sys.stdin.read(1_000_000))
    tree = validate_source(payload["source"])
    scope = {"__builtins__": {n: getattr(builtins, n) for n in BUILTINS}}
    scope["__builtins__"]["__import__"] = safe_import
    # User output never forms the result protocol.
    with open("/dev/null", "w") as sink, contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        exec(compile(tree, "<solution>", "exec"), scope)
        solution = scope.get("solution")
        if not callable(solution):
            raise ValueError("Missing solution function")
        results = []
        for call in payload["calls"]:
            args, kwargs = ast.literal_eval(call)
            value = solution(*args, **kwargs)
            plain_value(value)
            results.append(repr(value))
    response = json.dumps(results)
    if len(response) > 1_000_000:
        raise ValueError("Result too large")
    sys.stdout.write(response)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        sys.exit(1)
