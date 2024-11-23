from typing import List, Dict, Tuple
from collections import defaultdict
from functools import wraps
import time
import pandas as pd

fn_dict: Dict[str, List[float]] = defaultdict(lambda: list())
lookup_dict: Dict[str, List[str]] = defaultdict(lambda: list())
accept_dict: Dict[str, List[int]] = defaultdict(lambda: list())

decorator_flag: bool = False

def enable_decorator(mode: bool):
    global decorator_flag
    decorator_flag = mode

def clear_dict():
    fn_dict.clear()
    lookup_dict.clear()

def profile_decorator(fn_name: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if decorator_flag:
                start_time = time.perf_counter()
                result = fn(*args, **kwargs)
                end_time = time.perf_counter()
                fn_dict[fn_name].append(end_time - start_time)
            else:
                result = fn(*args, **kwargs)
            return result
        return wrapper
    return decorator


def profile_lookup_decorator(fn_name: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if decorator_flag:
                result = fn(*args, **kwargs)
                lookup_dict[fn_name].append(result[0])
            else:
                result = fn(*args, **kwargs)
            return result
        return wrapper
    return decorator

def profile_accept_length(name: str, length: int):
    if decorator_flag:
        accept_dict[name].append(length)

def export_result(root_name: str = "forward"):
    result = []
    if len(fn_dict) == 0:
        return None
    for name, value in fn_dict.items():
        print("name: {}, len(value): {}".format(name, len(value)))
        result.append((
            name, sum(value)
        ))
    result_dict = dict(result)
    sum_time = result_dict.get(root_name, max(result_dict.values()))
    if sum_time is None:
        sum_time = max(result_dict.values())
    df = pd.DataFrame(result, columns=["name", "time"])
    df["ratio"] = df["time"] / sum_time
    return df.to_string()

def export_lookup_result():
    import json
    result1 = {}
    result2 = {}
    for name, type_names in lookup_dict.items():
        result1[name] = {}
        result2[name] = {}
        for type_name in type_names:
            if type_name not in result1[name]:
                result1[name][type_name] = 0
            result1[name][type_name] += 1
        for type_name, length in zip(type_names, accept_dict[name]):
            if type_name not in result2[name]:
                result2[name][type_name] = 0
            result2[name][type_name] += length
        for key in result1[name].keys():
            result2[name][key] /= result1[name][key]
    return json.dumps({"result-1": result1, "result-2": result2}, indent=4, ensure_ascii=False)
