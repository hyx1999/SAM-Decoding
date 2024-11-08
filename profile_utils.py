from typing import List, Dict, Tuple
from collections import defaultdict
from functools import wraps
import time
import pandas as pd

fn_dict: Dict[str, List[float]] = defaultdict(lambda: list())

decorator_flag: bool = False

def enable_decorator(mode: bool):
    global decorator_flag
    decorator_flag = mode

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

def export_result(root_name: str = "SamdModel.decode"):
    result = []
    if len(fn_dict) == 0:
        return None
    for name, values in fn_dict.items():
        result.append((
            name, sum(values) / len(values)
        ))
    sum_time = dict(result)[root_name]
    df = pd.DataFrame(result, columns=["name", "time"])
    df["ratio"] = df["time"] / sum_time
    return df.to_string()
