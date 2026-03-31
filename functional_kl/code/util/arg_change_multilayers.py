import argparse
from copy import deepcopy

def flatten_dict(d, parent_key="", sep="."):

    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def set_by_dotted_key(d, dotted_key, value, sep="."):

    keys = dotted_key.split(sep)
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def get_by_dotted_key(d, dotted_key, sep="."):

    keys = dotted_key.split(sep)
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def add_config_to_argparser(config, parser: argparse.ArgumentParser, sep="."):

    flat = flatten_dict(config, sep=sep)

    for k, v in flat.items():
        arg_name = f"--{k}"

      
        if isinstance(v, bool):
         
            parser.add_argument(
                arg_name,
                dest=k,
                action="store_true" if v is False else "store_false",
                help=f"(default: {v})"
            )
        else:
            parser.add_argument(
                arg_name,
                dest=k,
                type=type(v),
                default=None,   
                help=f"(default: {v})"
            )

    return parser


def update_config_from_args(config, args, sep="."):

    config_before = deepcopy(config)
    updates = vars(args)  # { "a.b": value, ... }

    for dotted_k, updated_val in updates.items():
        if updated_val is None:
            continue  # 没传这个参数 -> 不覆盖

        orig_val = get_by_dotted_key(config, dotted_k, sep=sep)

        if updated_val != orig_val:
            print(f"Updated key '{dotted_k}': {orig_val} -> {updated_val}")
            set_by_dotted_key(config, dotted_k, updated_val, sep=sep)

    return config



