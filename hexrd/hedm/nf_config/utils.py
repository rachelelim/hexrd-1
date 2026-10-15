import copy


def merge_dicts(a, b):
    """Return a merged dict, updating values from `a` with values from `b`."""
    # need to pass a deep copy of a at the top level only:
    return _merge_dicts(copy.deepcopy(a), b)


def _merge_dicts(a, b):
    for k, v in b.items():
        if isinstance(v, dict):
            if a.get(k) is None:
                # happens in cases where all but section head is commented
                a[k] = {}
            _merge_dicts(a[k], v)
        else:
            if v is None and a.get(k) is not None:
                # entire section commented out. Inherit, don't overwrite
                pass
            else:
                a[k] = v
    return a


def none_strings_to_null(obj):
    """Recursively replace the string 'None' with None.

    The yaml SafeLoader reads a bare `None` as a string, but the input files
    use it to mean null.
    """
    if isinstance(obj, dict):
        return {k: none_strings_to_null(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [none_strings_to_null(v) for v in obj]
    return None if isinstance(obj, str) and obj == 'None' else obj


class Null():
    pass


null = Null()
