from typing import Any, Tuple


def get_dict_value(dicts: dict[str, Any], *keys: str, raise_missing: bool = True) -> Tuple[Any, bool]:
    """
    Recursively get a value from a nested dictionary using a sequence of keys.
    Creates intermediate dictionaries if they do not exist.

    Args:
        dicts (dict): The dictionary to search through.
        *keys (str): A sequence of keys representing the path to the desired value.
        raise_missing (bool): Whether to raise an error if the key is not found.

    Returns:
        tuple[value, found]: A tuple where 'value' is the retrieved value (or None if not found) and 'found' is a boolean indicating whether the value was found.
    """
    assert len(keys) > 0, "At least one key must be provided"

    current = dicts
    for i, key in enumerate(keys[:-1]):
        if not isinstance(current, dict):
            raise ValueError(f"Expected a dictionary at key path {'->'.join(keys[:i])}, but got {type(current).__name__}")
        current = current.setdefault(key, {} if not raise_missing else None)
        if current is None:
            raise ValueError(f"Key not found: {'->'.join(keys[: i + 1])}")

    last_key = keys[-1]
    if last_key not in current:
        if raise_missing:
            raise ValueError(f"Key not found: {'->'.join(keys)}")
        return None, False

    return current[last_key], True


def set_dict_value(
    dicts: dict[str, Any],
    *keys: str,
    value: Any,
    set_default: bool = False,
    raise_missing: bool = False,
) -> Any:
    """
    Recursively set a value in a nested dictionary using a sequence of keys.
    Creates intermediate dictionaries if they do not exist.

    Args:
        dicts (dict): The dictionary to modify.
        *keys (str): A sequence of keys representing the path where the value should be set.
        value (Any): The value to set at the specified key path.
        set_default (bool): If True, only set the value if the key path does not already exist.
        raise_missing (bool): Whether to raise an error if an intermediate key is not found.
    """
    assert len(keys) > 0, "At least one key must be provided"

    current = dicts
    for i, key in enumerate(keys[:-1]):
        if not isinstance(current, dict):
            raise ValueError(f"Expected a dictionary at key path {'->'.join(keys[:i])}, but got {type(current).__name__}")
        current = current.setdefault(key, {} if not raise_missing else None)
        if current is None:
            raise ValueError(f"Key not found: {'->'.join(keys[: i + 1])}")

    last_key = keys[-1]
    if not set_default or (last_key not in current):
        current[last_key] = value

    return current[last_key] 