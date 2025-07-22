from pathlib import Path


def api_key_from_file(path: str, do_raise: bool = False) -> str:
    """
    Read an API key from a file.

    Args:
        path (str): Path to the file containing the API key.

    Returns:
        str: The API key.

    Raises:
        FileNotFoundError: If the file is not found.
    """
    key_file = Path(path)
    if key_file.exists():
        with key_file.open("r", encoding="utf-8") as f:
            return f.read().strip()
    else:
        if do_raise:
            raise FileNotFoundError("API key file not found")
        return ""
