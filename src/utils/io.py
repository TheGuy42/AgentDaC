from pydantic import BaseModel
from pathlib import Path
from typing import TypeVar, overload, Literal
import json
import yaml
from src.utils.logging import create_logger


logger = create_logger(__name__)


YAML_SUFFIXES = frozenset({".yaml", ".yml"})
JSON_SUFFIXES = frozenset({".json"})


def is_yaml(path: str | Path) -> bool:
    """Whether the path is a YAML file, decided by its suffix."""
    return Path(path).suffix.lower() in YAML_SUFFIXES


def is_json(path: str | Path) -> bool:
    """Whether the path is a JSON file, decided by its suffix."""
    return Path(path).suffix.lower() in JSON_SUFFIXES


def dumps(obj: object, path: str | Path, **kwargs) -> str:
    """Serialize `obj` in the format implied by `path`."""
    if is_yaml(path):
        return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True, indent=4, **kwargs)
    if is_json(path):
        return json.dumps(obj, indent=4, **kwargs)
    raise ValueError(f"Unsupported file format '{Path(path).suffix}' for '{path}'.")


def loads(text: str, path: str | Path) -> object:
    """Deserialize `text` in the format implied by `path`."""
    if is_yaml(path):
        return yaml.safe_load(text)
    if is_json(path):
        return json.loads(text)
    raise ValueError(f"Unsupported file format '{Path(path).suffix}' for '{path}'.")


def save_base_model(
    model: BaseModel,
    path: str | Path,
    overwrite: bool = False,
    **kwargs,
) -> None:
    """
    Save a Pydantic model to a YAML or JSON file, chosen by the file suffix.
    If containing folder does not exist, it will be created.
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.parent.exists():
        path.parent.mkdir(parents=True)
        logger.debug(f"Created parent directory: {path.parent}")

    if path.exists() and not overwrite:
        raise FileExistsError(f"File '{path}' already exists.")

    path.write_text(dumps(model.model_dump(mode="json", **kwargs), path), encoding="utf-8")
    logger.debug(f"Saved {type(model).__name__} to '{path}'.")


T = TypeVar("T", bound=BaseModel)


@overload
def load_base_model(
    model_class: type[T],
    path: str | Path,
    do_raise: Literal[True],
    **kwargs,
) -> T: ...


@overload
def load_base_model(
    model_class: type[T],
    path: str | Path,
    do_raise: Literal[False],
    **kwargs,
) -> T | None: ...


def load_base_model(
    model_class: type[T],
    path: str | Path,
    do_raise: bool = True,
    **kwargs,
) -> T | None:
    """
    Load a Pydantic model from a YAML or JSON file, chosen by the file suffix.
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        if not do_raise:
            logger.warning(f"Model file '{path}' does not exist.")
            return None
        raise FileNotFoundError(f"Model file '{path}' does not exist.")

    try:
        data = loads(path.read_text(encoding="utf-8"), path)
        model = model_class.model_validate(data, **kwargs)
        logger.debug(f"Loaded {type(model).__name__} from '{path}'.")
        return model
    except Exception as e:
        if do_raise:
            raise e
        else:
            logger.error(f"Failed to load model from '{path}': {e}")
            return None


def save_object(
    obj: object,
    path: str | Path,
    overwrite: bool = False,
    **kwargs,
) -> None:
    """
    Save an object to a YAML or JSON file, chosen by the file suffix.
    If containing folder does not exist, it will be created.
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.parent.exists():
        path.parent.mkdir(parents=True)
        logger.debug(f"Created parent directory: {path.parent}")

    if path.exists() and not overwrite:
        raise FileExistsError(f"File '{path}' already exists.")

    path.write_text(dumps(obj, path, **kwargs), encoding="utf-8")
    logger.debug(f"Saved {type(obj).__name__} to '{path}'.")


def load_object(
    path: str | Path,
    do_raise: bool = True,
    **kwargs,
) -> object:
    """
    Load an object from a YAML or JSON file, chosen by the file suffix.
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        if not do_raise:
            logger.warning(f"Object file '{path}' does not exist.")
            return None
        raise FileNotFoundError(f"Object file '{path}' does not exist.")

    try:
        obj = loads(path.read_text(encoding="utf-8"), path)
        logger.debug(f"Loaded {type(obj).__name__} from '{path}'.")
        return obj
    except Exception as e:
        if do_raise:
            raise e
        else:
            logger.error(f"Failed to load object from '{path}': {e}")
            return None
