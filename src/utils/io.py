from pydantic import BaseModel
from pathlib import Path
from typing import TypeVar, overload, Literal
import json
from src.utils.logging import create_logger


logger = create_logger(__name__)


def save_base_model(
    model: BaseModel,
    path: str | Path,
    overwrite: bool = False,
    **kwargs,
) -> None:
    """
    Save a Pydantic model to a JSON file.
    If containing folder does not exist, it will be created.
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.parent.exists():
        path.parent.mkdir(parents=True)
        logger.info(f"Created parent directory: {path.parent}")

    if path.exists() and not overwrite:
        raise FileExistsError(f"File '{path}' already exists.")

    path.write_text(model.model_dump_json(indent=4, **kwargs), encoding="utf-8")
    logger.info(f"Saved {type(model).__name__} to '{path}'.")


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
    Load a Pydantic model from a JSON file.
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        if not do_raise:
            logger.warning(f"Model file '{path}' does not exist.")
            return None
        raise FileNotFoundError(f"Model file '{path}' does not exist.")

    try:
        data = path.read_text(encoding="utf-8")
        model = model_class.model_validate_json(data, **kwargs)
        logger.info(f"Loaded {type(model).__name__} from '{path}'.")
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
    Save an object to a JSON file.
    If containing folder does not exist, it will be created.
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.parent.exists():
        path.parent.mkdir(parents=True)
        logger.info(f"Created parent directory: {path.parent}")

    if path.exists() and not overwrite:
        raise FileExistsError(f"File '{path}' already exists.")

    path.write_text(json.dumps(obj, indent=4, **kwargs), encoding="utf-8")
    logger.info(f"Saved {type(obj).__name__} to '{path}'.")


def load_object(
    path: str | Path,
    do_raise: bool = True,
    **kwargs,
) -> object:
    """
    Load an object from a JSON file.
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        if not do_raise:
            logger.warning(f"Object file '{path}' does not exist.")
            return None
        raise FileNotFoundError(f"Object file '{path}' does not exist.")

    try:
        obj = json.loads(path.read_text(encoding="utf-8"), **kwargs)
        logger.info(f"Loaded {type(obj).__name__} from '{path}'.")
        return obj
    except Exception as e:
        if do_raise:
            raise e
        else:
            logger.error(f"Failed to load object from '{path}': {e}")
            return None
