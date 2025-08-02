from pydantic import BaseModel
from pathlib import Path
from typing import TypeVar, overload, Literal
import json

from src.utils.logging import create_logger


logger = create_logger(__name__)


def save_base_model(
    model: BaseModel,
    path: str | Path,
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

    path.write_text(
        model.model_dump_json(indent=4, **kwargs),
        encoding="utf-8",
    )


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
        return model_class.model_validate_json(data, **kwargs)
    except Exception as e:
        if do_raise:
            raise e
        else:
            logger.error(f"Failed to load model from '{path}': {e}")
            return None


def save_object(
    obj: object,
    path: str | Path,
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

    path.write_text(
        json.dumps(obj, indent=4, **kwargs),
        encoding="utf-8",
    )


def load_object(
    path: str | Path,
    **kwargs,
) -> object:
    """
    Load an object from a JSON file.
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Object file '{path}' does not exist.")

    try:
        return json.loads(path.read_text(encoding="utf-8"), **kwargs)
    except Exception as e:
        raise e
