from enum import StrEnum


class Backend(StrEnum):
    ART = "art"
    VERL = "verl"


def auto_backend() -> Backend:

    has_art = False
    has_verl = False

    try:
        import art  # noqa: F401

        has_art = True
    except ImportError:
        pass

    try:
        import verl  # noqa: F401

        has_verl = True
    except ImportError:
        pass

    if has_art and has_verl:
        raise RuntimeError("Both ART and VERL backends are available. Please specify which one to use.")

    elif has_art and not has_verl:
        return Backend.ART

    elif has_verl and not has_art:
        return Backend.VERL

    raise RuntimeError("Neither ART nor VERL backends are available. Please install one of them to proceed.")
