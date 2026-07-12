"""Make the repo root importable so tests can `import experiments...` / `import src...`
regardless of pytest's import mode or the working directory."""
import asyncio
import inspect
import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve()
while _root != _root.parent and not (_root / "pyproject.toml").exists():
    _root = _root.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    """Run `async def` tests in a fresh event loop (mirrors the pytest-async plugin, which
    isn't auto-registered here). Sync tests fall through to pytest's default handling."""
    func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(func):
        return None
    kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
    asyncio.run(func(**kwargs))
    return True
