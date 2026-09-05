"""Minimal test runner (the project venv has no pytest and the box has no internet).

Usage: python tests/run_tests.py [test_file_or_name_substring ...]
Supports plain test_* functions and @pytest.mark.parametrize("name", [values]) with or
without a real pytest installed.
"""

import importlib.util
import os
import sys
import time
import traceback
import types

HERE = os.path.dirname(os.path.abspath(__file__))


def _install_fake_pytest() -> None:
    try:
        import pytest  # noqa: F401

        return
    except ImportError:
        pass
    fake = types.ModuleType("pytest")

    class _Mark:
        @staticmethod
        def parametrize(names, values):
            def deco(fn):
                fn._parametrize = (names, values)
                return fn

            return deco

    def approx(x, rel=1e-6, abs=1e-9):  # noqa: A002
        class _A:
            def __eq__(self, other):
                return abs(other - x) <= max(abs, rel * abs(x))

        return _A()

    fake.mark = _Mark()
    fake.approx = approx
    fake.main = lambda args=None: 0
    sys.modules["pytest"] = fake


def _load(path: str):
    spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3], path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main(argv: list[str]) -> int:
    _install_fake_pytest()
    files = sorted(
        os.path.join(HERE, f)
        for f in os.listdir(HERE)
        if f.startswith("test_") and f.endswith(".py")
    )
    if argv:
        files = [f for f in files if any(a in f for a in argv)]
    passed = failed = 0
    for path in files:
        module = _load(path)
        for name in dir(module):
            if not name.startswith("test_"):
                continue
            fn = getattr(module, name)
            if not callable(fn):
                continue
            params = getattr(fn, "_parametrize", None)
            if params is None:  # real pytest installed: read its mark instead
                for mark in getattr(fn, "pytestmark", []):
                    if getattr(mark, "name", "") == "parametrize":
                        params = (mark.args[0], mark.args[1])
            cases = [((), "")]
            if params is not None:
                names, values = params
                cases = [((v,) if not isinstance(v, tuple) else v, f"[{v}]") for v in values]
            for args, label in cases:
                t0 = time.perf_counter()
                try:
                    fn(*args)
                    passed += 1
                    print(f"PASS {os.path.basename(path)}::{name}{label} ({time.perf_counter() - t0:.2f}s)")
                except Exception:
                    failed += 1
                    print(f"FAIL {os.path.basename(path)}::{name}{label}")
                    traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
