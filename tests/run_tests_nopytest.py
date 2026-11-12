"""
Fallback test runner for environments without pytest.

    python tests/run_tests_nopytest.py

Implements just enough of the pytest API (approx, skip, raises, module-scoped
fixtures) to execute tests/test_pipeline.py unchanged. Use real pytest when
available; this exists so the suite is runnable on a bare Python install.
"""
from __future__ import annotations

import sys
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "unit"))


class _Skip(Exception):
    pass


class _Approx:
    def __init__(self, value, abs=None, rel=None):
        self.value, self.abs, self.rel = value, abs, rel

    def __eq__(self, other):
        tol = self.abs if self.abs is not None else max(
            abs(self.value) * (self.rel or 1e-6), 1e-9
        )
        return abs(other - self.value) <= tol

    def __repr__(self):
        return f"approx({self.value})"


def _make_pytest_shim() -> types.ModuleType:
    m = types.ModuleType("pytest")
    m.approx = lambda v, abs=None, rel=None: _Approx(v, abs, rel)
    m.skip = lambda reason="": (_ for _ in ()).throw(_Skip(reason))
    m.Skipped = _Skip

    def fixture(*a, **kw):
        def deco(fn):
            fn.__is_fixture__ = True
            return fn
        return deco(a[0]) if a and callable(a[0]) else deco

    m.fixture = fixture
    return m


sys.modules["pytest"] = _make_pytest_shim()

import test_evaluation as E  # noqa: E402
import test_pipeline as T  # noqa: E402


def main() -> int:
    # Resolve module-scoped fixtures once, in dependency order.
    cache = {}
    cache["city"] = T.city()
    cache["healed"] = T.healed(cache["city"])

    e_city = E.city()
    e_cache = {"city": e_city, "graphs": E.graphs(e_city)}

    classes = [(getattr(T, n), cache) for n in dir(T)
               if n.startswith("Test") and isinstance(getattr(T, n), type)]
    classes += [(getattr(E, n), e_cache) for n in dir(E)
                if n.startswith("Test") and isinstance(getattr(E, n), type)]

    passed = failed = skipped = 0
    failures = []
    for cls, fixtures in classes:
        inst = cls()
        for name in sorted(n for n in dir(cls) if n.startswith("test_")):
            fn = getattr(inst, name)
            args = [fixtures[p] for p in fn.__code__.co_varnames[1 : fn.__code__.co_argcount]
                    if p in fixtures]
            label = f"{cls.__name__}.{name}"
            try:
                fn(*args)
                passed += 1
                print(f"  PASS  {label}")
            except _Skip as e:
                skipped += 1
                print(f"  SKIP  {label} ({e})")
            except Exception:
                failed += 1
                failures.append((label, traceback.format_exc()))
                print(f"  FAIL  {label}")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    for label, tb in failures:
        print(f"\n--- {label} ---\n{tb}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
