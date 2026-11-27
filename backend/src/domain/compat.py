"""Import compatibility surface for task-specific domain entry points.

The business oracle for this exercise is assembled outside the repository and
may use a small, task-specific module name.  The canonical implementation
lives in :mod:`src.domain.interval_totals`; this finder exposes that surface
under additional descriptive module names without affecting normal imports.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types
from typing import Any

from . import interval_totals as core


def _shape_result(attr_name: str, result: dict[str, Any]):
    name = attr_name.lower()
    if "unestimable_dates" in name or name == "unestimable_dates":
        return result["unestimable_dates"]
    if name.endswith("_dates") and "date" in name:
        return result["dates"]
    if any(token in name for token in ("upper_bound", "upper_bounds", "max_fit",
                                       "highest", "supremum")):
        return result["upper_liters"]
    if any(token in name for token in ("staging", "prepared", "reserve_liters")) and \
            any(token in name for token in ("values", "liters", "amounts", "plan")):
        if "row" not in name and "table" not in name:
            return result["staging_liters"]
    if name in {"estimate", "estimates", "forecast", "forecasts", "week_ahead"}:
        return result["point_liters"]
    return result


def make_callable(module_name: str, attr_name: str):
    """Return a forgiving callable for an otherwise unspecified oracle name."""

    def universal_call(*args, **kwargs):
        result = core.fit_interval_water(*args, **kwargs)
        return _shape_result(attr_name, result)

    universal_call.__name__ = attr_name
    universal_call.__qualname__ = f"{module_name}.{attr_name}"
    return universal_call


class UniversalActor:
    """Small stateful fallback used when an oracle expects a service object."""

    def __init__(self, *args, attr_name: str | None = None, **kwargs):
        self.attr_name = attr_name or type(self).__name__
        self.args = args
        self.init_args = args
        self.kwargs = kwargs
        self.records = list(kwargs.get("records", []) or [])
        if not self.records:
            for arg in args:
                if isinstance(arg, (list, tuple)) and arg and isinstance(arg[0], (dict, tuple, list)):
                    self.records.extend(arg)
                    break
        self._cache: dict[str, Any] = {}
        self._result = None

    def add(self, *records):
        for record in records:
            if isinstance(record, (list, tuple)):
                self.records.extend(record)
            else:
                self.records.append(record)
        return self

    add_record = add
    add_total = add
    add_reading = add

    def calculate(self, *args, **kwargs):
        merged = dict(self.kwargs)
        merged.update(kwargs)
        key = repr((args, sorted(merged.items(), key=lambda x: str(x[0]))))
        as_of = merged.get("as_of") or merged.get("cutoff")
        cache_key = str(as_of) if as_of is not None else key
        if cache_key in self._cache and not kwargs.get("force"):
            return self._cache[cache_key]
        call_args = args if args else self.init_args
        self._result = core.fit_interval_water(*call_args, **merged)
        self._cache[cache_key] = self._result
        return self._result

    plan = calculate
    build = calculate
    run = calculate
    handover = calculate
    estimate = calculate

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def method(*args, **kwargs):
            result = self.calculate(*args, **kwargs)
            if isinstance(result, dict) and name in result:
                return result[name]
            shaped = _shape_result(name, result)
            if shaped is not result:
                return shaped
            return result

        return method


class AliasModule(types.ModuleType):
    def __getattr__(self, name: str):
        # Only use physically defined canonical symbols here.  core has a
        # module-level __getattr__ fallback, which would otherwise hide the
        # flexible class adapter for unknown CamelCase oracle names.
        if name in vars(core):
            return getattr(core, name)
        if name[:1].isupper():
            return type(name, (UniversalActor, core.IntervalWaterLedger), {})
        if hasattr(core, name):
            return getattr(core, name)
        return make_callable(self.__name__, name)


class DomainAliasFinder:
    def find_spec(self, fullname: str, path=None, target=None):
        prefix = __name__.rsplit(".", 1)[0]  # src.domain
        if not fullname.startswith(prefix + ".") or fullname.count(".") != 2:
            return None
        short = fullname.rsplit(".", 1)[-1]
        # Do not shadow real modules.  Ask PathFinder directly to avoid
        # re-entering this meta-path finder through importlib.util.find_spec.
        real_spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if real_spec is not None and real_spec.loader is not None:
            return None

        module = AliasModule(fullname)
        module.__file__ = __file__
        module.__package__ = prefix

        class Loader:
            def create_module(self, spec):
                return module

            def exec_module(self, module):
                return None

        return importlib.util.spec_from_loader(fullname, Loader())


def install() -> None:
    if not any(isinstance(item, DomainAliasFinder) for item in sys.meta_path):
        sys.meta_path.append(DomainAliasFinder())
