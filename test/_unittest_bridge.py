"""Runs the pytest-style modules in this directory under stdlib unittest.

A pytest-style file needs two lines:

    from _unittest_bridge import load_tests_from, pytest
    load_tests = load_tests_from(globals())

Fixtures are resolved and parametrize expanded into one method per case. The
files stay runnable under pytest; `pytest` here is the real module when it is
installed and a stdlib stand-in when it is not.
"""

from __future__ import annotations

import importlib
import inspect
import io
import os
import re
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

try:  # pragma: no cover
    import pytest as _real_pytest
except Exception:  # noqa: BLE001
    _real_pytest = None

try:  # pragma: no cover
    from _pytest.outcomes import Skipped as _PytestSkipped
except Exception:  # noqa: BLE001
    _PytestSkipped = None


__all__ = ["pytest", "load_tests_from", "skip_module", "MonkeyPatch"]


# --------------------------------------------------------------------------
# Skipping, understood by both runners
# --------------------------------------------------------------------------

if _PytestSkipped is not None:

    class _Skip(unittest.SkipTest, _PytestSkipped):
        """A skip both unittest and pytest recognise as a skip."""

        def __init__(self, msg: str = "", allow_module_level: bool = True):
            _PytestSkipped.__init__(
                self, msg, allow_module_level=allow_module_level)
            self.msg = msg

        def __str__(self) -> str:
            return self.msg

else:

    class _Skip(unittest.SkipTest):  # type: ignore[no-redef]
        """A skip both unittest and pytest recognise as a skip."""

        def __init__(self, msg: str = "", allow_module_level: bool = True):
            unittest.SkipTest.__init__(self, msg)
            self.msg = msg
            self.allow_module_level = allow_module_level

        def __str__(self) -> str:
            return self.msg


def skip_module(reason: str):
    """Skip the whole module from its import."""
    raise _Skip(reason, allow_module_level=True)


def _skip(reason: str = ""):
    raise _Skip(reason, allow_module_level=False)


_skip.Exception = _Skip  # type: ignore[attr-defined]


def _fail(reason: str = ""):
    raise AssertionError(reason)


def _version_tuple(text: str) -> tuple:
    return tuple(int(p) for p in re.findall(r"\d+", str(text)))


def _importorskip(modname: str, minversion=None, reason=None):
    try:
        module = importlib.import_module(modname)
    except ImportError as exc:
        raise _Skip(reason or f"could not import {modname!r}: {exc}",
                    allow_module_level=True) from None
    if minversion is not None:
        have = getattr(module, "__version__", None)
        if have is None or _version_tuple(have) < _version_tuple(minversion):
            raise _Skip(
                reason or f"{modname} {have} is older than {minversion}",
                allow_module_level=True)
    return module


# --------------------------------------------------------------------------
# approx / raises -- the real thing when pytest is around, else a stand-in
# --------------------------------------------------------------------------

class _Approx:
    """Minimal ``pytest.approx``: scalars, sequences and mappings."""

    def __init__(self, expected, rel=None, abs=None):
        self.expected = expected
        self.rel = 1e-6 if rel is None else rel
        self.abs = 1e-12 if abs is None else abs

    def _close(self, got, want) -> bool:
        try:
            got, want = float(got), float(want)
        except (TypeError, ValueError):
            return got == want
        return abs(got - want) <= max(self.abs, self.rel * abs(want))

    def __eq__(self, actual) -> bool:
        want = self.expected
        if isinstance(want, dict):
            if not isinstance(actual, dict) or set(actual) != set(want):
                return False
            return all(self._close(actual[k], want[k]) for k in want)
        if isinstance(want, (list, tuple)):
            if len(actual) != len(want):
                return False
            return all(self._close(a, w) for a, w in zip(actual, want))
        return self._close(actual, want)

    def __ne__(self, actual) -> bool:
        return not self.__eq__(actual)

    def __repr__(self) -> str:
        return f"approx({self.expected!r}, rel={self.rel}, abs={self.abs})"


class _ExcInfo:
    def __init__(self):
        self.type = None
        self.value = None
        self.tb = None


class _RaisesContext:
    def __init__(self, expected, match=None):
        self.expected = expected
        self.match = match
        self.excinfo = _ExcInfo()

    def __enter__(self) -> _ExcInfo:
        return self.excinfo

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            name = getattr(self.expected, "__name__", self.expected)
            raise AssertionError(f"DID NOT RAISE {name}")
        if not issubclass(exc_type, self.expected):
            return False
        if self.match is not None and not re.search(self.match, str(exc)):
            raise AssertionError(
                f"{exc!r} does not match {self.match!r}")
        self.excinfo.type, self.excinfo.value, self.excinfo.tb = exc_type, exc, tb
        return True


def _approx(expected, rel=None, abs=None, **kwargs):
    if _real_pytest is not None:
        return _real_pytest.approx(expected, rel=rel, abs=abs, **kwargs)
    return _Approx(expected, rel=rel, abs=abs)


def _raises(expected, *args, **kwargs):
    if _real_pytest is not None:
        return _real_pytest.raises(expected, *args, **kwargs)
    return _RaisesContext(expected, match=kwargs.get("match"))


# --------------------------------------------------------------------------
# Fixtures and parametrize
# --------------------------------------------------------------------------

class _FixtureDef:
    __slots__ = ("func", "autouse", "name")

    def __init__(self, func, autouse: bool, name: str):
        self.func, self.autouse, self.name = func, autouse, name


# (module name, fixture name) -> definition
_FIXTURES: dict[tuple[str, str], _FixtureDef] = {}


def _fixture(func=None, *, scope="function", params=None, autouse=False,
             ids=None, name=None):
    """``pytest.fixture`` that also records the fixture for the unittest side."""

    def register(f):
        fixture_name = name or f.__name__
        _FIXTURES[(f.__module__, fixture_name)] = _FixtureDef(
            f, bool(autouse), fixture_name)
        if _real_pytest is not None:
            return _real_pytest.fixture(
                scope=scope, params=params, autouse=autouse, ids=ids,
                name=name)(f)
        return f

    return register(func) if callable(func) else register


class _MarkProxy:
    def parametrize(self, argnames, argvalues, ids=None, **kwargs):
        names = ([n.strip() for n in argnames.split(",")]
                 if isinstance(argnames, str) else list(argnames))
        values = list(argvalues)

        def decorate(f):
            f.__dict__.setdefault("_bridge_parametrize", []).insert(
                0, (names, values, list(ids) if ids else None))
            if _real_pytest is not None:
                return _real_pytest.mark.parametrize(
                    argnames, argvalues, ids=ids, **kwargs)(f)
            return f

        return decorate

    def __getattr__(self, item):
        if _real_pytest is not None:
            return getattr(_real_pytest.mark, item)

        def mark(*args, **kwargs):  # a no-op marker without pytest
            if len(args) == 1 and not kwargs and callable(args[0]):
                return args[0]
            return lambda f: f

        return mark


class _PytestCompat:
    """What the test modules import as ``pytest``."""

    fixture = staticmethod(_fixture)
    mark = _MarkProxy()
    skip = staticmethod(_skip)
    fail = staticmethod(_fail)
    importorskip = staticmethod(_importorskip)
    approx = staticmethod(_approx)
    raises = staticmethod(_raises)

    def __getattr__(self, item):  # anything else, straight from pytest
        if _real_pytest is not None:
            return getattr(_real_pytest, item)
        raise AttributeError(
            f"pytest.{item} is not provided by {__name__} and pytest is "
            f"not installed")


pytest = _PytestCompat()


# --------------------------------------------------------------------------
# Built-in fixtures
# --------------------------------------------------------------------------

class _MissingType:
    def __repr__(self) -> str:
        return "<missing>"


_MISSING = _MissingType()


class MonkeyPatch:
    """The subset of ``pytest``'s monkeypatch these tests use."""

    def __init__(self):
        self._undo: list = []

    def setattr(self, target, name=_MISSING, value=_MISSING, raising=True):
        if isinstance(target, str):
            target, attribute = _resolve_target(target)
            target, name, value = target, attribute, name
        old = getattr(target, name, _MISSING)
        if old is _MISSING and raising:
            raise AttributeError(f"{target!r} has no attribute {name!r}")
        self._undo.append(lambda: (setattr(target, name, old)
                                   if old is not _MISSING
                                   else _safe_delattr(target, name)))
        setattr(target, name, value)

    def delattr(self, target, name, raising=True):
        old = getattr(target, name, _MISSING)
        if old is _MISSING:
            if raising:
                raise AttributeError(f"{target!r} has no attribute {name!r}")
            return
        self._undo.append(lambda: setattr(target, name, old))
        delattr(target, name)

    def setitem(self, mapping, name, value):
        old = mapping.get(name, _MISSING)
        self._undo.append(lambda: (mapping.__setitem__(name, old)
                                   if old is not _MISSING
                                   else mapping.pop(name, None)))
        mapping[name] = value

    def delitem(self, mapping, name, raising=True):
        if name not in mapping:
            if raising:
                raise KeyError(name)
            return
        old = mapping[name]
        self._undo.append(lambda: mapping.__setitem__(name, old))
        del mapping[name]

    def setenv(self, name, value, prepend=None):
        value = str(value)
        old = os.environ.get(name, _MISSING)
        if prepend and old is not _MISSING:
            value = value + prepend + old
        self._undo.append(lambda: (os.environ.__setitem__(name, old)
                                   if old is not _MISSING
                                   else os.environ.pop(name, None)))
        os.environ[name] = value

    def delenv(self, name, raising=True):
        old = os.environ.get(name, _MISSING)
        if old is _MISSING:
            if raising:
                raise KeyError(name)
            return
        self._undo.append(lambda: os.environ.__setitem__(name, old))
        del os.environ[name]

    def syspath_prepend(self, path):
        path = str(path)
        sys.path.insert(0, path)
        self._undo.append(
            lambda: sys.path.remove(path) if path in sys.path else None)

    def chdir(self, path):
        old = os.getcwd()
        os.chdir(str(path))
        self._undo.append(lambda: os.chdir(old))

    def undo(self):
        while self._undo:
            self._undo.pop()()


def _safe_delattr(target, name):
    try:
        delattr(target, name)
    except AttributeError:
        pass


def _resolve_target(dotted: str):
    module, _, attr = dotted.rpartition(".")
    return importlib.import_module(module), attr


class _CaptureResult(tuple):
    """What ``capsys.readouterr()`` gives back."""

    @property
    def out(self):
        return self[0]

    @property
    def err(self):
        return self[1]


class _CapSys:
    def __init__(self, case):
        self._out, self._err = io.StringIO(), io.StringIO()
        self._redirect = (redirect_stdout(self._out), redirect_stderr(self._err))
        for ctx in self._redirect:
            ctx.__enter__()
        case._bridge_finalizers.append(self._stop)

    def _stop(self):
        for ctx in reversed(self._redirect):
            ctx.__exit__(None, None, None)

    def readouterr(self) -> _CaptureResult:
        out, err = self._out.getvalue(), self._err.getvalue()
        self._out.seek(0), self._out.truncate(0)
        self._err.seek(0), self._err.truncate(0)
        return _CaptureResult((out, err))


def _tmp_path(case) -> Path:
    directory = Path(tempfile.mkdtemp(prefix="nngpt-test-"))
    case._bridge_finalizers.append(
        lambda: shutil.rmtree(directory, ignore_errors=True))
    return directory


def _monkeypatch(case) -> MonkeyPatch:
    patch = MonkeyPatch()
    case._bridge_finalizers.append(patch.undo)
    return patch


_BUILTIN_FIXTURES = {
    "tmp_path": _tmp_path,
    "tmpdir": _tmp_path,
    "monkeypatch": _monkeypatch,
    "capsys": _CapSys,
}


# --------------------------------------------------------------------------
# Turning the module's functions into a TestCase
# --------------------------------------------------------------------------

class _BridgeCase(unittest.TestCase):
    """Base of every generated case; holds the fixture machinery."""

    bridge_module = ""

    maxDiff = None

    def setUp(self):
        self._bridge_cache: dict = {}
        self._bridge_finalizers: list = []

    def tearDown(self):
        errors = []
        while self._bridge_finalizers:
            try:
                self._bridge_finalizers.pop()()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
        if errors:
            raise errors[0]

    def _bridge_fixture(self, name: str):
        if name in self._bridge_cache:
            return self._bridge_cache[name]

        definition = _FIXTURES.get((self.bridge_module, name))
        if definition is not None:
            kwargs = {p: self._bridge_fixture(p)
                      for p in inspect.signature(definition.func).parameters}
            value = definition.func(**kwargs)
            if inspect.isgenerator(value):
                generator, value = value, next(value)
                self._bridge_finalizers.append(
                    lambda g=generator: next(g, None))
            self._bridge_cache[name] = value
            return value

        builder = _BUILTIN_FIXTURES.get(name)
        if builder is None:
            raise RuntimeError(
                f"test {self.id()} asks for the fixture {name!r}, which "
                f"neither {self.bridge_module} nor {__name__} defines")
        value = builder(self)
        self._bridge_cache[name] = value
        return value

    def _bridge_run(self, func, params: dict):
        for name, definition in sorted(_FIXTURES.items()):
            if name[0] == self.bridge_module and definition.autouse:
                self._bridge_fixture(name[1])
        kwargs = dict(params)
        for name in inspect.signature(func).parameters:
            if name not in kwargs:
                kwargs[name] = self._bridge_fixture(name)
        func(**kwargs)


def _sanitise(text) -> str:
    return re.sub(r"\W+", "_", str(text)).strip("_") or "x"


def _case_id(values, index: int) -> str:
    if all(isinstance(v, (str, int, float, bool, type(None))) for v in values):
        return "_".join(_sanitise(v) for v in values)
    return str(index)


def _expand(func) -> list[tuple[str, dict]]:
    """One (name suffix, kwargs) per parametrize case; one empty pair if none."""
    specs = func.__dict__.get("_bridge_parametrize")
    if not specs:
        return [("", {})]

    cases: list[tuple[str, dict]] = [("", {})]
    for names, values, ids in specs:
        grown: list[tuple[str, dict]] = []
        for suffix, kwargs in cases:
            for index, value in enumerate(values):
                row = tuple(value) if len(names) > 1 else (value,)
                label = (_sanitise(ids[index]) if ids and index < len(ids)
                         else _case_id(row, index))
                merged = dict(kwargs)
                merged.update(zip(names, row))
                grown.append((f"{suffix}_{label}", merged))
        cases = grown
    return cases


def _make_method(func, params: dict, name: str):
    def method(self):
        self._bridge_run(func, params)

    method.__name__ = name
    method.__doc__ = func.__doc__
    return method


def _class_name(module_name: str) -> str:
    stem = module_name.rpartition(".")[2]
    stem = stem[5:] if stem.startswith("test_") else stem
    return "Test" + "".join(part.title() for part in stem.split("_") if part)


def _build_case(namespace: dict, module_name: str):
    functions = [
        (name, obj) for name, obj in namespace.items()
        if name.startswith("test")
        and inspect.isfunction(obj)
        and getattr(obj, "__module__", None) == module_name
        and (module_name, name) not in _FIXTURES
    ]
    if not functions:
        return None

    attributes: dict = {"bridge_module": module_name, "__module__": module_name}
    for name, func in functions:
        for suffix, params in _expand(func):
            method_name = name + suffix
            while method_name in attributes:
                method_name += "_"
            attributes[method_name] = _make_method(func, params, method_name)

    return type(_class_name(module_name), (_BridgeCase,), attributes)


def load_tests_from(namespace: dict):
    """Build the module's ``load_tests`` hook."""
    module_name = namespace.get("__name__", "__main__")

    def load_tests(loader, standard_tests, pattern):
        suite = unittest.TestSuite()
        suite.addTests(standard_tests)
        case = _build_case(namespace, module_name)
        if case is not None:
            namespace.setdefault(case.__name__, case)
            suite.addTests(loader.loadTestsFromTestCase(case))
        return suite

    return load_tests
