"""
tests/mock_deps.py
==================
Stubs for third-party packages not installed in the test environment.
Import this module FIRST in every test file so the stubs are registered
before any TradeIQ module is imported.
"""
import sys
import types

# ── pytz stub — MUST come before pandas import ────────────────────────────────
# pandas checks pytz.__version__ internally; provide a dummy one.
import datetime as _dt

class _FixedTZ(_dt.tzinfo):
    def __init__(self, name="UTC", offset_h=0):
        self._name = name
        self._offset = _dt.timedelta(hours=offset_h)
    def utcoffset(self, dt): return self._offset
    def tzname(self, dt):    return self._name
    def dst(self, dt):       return _dt.timedelta(0)
    def localize(self, dt, is_dst=None): return dt.replace(tzinfo=self)
    def normalize(self, dt): return dt

class _pytz:
    __version__ = "2024.1"
    UTC = _FixedTZ("UTC", 0)
    @staticmethod
    def timezone(name):
        return _FixedTZ(name, -5)
    class exceptions:
        class AmbiguousTimeError(Exception): pass
        class NonExistentTimeError(Exception): pass

pytz_mod = types.ModuleType("pytz")
pytz_mod.__version__ = "2024.1"
pytz_mod.timezone = _pytz.timezone
pytz_mod.UTC = _pytz.UTC
pytz_mod.exceptions = _pytz.exceptions
sys.modules.setdefault("pytz", pytz_mod)

# ── loguru stub ──────────────────────────────────────────────────────────────
def _noop(*args, **kwargs): pass

class _Logger:
    debug = info = warning = error = critical = exception = success = staticmethod(_noop)
    def bind(self, **kw): return self
    def opt(self, *a, **kw): return self

loguru_mod = types.ModuleType("loguru")
loguru_mod.logger = _Logger()
sys.modules.setdefault("loguru", loguru_mod)

# ── robin_stocks stub ────────────────────────────────────────────────────────
robin_stocks_mod = types.ModuleType("robin_stocks")
robin_stocks_mod.robinhood = types.ModuleType("robin_stocks.robinhood")
sys.modules.setdefault("robin_stocks", robin_stocks_mod)
sys.modules.setdefault("robin_stocks.robinhood", robin_stocks_mod.robinhood)

# ── schedule stub ────────────────────────────────────────────────────────────
schedule_mod = types.ModuleType("schedule")
def _every(*a, **kw):
    class _Chain:
        def __getattr__(self, name):
            def _(*args, **kwargs): return self
            return _
    return _Chain()
schedule_mod.every = _every
schedule_mod.run_pending = _noop
schedule_mod.clear = _noop
sys.modules.setdefault("schedule", schedule_mod)

# ── sendgrid stub ────────────────────────────────────────────────────────────
sg_mod = types.ModuleType("sendgrid")
sys.modules.setdefault("sendgrid", sg_mod)

# ── anthropic stub ───────────────────────────────────────────────────────────
anthropic_mod = types.ModuleType("anthropic")
class _Anthropic:
    def __init__(self, **kw): pass
    class messages:
        @staticmethod
        def create(**kw): return type("R", (), {"content": [type("B", (), {"text": "{}"})()]})()
anthropic_mod.Anthropic = _Anthropic
sys.modules.setdefault("anthropic", anthropic_mod)

# ── dotenv stub ──────────────────────────────────────────────────────────────
dotenv_mod = types.ModuleType("dotenv")
dotenv_mod.load_dotenv = _noop
sys.modules.setdefault("dotenv", dotenv_mod)
sys.modules.setdefault("python_dotenv", dotenv_mod)


# ── robin_stocks stub ────────────────────────────────────────────────────────
robin_stocks_mod = types.ModuleType("robin_stocks")
robin_stocks_mod.robinhood = types.ModuleType("robin_stocks.robinhood")
sys.modules.setdefault("robin_stocks", robin_stocks_mod)
sys.modules.setdefault("robin_stocks.robinhood", robin_stocks_mod.robinhood)

# ── schedule stub ────────────────────────────────────────────────────────────
schedule_mod = types.ModuleType("schedule")
def _every(*a, **kw):
    class _Chain:
        def __getattr__(self, name):
            def _(*args, **kwargs): return self
            return _
    return _Chain()
schedule_mod.every = _every
schedule_mod.run_pending = _noop
schedule_mod.clear = _noop
sys.modules.setdefault("schedule", schedule_mod)

# ── sendgrid stub ────────────────────────────────────────────────────────────
sg_mod = types.ModuleType("sendgrid")
sys.modules.setdefault("sendgrid", sg_mod)

# ── anthropic stub ───────────────────────────────────────────────────────────
anthropic_mod = types.ModuleType("anthropic")
class _Anthropic:
    def __init__(self, **kw): pass
    class messages:
        @staticmethod
        def create(**kw): return type("R", (), {"content": [type("B", (), {"text": "{}"})()]})()
anthropic_mod.Anthropic = _Anthropic
sys.modules.setdefault("anthropic", anthropic_mod)

# ── dotenv stub ──────────────────────────────────────────────────────────────
dotenv_mod = types.ModuleType("dotenv")
dotenv_mod.load_dotenv = _noop
sys.modules.setdefault("dotenv", dotenv_mod)
sys.modules.setdefault("python_dotenv", dotenv_mod)
