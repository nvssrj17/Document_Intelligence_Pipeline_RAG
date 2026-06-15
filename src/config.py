from importlib import util
from pathlib import Path

_ROOT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.py"
_SPEC = util.spec_from_file_location("_root_config", _ROOT_CONFIG_PATH)

if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load configuration from {_ROOT_CONFIG_PATH}")

_MODULE = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name in dir(_MODULE):
    if _name.isupper():
        globals()[_name] = getattr(_MODULE, _name)
