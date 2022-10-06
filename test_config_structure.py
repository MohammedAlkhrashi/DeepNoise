# from configs.algorithms.default_erm import cfg

# print(cfg)

# %%
import importlib.util
import sys
from runpy import run_path

# %%
config_file = run_path("configs/algorithms/default_erm.py")
cfg = config_file["cfg"]
print(cfg)

# %%
import importlib.util
import sys

spec = importlib.util.spec_from_file_location("module.name", "/path/to/file.py")
foo = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = foo
spec.loader.exec_module(foo)
foo.MyClass()

# %%
import importlib.util
import sys

spec = importlib.util.spec_from_file_location(
    "module.name", "configs/algorithms/default_erm.py"
)
foo = importlib.util.module_from_spec(spec)
sys.modules["module.name"] = foo
spec.loader.exec_module(foo)
# %%
print(foo.cfg)
print()
print(cfg)

# %%
