import os
import capnp
from importlib.resources import as_file, files

capnp.remove_import_hook()

with as_file(files("cereal")) as fspath:
  CEREAL_PATH = fspath.as_posix()
  SCHEMA_PATH = os.path.join(CEREAL_PATH, "gen", "capnp")
  if not os.path.exists(os.path.join(SCHEMA_PATH, "car.capnp")):
    SCHEMA_PATH = CEREAL_PATH

  log = capnp.load(os.path.join(SCHEMA_PATH, "log.capnp"))
  car = capnp.load(os.path.join(SCHEMA_PATH, "car.capnp"))
  custom = capnp.load(os.path.join(SCHEMA_PATH, "custom.capnp"))
