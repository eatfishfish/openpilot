import errno
import os

try:
  import xattr
except ImportError:
  xattr = None

_cached_attributes: dict[tuple, bytes | None] = {}

def getxattr(path: str, attr_name: str) -> bytes | None:
  key = (path, attr_name)
  if key not in _cached_attributes:
    try:
      if xattr is not None:
        response = xattr.getxattr(path, attr_name)
      else:
        response = os.getxattr(path, attr_name)
    except OSError as e:
      # ENODATA (Linux) or ENOATTR (macOS) means attribute hasn't been set
      if e.errno == errno.ENODATA or (hasattr(errno, 'ENOATTR') and e.errno == errno.ENOATTR):
        response = None
      else:
        raise
    _cached_attributes[key] = response
  return _cached_attributes[key]

def setxattr(path: str, attr_name: str, attr_value: bytes) -> None:
  _cached_attributes.pop((path, attr_name), None)
  if xattr is not None:
    xattr.setxattr(path, attr_name, attr_value)
  else:
    os.setxattr(path, attr_name, attr_value)
