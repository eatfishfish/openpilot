#!/usr/bin/env bash
set -e

# Increase the pip timeout to handle TimeoutError
export PIP_DEFAULT_TIMEOUT=200

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
ROOT="$DIR"/../
cd "$ROOT"

find_uv() {
  local candidate
  for candidate in \
    "$(command -v uv 2>/dev/null || true)" \
    "/data/uv-bin/uv" \
    "$HOME/.local/bin/uv" \
    "/usr/local/bin/uv"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      UV_BIN="$(dirname "$candidate")"
      PATH="$UV_BIN:$PATH"
      export PATH
      return 0
    fi
  done
  return 1
}

install_uv() {
  local install_dir="${UV_INSTALL_DIR:-$HOME/.local/bin}"
  local arch asset installer archive extracted
  mkdir -p "$install_dir"

  echo "uv not found, installing..."

  # astral.sh is the normal installer, but it may be unavailable in
  # restricted CI networks. Verify the download before executing it.
  installer="$(mktemp)"
  if curl --fail --retry 3 --retry-delay 2 -LsS https://astral.sh/uv/install.sh -o "$installer" &&
     sh "$installer" &&
     find_uv; then
    rm -f "$installer"
    return 0
  fi
  rm -f "$installer"

  case "$(uname -m)" in
    x86_64|amd64) asset="uv-x86_64-unknown-linux-gnu.tar.gz" ;;
    aarch64|arm64) asset="uv-aarch64-unknown-linux-gnu.tar.gz" ;;
    *)
      echo "Unsupported architecture for uv fallback: $(uname -m)" >&2
      return 1
      ;;
  esac

  archive="$(mktemp)"
  extracted="$(mktemp -d)"
  if curl --fail --retry 3 --retry-delay 2 -LsS \
      "https://github.com/astral-sh/uv/releases/latest/download/${asset}" -o "$archive" &&
     tar -xzf "$archive" -C "$extracted"; then
    candidate="$(find "$extracted" -type f -name uv -perm -u+x -print -quit)"
    if [[ -n "${candidate:-}" ]]; then
      install -m 0755 "$candidate" "$install_dir/uv"
      rm -f "$archive"
      rm -rf "$extracted"
      find_uv
      return 0
    fi
  fi
  rm -f "$archive"
  rm -rf "$extracted"

  # Last fallback for runners that can reach PyPI but not the binary hosts.
  if command -v python3 >/dev/null 2>&1 &&
     python3 -m pip install --user --upgrade uv &&
     find_uv; then
    return 0
  fi

  echo "Unable to install uv. Checked PATH, /data/uv-bin, astral.sh, GitHub Releases, and PyPI." >&2
  return 1
}

if ! find_uv; then
  install_uv || exit 1
fi

echo "updating uv..."
# ok to fail, can also fail due to installing with brew
uv self update || true

echo "installing python packages..."
uv sync ${UV_SYNC_ARGS:---frozen --all-extras}
source .venv/bin/activate

if [[ "$(uname)" == 'Darwin' ]]; then
  touch "$ROOT"/.env
  echo "# msgq doesn't work on mac" >> "$ROOT"/.env
  echo "export ZMQ=1" >> "$ROOT"/.env
  echo "export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES" >> "$ROOT"/.env
fi
