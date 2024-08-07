#!/usr/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd $DIR

if [ ! -z "$(git status --porcelain)" ]; then
  echo "Dirty working tree after build:"
  git status --porcelain
  exit 0
fi
