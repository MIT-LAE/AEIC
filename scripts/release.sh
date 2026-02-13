#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 {major|minor|patch}"
  exit 1
fi

PART=$1

# Ensure clean working tree
if ! git diff-index --quiet HEAD --; then
  echo "Working tree is dirty. Commit or stash changes."
  exit 1
fi

git checkout main
git pull upstream main

uv run bump-my-version bump "$PART"
uv lock
git add uv.lock
git commit --amend --no-edit

VERSION=$(uv run python - <<'EOF'
import tomllib
with open("pyproject.toml", "rb") as f:
    print(tomllib.load(f)["project"]["version"])
EOF
)

BRANCH="release-$VERSION"

git checkout -b "$BRANCH"
git push origin "$BRANCH"

echo
echo "Release branch created: $BRANCH"
echo "Open a PR, merge it, then run:"
echo
echo "  git checkout main"
echo "  git pull upstream main"
echo "  git tag -a v$VERSION -m 'Release v$VERSION'"
echo "  git push upstream v$VERSION"
