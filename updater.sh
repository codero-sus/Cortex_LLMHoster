#!/usr/bin/env bash
# Clone and fast-forward only the fixed Cortex LLMHoster update branch.
set -Eeuo pipefail

readonly UPDATE_BRANCH="arena/b3072048-cortex-llmhoster"
readonly REPOSITORY_URL="https://github.com/codero-sus/Cortex_LLMHoster.git"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  echo "Error: run this updater from a Git checkout of Cortex LLMHoster." >&2
  exit 1
fi
readonly PROJECT_ROOT
cd "$PROJECT_ROOT"

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$UPDATE_BRANCH" ]]; then
  echo "Error: expected branch '$UPDATE_BRANCH'; currently on '$current_branch'." >&2
  echo "The updater will not switch to or update main." >&2
  exit 1
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$"
backup_root="$PROJECT_ROOT/backups/json-$stamp"
mkdir -p "$backup_root"
json_count=0
while IFS= read -r -d '' json_file; do
  relative_path="${json_file#"$PROJECT_ROOT"/}"
  mkdir -p "$backup_root/$(dirname "$relative_path")"
  cp -p "$json_file" "$backup_root/$relative_path"
  json_count=$((json_count + 1))
done < <(
  find "$PROJECT_ROOT" \
    -type d \( \
      -path "$PROJECT_ROOT/.git" -o \
      -path "$PROJECT_ROOT/.venv" -o \
      -path "$PROJECT_ROOT/venv" -o \
      -path "$PROJECT_ROOT/backups" -o \
      -path "$PROJECT_ROOT/node_modules" -o \
      -path "$PROJECT_ROOT/build" -o \
      -path "$PROJECT_ROOT/dist" -o \
      -path "$PROJECT_ROOT/.cache" -o \
      -path "$PROJECT_ROOT/.pytest_cache" -o \
      -path "$PROJECT_ROOT/.mypy_cache" -o \
      -path "$PROJECT_ROOT/.ruff_cache" -o \
      -path "$PROJECT_ROOT/.tox" -o \
      -path "$PROJECT_ROOT/.nox" -o \
      -path "$PROJECT_ROOT/__pycache__" -o \
      -path "$PROJECT_ROOT/coverage" -o \
      -path "$PROJECT_ROOT/out" -o \
      -path "$PROJECT_ROOT/target" -o \
      -path "$PROJECT_ROOT/.next" -o \
      -path "$PROJECT_ROOT/.nuxt" -o \
      -path "$PROJECT_ROOT/.svelte-kit" -o \
      -path "$PROJECT_ROOT/.turbo" -o \
      -path "$PROJECT_ROOT/.vite" -o \
      -path "$PROJECT_ROOT/.parcel-cache" \
    \) -prune -o \
    -type f -name '*.json' -print0
)
echo "JSON backup created at: $backup_root ($json_count file(s))"

clone_parent="${TMPDIR:-/tmp}"
clone_root="$clone_parent/Cortex_LLMHoster-update-$$-$RANDOM"
while [[ -e "$clone_root" ]]; do
  clone_root="$clone_parent/Cortex_LLMHoster-update-$$-$RANDOM"
done
cleanup() { rm -rf -- "$clone_root"; }
trap cleanup EXIT

echo "Cloning branch $UPDATE_BRANCH from $REPOSITORY_URL ..."
git clone --branch "$UPDATE_BRANCH" "$REPOSITORY_URL" "$clone_root"

cloned_branch="$(git -C "$clone_root" branch --show-current)"
if [[ "$cloned_branch" != "$UPDATE_BRANCH" ]]; then
  echo "Error: Git cloned unexpected branch '$cloned_branch'. No checkout files were changed." >&2
  exit 1
fi

echo "Fast-forwarding the existing checkout from the cloned branch..."
git fetch --no-tags "$clone_root" \
  "refs/heads/$UPDATE_BRANCH:refs/remotes/origin/$UPDATE_BRANCH"
git merge --ff-only "refs/remotes/origin/$UPDATE_BRANCH"

echo "Cortex LLMHoster source updated from $UPDATE_BRANCH."
echo "This updater uses Git only; it does not require Python, pip, or a virtual environment."
echo "Restart Cortex LLMHoster to load the updated source."
echo "JSON backup retained at: $backup_root"
