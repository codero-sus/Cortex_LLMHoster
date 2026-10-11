#!/usr/bin/env bash
# Cortex LLMHoster source updater. Updates only the current project branch.
set -Eeuo pipefail

readonly UPDATE_BRANCH="arena/b3072048-cortex-llmhoster"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! PROJECT_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  echo "Error: this updater must run inside a Git checkout." >&2
  exit 1
fi
readonly PROJECT_ROOT
cd "$PROJECT_ROOT"

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$UPDATE_BRANCH" ]]; then
  echo "Error: expected branch '$UPDATE_BRANCH'; currently on '$current_branch'." >&2
  echo "Cortex LLMHoster updater will not switch to or update main." >&2
  exit 1
fi
if ! git remote get-url origin >/dev/null 2>&1; then
  echo "Error: this checkout has no Git remote named origin." >&2
  exit 1
fi

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
backup_root="$PROJECT_ROOT/backups/json-$stamp-$$"
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
      -path "$PROJECT_ROOT/dist" \
    \) -prune -o \
    -type f -name '*.json' -print0
)
echo "JSON backup created at: $backup_root ($json_count file(s))"

remote_ref="refs/remotes/origin/$UPDATE_BRANCH"
echo "Fetching all tracked project-file updates from origin/$UPDATE_BRANCH ..."
git fetch origin "$UPDATE_BRANCH:$remote_ref"
# Fast-forward only: preserve local edits and refuse divergent history.
git merge --ff-only "$remote_ref"

echo "Cortex LLMHoster repository files are updated from $UPDATE_BRANCH."

bootstrap_python=""
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  bootstrap_python="$PROJECT_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  bootstrap_python="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  bootstrap_python="$(command -v python)"
else
  echo "Error: source files were updated, but no Python interpreter is available to refresh the install." >&2
  exit 1
fi

selected_python="$(
  PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$bootstrap_python" -c \
    'from pathlib import Path; from cortex_llmhoster.python_runtime import resolve_python_executable; print(resolve_python_executable(project_root=Path.cwd()))'
)"
if [[ -z "$selected_python" ]]; then
  echo "Error: could not resolve the configured Python interpreter." >&2
  exit 1
fi
"$selected_python" -m pip install --upgrade --no-input --editable "$PROJECT_ROOT"

echo "Cortex LLMHoster update complete. Restart the running server to load the new files."
echo "JSON backup retained at: $backup_root"
