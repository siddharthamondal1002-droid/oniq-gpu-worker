#!/usr/bin/env bash
# Mutation check for the handler's refusal diagnostic (2026-09-12).
#
# Each mutation opens exactly the hole one assertion claims to guard. A
# mutation that reports GREEN is a test that is not testing; one that
# reports NOTAPPLIED is a stale anchor and NOT a verdict.
#
# THE UNDO IS A FILE COPY, NEVER `git checkout --`. On 2026-09-11 a
# mutation runner in the sibling repository restored with git and deleted
# the uncommitted work it was meant to protect: git's undo is relative to
# a COMMIT, and a mutation is relative to the file it found.
set -uo pipefail
cd "$(dirname "$0")/.."

TESTS="tests/test_handler.py"
SAVE="$(mktemp -d)"
trap 'restore; rm -rf "$SAVE"' EXIT

save()    { cp handler.py "$SAVE/handler.py"; }
restore() { [ -f "$SAVE/handler.py" ] && cp "$SAVE/handler.py" handler.py; }

green() { python3 -m pytest "$TESTS" -q --tb=no -p no:cacheprovider >/dev/null 2>&1; }

changed() {
  if cmp -s handler.py "$SAVE/handler.py"; then
    echo "  NOTAPPLIED — the anchor did not match. Not a verdict; fix the anchor."
    return 1
  fi
  return 0
}

verdict() {
  local label="$1"
  if ! changed; then return; fi
  if green; then
    echo "  GREEN  <- ESCAPED: $label"
  else
    echo "  RED    $label"
  fi
}

echo "== baseline =="
if green; then
  echo "  baseline GREEN"
else
  echo "  BASELINE IS RED — stop. No verdict below means anything."
  exit 1
fi

save

echo "== M1: the fallback reverts to the class name alone =="
python3 - <<'PY'
import pathlib
p = pathlib.Path("handler.py"); s = p.read_text()
s = s.replace("""        own = _own_refusal(exc)
        if own is not None:
            return _error(own[0], own[1])
        return _error("unexpected-exception", _foreign_detail(exc))""",
"""        return _error("unexpected-exception", type(exc).__name__)""")
p.write_text(s)
PY
verdict "CheckpointInconsistent is reported by name only"
restore

echo "== M2: every exception counts as ONIQ's own =="
python3 - <<'PY'
import pathlib
p = pathlib.Path("handler.py"); s = p.read_text()
s = s.replace("""    module = sys.modules.get(type(exc).__module__)
    path = getattr(module, "__file__", None)
    if not path:
        return False
    return os.path.dirname(os.path.abspath(path)) == _OWN_DIR""",
"""    return True""")
p.write_text(s)
PY
verdict "a dependency's message reaches the caller"
restore

echo "== M3: foreign details become the raw message =="
python3 - <<'PY'
import pathlib
p = pathlib.Path("handler.py"); s = p.read_text()
s = s.replace("""    name = type(exc).__name__
    if not isinstance(exc, OSError):
        return name""",
"""    name = str(exc)
    if not isinstance(exc, OSError):
        return name""")
p.write_text(s)
PY
verdict "a dependency's text is echoed verbatim"
restore

echo "== M4: OSError loses its errno and path =="
python3 - <<'PY'
import pathlib
p = pathlib.Path("handler.py"); s = p.read_text()
s = s.replace("""    if not isinstance(exc, OSError):
        return name""",
"""    return name
    if not isinstance(exc, OSError):
        return name""")
p.write_text(s)
PY
verdict "PermissionError stops naming the directory"
restore

echo "== M5: a refusal's .state is not read as its code =="
python3 - <<'PY'
import pathlib
p = pathlib.Path("handler.py"); s = p.read_text()
s = s.replace("""    code = getattr(exc, "code", None) or getattr(exc, "state", None) or _code_for(exc)""",
"""    code = getattr(exc, "code", None) or _code_for(exc)""")
p.write_text(s)
PY
verdict "HydrationRefused reports a derived code instead of its state"
restore

echo "== M6: the detail is unbounded =="
python3 - <<'PY'
import pathlib
p = pathlib.Path("handler.py"); s = p.read_text()
s = s.replace("""    return str(code)[:80], str(detail)[:MAX_REFUSAL_DETAIL]""",
"""    return str(code)[:80], str(detail)""")
p.write_text(s)
PY
verdict "an unbounded refusal detail reaches the caller"
restore

echo "== M7: own-ness stops depending on the directory =="
python3 - <<'PY'
import pathlib
p = pathlib.Path("handler.py"); s = p.read_text()
s = s.replace("""    return os.path.dirname(os.path.abspath(path)) == _OWN_DIR""",
"""    return True""")
p.write_text(s)
PY
verdict "any importable class counts as ONIQ's own"
restore

echo "== done =="
