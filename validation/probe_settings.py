"""What each benchmark candidate's PUBLISHER says to run it with. $0.

Every probe row needs a step count and a guidance scale, and there are only
two honest ways to have one: read it from the publisher, or don't pass it and
let the pipeline's own default stand. Inventing a number would make the
benchmark a measurement of the invention — a 13B distilled model sampled at a
non-distilled step count is not that model performing badly, it is the wrong
experiment.

So this reads, for each candidate, the two places the answer actually lives:

- README.md, the model card. Publishers put their reference snippet there, and
  a snippet is a citation. The extract below is deliberately DUMB: it prints
  the lines that mention a step or guidance setting and lets a human read
  them. A regex that silently picked the wrong number would be worse than no
  reading at all.
- scheduler/scheduler_config.json, which carries the sampler's own defaults
  and, for distilled checkpoints, sometimes the shortened timestep schedule.

Read-only over plain HTTPS against the public registry. No token is required
for a public repo, no GPU is rented, and nothing here writes.
"""

from __future__ import annotations

import json
import re
import urllib.request

import modelprobe

HOST = "https://huggingface.co"
INTERESTING = re.compile(
    r"(num_inference_steps|guidance_scale|guidance_scale_2|timesteps|"
    r"num_steps|sampling_steps|flow_shift|shift\s*=)",
    re.IGNORECASE,
)


def fetch(url: str, opener=urllib.request.urlopen) -> str | None:
    try:
        with opener(url, timeout=60) as response:
            return response.read().decode("utf-8", "replace")
    except Exception:
        return None


def card_url(repo: str, revision: str, path: str) -> str:
    return f"{HOST}/{repo}/raw/{revision}/{path}"


def interesting_lines(text: str, limit: int = 40) -> list[str]:
    """Lines a human should read, in file order, deduplicated."""
    out: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or not INTERESTING.search(line):
            continue
        if line in out:
            continue
        out.append(line)
        if len(out) >= limit:
            break
    return out


def scheduler_defaults(text: str) -> dict:
    try:
        doc = json.loads(text)
    except Exception:
        return {}
    if not isinstance(doc, dict):
        return {}
    keys = ("num_train_timesteps", "shift", "flow_shift", "use_dynamic_shifting",
            "base_shift", "max_shift", "_class_name")
    return {k: doc[k] for k in keys if k in doc}


def report(fetcher=fetch) -> int:
    print("=== PUBLISHER-STATED SAMPLING SETTINGS, read at $0 ===")
    print("Nothing here is applied automatically. A row gets a step count only")
    print("when a human reads one of these citations and writes it down.\n")
    for key, row in sorted(modelprobe.PROBE_MODELS.items()):
        print(f"--- {key}  ({row['repo']} @ {row['revision'][:12]})")
        print(f"    currently probing with: "
              f"steps={row.get('steps') or 'PIPELINE DEFAULT'}, "
              f"guidance={row.get('guidance') if row.get('guidance') is not None else 'PIPELINE DEFAULT'}")
        card = fetcher(card_url(row["repo"], row["revision"], "README.md"))
        if card is None:
            print("    README.md: UNREADABLE (no citation available)")
        else:
            lines = interesting_lines(card)
            if not lines:
                print("    README.md: readable, states no step or guidance setting")
            for line in lines:
                print(f"    card | {line}")
        sched = fetcher(
            card_url(row["repo"], row["revision"], "scheduler/scheduler_config.json")
        )
        if sched is None:
            print("    scheduler/scheduler_config.json: absent or unreadable")
        else:
            print(f"    scheduler | {scheduler_defaults(sched)}")
        print()
    for key, reason in sorted(modelprobe.NOT_EVALUATED.items()):
        print(f"--- {key}: NOT_EVALUATED ({reason}) — not read, not probed")
    return 0


if __name__ == "__main__":
    raise SystemExit(report())
