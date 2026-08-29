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


_SHA = re.compile(r"^[0-9a-f]{40}$")


def read_ref(row: dict) -> str:
    """What ref to READ a row at. A pinned sha reads at the pin; a row still
    wearing the interim PENDING-REGISTRY-PIN marker reads at main, because
    resolving that marker into a sha is exactly what this reader is for."""
    revision = row.get("revision") or ""
    return revision if _SHA.match(revision) else "main"


def api_url(repo: str, revision: str) -> str:
    return f"{HOST}/api/models/{repo}/revision/{revision}?blobs=true"


def fnmatch_any(patterns, path: str) -> bool:
    import fnmatch

    return any(fnmatch.fnmatch(path, p) for p in patterns)


def listing_summary(text: str, allow) -> dict:
    """sha + what the allow list would actually fetch, from one API read.

    THE PIN AND THE BILL COME FROM HERE. `sha` is the commit the row must be
    pinned to before any spend, and `allow_bytes` is what snapshot_download
    would move — matched with fnmatch, whose `*` crosses slashes exactly the
    way huggingface_hub's does, so this predicts the hub's behaviour rather
    than a tidier one.
    """
    try:
        doc = json.loads(text)
    except Exception:
        return {}
    if not isinstance(doc, dict) or not doc.get("sha"):
        return {}
    per_component: dict[str, int] = {}
    total = 0
    unsized = []
    for entry in doc.get("siblings") or []:
        name = entry.get("rfilename") or ""
        if not name or not fnmatch_any(allow, name):
            continue
        size = entry.get("size")
        if size is None:
            unsized.append(name)
            continue
        total += size
        head = name.split("/", 1)[0]
        per_component[head] = per_component.get(head, 0) + size
    return {
        "sha": doc["sha"],
        "allow_bytes": total,
        "allow_gib": round(total / 1024**3, 2),
        "per_component_gib": {
            k: round(v / 1024**3, 2) for k, v in sorted(per_component.items())
        },
        "unsized": unsized,
    }


# Config files worth reading per row, beyond the scheduler: the transformer's
# config answers the questions that kept HunyuanVideo-1.5 NOT_EVALUATED (does
# the checkpoint declare use_meanflow; how many channels does it eat), the
# guider config carries the distilled guidance scale that is NOT a __call__
# argument, and the VAE config carries the compression ratios the frame
# arithmetic depends on.
PEEK_FILES = ("transformer/config.json", "guider/config.json", "vae/config.json")
PEEK_KEYS = ("_class_name", "use_meanflow", "in_channels", "out_channels",
             "num_layers", "guidance_scale", "spatial_compression_ratio",
             "temporal_compression_ratio", "latent_channels", "target_size",
             "task_type", "scaling_factor")


def config_peek(text: str) -> dict:
    try:
        doc = json.loads(text)
    except Exception:
        return {}
    if not isinstance(doc, dict):
        return {}
    return {k: doc[k] for k in PEEK_KEYS if k in doc}


def interesting_lines(text: str, limit: int = 40, context: int = 6) -> list[str]:
    """Lines a human should read, in file order, with their surroundings.

    CONTEXT IS THE POINT. LTX's card carries two different step counts, and
    which one applies depends on which pipeline call it sits inside — a bare
    grep would have offered a choice between two numbers with nothing to
    decide it by, which is how a guess gets made while looking like a
    citation. The lines around the hit are what make it a citation.
    """
    lines = text.splitlines()
    keep: set[int] = set()
    hits = 0
    for i, raw in enumerate(lines):
        if not raw.strip() or not INTERESTING.search(raw):
            continue
        hits += 1
        keep.update(range(max(0, i - context), min(len(lines), i + context + 1)))
        if hits >= limit:
            break
    out: list[str] = []
    previous = None
    for i in sorted(keep):
        if previous is not None and i != previous + 1:
            out.append("...")
        text_line = lines[i].rstrip()
        out.append(text_line if text_line.strip() else "")
        previous = i
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


def declared_components(text: str) -> dict:
    """The components a repo's model_index.json says its pipeline needs.

    Keys whose value is a [library, class] pair are subfolders that must be
    on disk; everything else (_class_name, _diffusers_version, scalars) is
    configuration.
    """
    try:
        doc = json.loads(text)
    except Exception:
        return {}
    if not isinstance(doc, dict):
        return {}
    return {
        key: value[1]
        for key, value in doc.items()
        if isinstance(value, list) and len(value) == 2 and value[0]
    }


def covered_by(patterns, folder: str) -> bool:
    """Would the allow list fetch anything from this subfolder?

    fnmatch-style, and deliberately simple: a pattern either names the
    folder's prefix or names a file inside it. Getting this wrong the other
    way is what the LTX vae case was about — this asks the narrower question,
    "is the folder reachable at all".
    """
    for pattern in patterns:
        head = pattern.split("/", 1)[0]
        if head in (folder, "*", "**"):
            return True
    return False


def report(fetcher=fetch) -> int:
    print("=== PUBLISHER-STATED SAMPLING SETTINGS, read at $0 ===")
    print("Nothing here is applied automatically. A row gets a step count only")
    print("when a human reads one of these citations and writes it down.\n")
    for key, row in sorted(modelprobe.PROBE_MODELS.items()):
        ref = read_ref(row)
        print(f"--- {key}  ({row['repo']} @ {row['revision'][:12]})")
        if ref != row["revision"]:
            print(f"    INTERIM: revision {row['revision']!r} is not a pin — "
                  f"reading at {ref!r}; the sha below is the pin to write")
        print(f"    currently probing with: "
              f"steps={row.get('steps') or 'PIPELINE DEFAULT'}, "
              f"guidance={row.get('guidance') if row.get('guidance') is not None else 'PIPELINE DEFAULT'}")
        info = fetcher(api_url(row["repo"], ref))
        summary = listing_summary(info, row["allow"]) if info else {}
        if not summary:
            print("    listing | UNREADABLE — sha and sizes not resolved")
        else:
            print(f"    listing | sha={summary['sha']}")
            print(f"    listing | allow list fetches {summary['allow_gib']} GiB "
                  f"(row says download_gib={row['download_gib']})")
            print(f"    listing | per component: {summary['per_component_gib']}")
            if summary["unsized"]:
                print(f"    listing | *** {len(summary['unsized'])} matched "
                      f"file(s) report no size: {summary['unsized'][:5]} ***")
        for peek_path in PEEK_FILES:
            body = fetcher(card_url(row["repo"], ref, peek_path))
            if body is None:
                continue
            peeked = config_peek(body)
            if peeked:
                print(f"    {peek_path} | {peeked}")
        card = fetcher(card_url(row["repo"], ref, "README.md"))
        if card is None:
            print("    README.md: UNREADABLE (no citation available)")
        else:
            lines = interesting_lines(card)
            if not lines:
                print("    README.md: readable, states no step or guidance setting")
            for line in lines:
                print(f"    card | {line}")
        sched = fetcher(
            card_url(row["repo"], ref, "scheduler/scheduler_config.json")
        )
        if sched is None:
            print("    scheduler/scheduler_config.json: absent or unreadable")
        else:
            print(f"    scheduler | {scheduler_defaults(sched)}")
        # DOES THE ALLOW LIST COVER WHAT THE PIPELINE NEEDS. A missing
        # subfolder is not discovered until from_pretrained runs, which is
        # after the whole download has been paid for on a rented GPU.
        index = fetcher(card_url(row["repo"], ref, "model_index.json"))
        if index is None:
            print("    model_index.json: UNREADABLE — coverage NOT checked")
        else:
            declared = declared_components(index)
            # A body that is not an index is a real outcome (a 404 page, a
            # redirect) and must read as "unknown", never crash the reader.
            try:
                klass = json.loads(index).get("_class_name")
            except Exception:  # noqa: BLE001
                klass = "UNREADABLE"
            print(f"    model_index | _class_name={klass}  probing with {row['pipeline']}")
            missing = [f for f in declared if not covered_by(row["allow"], f)]
            print(f"    components  | {', '.join(sorted(declared)) or '(none declared)'}")
            if missing:
                print(f"    COVERAGE    | *** MISSING FROM allow: {', '.join(sorted(missing))} ***")
            else:
                print("    COVERAGE    | every declared component is inside the allow list")
        print()
    for key, reason in sorted(modelprobe.NOT_EVALUATED.items()):
        print(f"--- {key}: NOT_EVALUATED ({reason}) — not read, not probed")
    return 0


if __name__ == "__main__":
    raise SystemExit(report())
