"""What templates exist, and can one be created without the console?

The endpoint p3zmlv8ek10dzt reports templateId hhhdwtjw0y; a REST GET for
that id answers 404 and the owner cannot find it in the console list. This
reports what the account ACTUALLY holds and whether the API offers a path
to create a template and attach it — because the last time an endpoint
mutation was assumed to exist (workersStandby) it turned out not to be in
the schema at all, and three cycles went into finding that out the hard way.

Read-only. Nothing here creates, patches or deletes.
"""

from __future__ import annotations

import json

import storage


def _render(names):
    if names is None:
        return "unreadable (UNKNOWN)"
    return f"{len(names)} set - " + (", ".join(sorted(names)) or "(none)")


def _rest_env_names(client, template_id: str):
    """The same question asked of the REST template, NAMES ONLY.

    RunPod's REST template carries env as either a mapping or a list of
    {key, value} pairs depending on the shape of the day; both are read,
    and the values are dropped on the floor here rather than travelling
    any further.
    """
    try:
        _, doc = client.get_template(template_id)
    except Exception:
        return None
    if not isinstance(doc, dict):
        return None
    env = doc.get("env")
    if isinstance(env, dict):
        return set(env.keys())
    if isinstance(env, list):
        return {
            e.get("key") for e in env
            if isinstance(e, dict) and e.get("key")
        }
    if env is None:
        # A template with no env at all answers with nothing, and that is
        # a real "none set" rather than a failure to look.
        return set()
    return None


def _report_storage(client, template_id: str):
    """Which storage variables are set on the template. NAMES ONLY.

    A RunPod serverless worker reads its environment from its TEMPLATE, so
    "the secrets are added" is a claim about this object and nowhere else -
    not about the endpoint, and certainly not about a GitHub secret. Checked
    here because a job that cannot write its output should never have been
    started, and because the owner is entitled to see the claim verified
    rather than assumed.

    Unreadable is UNKNOWN, never "ready". Returns (state, verdict line),
    where state is True, False or None.
    """
    # TWO PATHS, because one path is an opinion. The GraphQL view and the
    # REST view of a template are different endpoints on different hosts,
    # and the owner has now said three times that these variables are set
    # while one of them said otherwise. If they disagree, the disagreement
    # IS the finding - reporting either number alone would be a guess
    # wearing a measurement's clothes.
    graph = client.template_env_names_graphql(template_id)
    rest = _rest_env_names(client, template_id)
    print(f"ENV via GraphQL: {_render(graph)}")
    print(f"ENV via REST   : {_render(rest)}")

    if graph is not None and rest is not None and graph != rest:
        only_rest = sorted(rest - graph)
        only_graph = sorted(graph - rest)
        verdict = (
            "ENV DISAGREEMENT: the two APIs do not describe the same template - "
            f"REST-only {only_rest}, GraphQL-only {only_graph}. Trust neither "
            "until this is explained."
        )
        print(verdict)
        return None, verdict

    names = rest if rest is not None else graph
    if names is None:
        verdict = "ENV: unreadable - the answer is UNKNOWN, not 'nothing is set'"
        print(verdict)
        return None, verdict
    print(
        f"ENV ON {template_id}: {len(names)} set - "
        f"{', '.join(sorted(names)) or '(none)'}  (names only, never values)"
    )
    absent = [name for name in storage.REQUIRED_VARS if name not in names]
    if absent:
        verdict = (
            f"STORAGE NOT READY: {', '.join(absent)} not set - every job would "
            "fail closed with storage-not-configured"
        )
        print(verdict)
        return False, verdict
    verdict = "STORAGE READY: every variable storage.py requires is set on the template"
    print(verdict)
    return True, verdict


def report(client, expected_template_id: str) -> tuple:
    # A blank id would make the membership test trivially true and print
    # "MISSING" for a question nobody asked - run 49 did exactly that, because
    # the workflow handed this the endpoint id input, which was empty. An
    # assertion that cannot fail is worse than no assertion: it reads like
    # evidence. Refuse instead.
    expected_template_id = (expected_template_id or "").strip()
    if not expected_template_id:
        print("NO TEMPLATE ID GIVEN: nothing to look for, so nothing is proven")
        return 2, {"templates": None, "surface": None}

    storage_state = "unchecked"
    storage_line = None
    present = False
    templates = client.list_templates_graphql()
    if templates is None:
        print("TEMPLATES: unreadable — the answer is UNKNOWN, not 'none exist'")
    else:
        print(f"TEMPLATES: {len(templates)} on the account")
        for t in templates:
            # containerDiskInGb decides whether a job can download a
            # checkpoint at all, and it is the number that settles whether a
            # candidate model is probeable on this endpoint or needs storage
            # it does not have. Printed because "the model did not fit on
            # disk" and "the model does not work" are different findings and
            # only one of them is about the model.
            print(f"  id={t.get('id')!r} name={t.get('name')!r} image={t.get('imageName')!r}"
                  f" containerDiskInGb={t.get('containerDiskInGb')!r}"
                  f" volumeInGb={t.get('volumeInGb')!r}"
                  f" volumeMountPath={t.get('volumeMountPath')!r}")
        ids = {t.get("id") for t in templates}
        if expected_template_id in ids:
            print(f"FOUND: {expected_template_id} exists after all")
            present = True
            storage_state, storage_line = _report_storage(client, expected_template_id)
        else:
            print(
                f"MISSING: {expected_template_id} is NOT among them — the endpoint's "
                "templateId is a dangling reference"
            )

    surface = client.rest_template_surface()
    print("=== REST surface (from the public OpenAPI document) ===")
    print(json.dumps(surface, indent=1, sort_keys=True))

    # A template can only point at an image that already exists. If the create
    # body takes an imageName and the API exposes no route that turns a
    # repository into an image, then creating a template here would produce a
    # SECOND dangling reference - the same broken state under a new id - and
    # the console's build integration is the only way to get an image at all.
    if surface:
        create = surface.get("template_create_body") or {}
        props = create.get("properties") or []
        required = create.get("required") or []
        print(f"CREATE REQUIRES: {required}")
        print(f"CREATE ACCEPTS: {props}")
        source_fields = [
            k for k in props
            if any(w in k.lower() for w in ("repo", "github", "git", "build", "source"))
        ]
        if source_fields:
            print(f"BUILD FIELD PRESENT: {source_fields}")
        else:
            print(
                "NO BUILD FIELD: the create body names an image, never a "
                "repository - a template cannot build one"
            )
        build_like = surface.get("build_like_paths")
        print(f"BUILD-LIKE PATHS ANYWHERE IN THE API: {build_like}")
    # THE VERDICT GOES LAST. Twice now the one line this job exists to
    # print has landed ABOVE a two-hundred-line schema dump, out of reach
    # of anyone reading the tail of a log.
    print("=== SUMMARY ===")
    print(f"TEMPLATE {expected_template_id}: {'present' if present else 'absent'}")
    if storage_line:
        print(storage_line)

    facts = {"templates": templates, "surface": surface, "storage": storage_state}
    if not surface or not surface.get("template_paths"):
        print(
            "NO TEMPLATE API: the spec exposes no template path, so creating one "
            "is a console action. Saying so beats probing with a real POST."
        )
        return 1, facts
    if storage_state in (False, None):
        print("NOT LAUNCH READY: the storage variables above are not confirmed set")
        return 1, facts
    return 0, facts


def main(argv) -> int:
    import runpod_client

    expected = argv[1] if len(argv) > 1 else "hhhdwtjw0y"
    code, _ = report(runpod_client, expected)
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
