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


def _report_storage(client, template_id: str):
    """Which storage variables are set on the template. NAMES ONLY.

    A RunPod serverless worker reads its environment from its TEMPLATE, so
    "the secrets are added" is a claim about this object and nowhere else -
    not about the endpoint, and certainly not about a GitHub secret. Checked
    here because a job that cannot write its output should never have been
    started, and because the owner is entitled to see the claim verified
    rather than assumed.

    Unreadable is UNKNOWN, never "ready". Returns True, False or None.
    """
    names = client.template_env_names_graphql(template_id)
    if names is None:
        print("ENV: unreadable - the answer is UNKNOWN, not 'nothing is set'")
        return None
    print(
        f"ENV ON {template_id}: {len(names)} set - "
        f"{', '.join(sorted(names)) or '(none)'}  (names only, never values)"
    )
    absent = [name for name in storage.REQUIRED_VARS if name not in names]
    if absent:
        print(
            f"STORAGE NOT READY: {', '.join(absent)} not set - every job would "
            "fail closed with storage-not-configured"
        )
        return False
    print("STORAGE READY: every variable storage.py requires is set on the template")
    return True


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
    templates = client.list_templates_graphql()
    if templates is None:
        print("TEMPLATES: unreadable — the answer is UNKNOWN, not 'none exist'")
    else:
        print(f"TEMPLATES: {len(templates)} on the account")
        for t in templates:
            print(f"  id={t.get('id')!r} name={t.get('name')!r} image={t.get('imageName')!r}")
        ids = {t.get("id") for t in templates}
        if expected_template_id in ids:
            print(f"FOUND: {expected_template_id} exists after all")
            storage_state = _report_storage(client, expected_template_id)
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
