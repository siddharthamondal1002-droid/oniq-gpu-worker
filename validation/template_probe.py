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


def report(client, expected_template_id: str) -> tuple:
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
        else:
            print(
                f"MISSING: {expected_template_id} is NOT among them — the endpoint's "
                "templateId is a dangling reference"
            )

    surface = client.rest_template_surface()
    print("=== REST surface (from the public OpenAPI document) ===")
    print(json.dumps(surface, indent=1, sort_keys=True))
    if not surface or not surface.get("template_paths"):
        print(
            "NO TEMPLATE API: the spec exposes no template path, so creating one "
            "is a console action. Saying so beats probing with a real POST."
        )
        return 1, {"templates": templates, "surface": surface}
    return 0, {"templates": templates, "surface": surface}


def main(argv) -> int:
    import runpod_client

    expected = argv[1] if len(argv) > 1 else "hhhdwtjw0y"
    code, _ = report(runpod_client, expected)
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv))
