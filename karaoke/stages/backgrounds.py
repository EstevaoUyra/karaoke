"""Stage 6 — backgrounds (default only). Owner: workstream E.

Input:  ``meta.json``.
Output: ``background.json`` (models.Background).

If background.json already exists (made by the user, the UI or the AI skill), do NOTHING.
Otherwise write the default: one scene covering ``0..meta.duration`` with a dark gradient
(``style = "default"``). The art-directed version is the ``karaoke-backgrounds`` skill, not this stage.
"""

from karaoke.pipeline import StageContext


def run(ctx: StageContext) -> None:
    raise NotImplementedError("backgrounds: Phase 1, workstream E")
