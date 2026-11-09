"""Stage 6 — backgrounds (default only). Owner: workstream E.

Input:  ``meta.json``.
Output: ``background.json`` (models.Background).

If background.json already exists (made by the user, the UI or the AI skill), do NOTHING.
Otherwise write the default: one scene covering ``0..meta.duration`` with a dark gradient
(``style = "default"``). The art-directed version is the ``karaoke-backgrounds`` skill, not this stage.
"""

from karaoke.models import Background, BgSource, Meta, Scene, Transition
from karaoke.pipeline import StageContext
from karaoke.store import read_model, write_model

DEFAULT_GRADIENT = "#0b0b1a,#2a1f4a"


def default_background(duration: float) -> Background:
    return Background(
        style="default",
        scenes=[Scene(
            id="s1", start=0.0, end=round(duration, 3),
            source=BgSource(type="gradient", value=DEFAULT_GRADIENT),
            transition=Transition(type="cut", duration=0.0),
            reason="default background",
        )],
    )


def run(ctx: StageContext) -> None:
    p = ctx.paths
    if p.background.exists():
        ctx.log.info("background.json already exists; leaving it untouched")
        ctx.progress(1.0, "kept existing background.json")
        return
    meta = read_model(p.meta, Meta)
    write_model(p.background, default_background(meta.duration))
    ctx.progress(1.0, "wrote default background.json")
