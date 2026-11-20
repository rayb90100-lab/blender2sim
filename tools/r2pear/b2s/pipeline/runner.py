"""Stage runner for the b2s conversion pipeline."""

from pathlib import Path

from .models import PipelineContext, PipelineOptions
from .stages import (
    build_demo,
    identify_embodiment,
    load_context,
    parse_scene,
    run_environment_tests,
    run_speed_diagnosis,
    validate,
    write_spec,
)


def run_pipeline(options: PipelineOptions) -> PipelineContext:
    """Run the stable stage order and return all intermediate results."""
    ctx = load_context(options)
    identify_embodiment(ctx)
    parse_scene(ctx)
    build_demo(ctx)
    write_spec(ctx)
    validate(ctx)
    run_environment_tests(ctx)
    # Timing diagnosis remains last and never replaces ordinary validation.
    run_speed_diagnosis(ctx)
    web_ui = Path(__file__).resolve().parents[2] / "web_ui.py"
    print(f"\nWeb UI:  MUJOCO_GL=egl {__import__('common').PY} {web_ui} "
          f"--testdir {options.out} --port 8112")
    return ctx
