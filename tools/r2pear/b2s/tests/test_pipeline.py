"""Small contract tests for the orchestration layer."""

from pathlib import Path

import pytest

from pipeline.models import PipelineOptions


def test_pipeline_options_expose_stable_stage_paths():
    options = PipelineOptions(Path("scene.blend"), Path("out"))
    assert options.extract_json == Path("out/stage0_extract/extract.json")
    assert options.spec_json == Path("out/stage0_spec/task_spec.json")


def test_speed_diagnosis_requires_normal_validation():
    options = PipelineOptions(
        Path("scene.blend"), Path("out"),
        skip_validate=True, last_resort_speed_diagnosis="reviewed",
    )
    with pytest.raises(ValueError, match="常规验证"):
        options.validate()
