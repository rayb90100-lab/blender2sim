"""State and options shared by the b2s pipeline stages."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class PipelineOptions:
    """All command-line choices, kept separate from stage implementations."""

    blend: Path
    out: Path
    config: Optional[Path] = None
    videos: Optional[Path] = None
    reextract: bool = False
    skip_validate: bool = False
    env_tests: bool = False
    n_init: int = 50
    last_resort_speed_diagnosis: Optional[str] = None

    def validate(self) -> None:
        if self.last_resort_speed_diagnosis is not None:
            if self.skip_validate:
                raise ValueError("速度诊断需要本轮常规验证，不能与 --skip-validate 并用")
            if not self.last_resort_speed_diagnosis.strip():
                raise ValueError("速度诊断说明不能为空")
    @property
    def blend_dir(self) -> Path:
        return self.blend.resolve().parent

    @property
    def extract_dir(self) -> Path:
        return self.out / "stage0_extract"

    @property
    def spec_dir(self) -> Path:
        return self.out / "stage0_spec"

    @property
    def extract_json(self) -> Path:
        return self.extract_dir / "extract.json"

    @property
    def spec_json(self) -> Path:
        return self.spec_dir / "task_spec.json"


@dataclass
class PipelineContext:
    """Mutable hand-off object passed between stages."""

    options: PipelineOptions
    cfg: dict[str, Any]
    extract: Any = None
    decisions: Any = None
    embodiment: Any = None
    robot: Any = None
    scene: Any = None
    demo: Any = None
    spec: Any = None
    registration_report: Any = None
    q0: Any = None
    outputs: dict[str, Path] = field(default_factory=dict)
