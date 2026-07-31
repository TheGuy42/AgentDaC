from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from src.trajectory import Trajectory
from src.utils.io import save_object
from src.utils.logging import create_logger


logger = create_logger(__name__)


class TrajectoryWriter:
    """
    Writes full rollout trajectories to disk as readable JSON, grouped by training step.

    Layout: `{output_dir}/step_{step:05d}/{stage}/{rollout_id}.json`

    One file per trajectory means concurrent writers (e.g. multiple runner processes)
    never contend on the same file, so no synchronization is needed beyond the filesystem.
    The instance only holds a path, so it is cheap to pickle into worker processes.
    """

    def __init__(self, output_dir: str | Path, enabled: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.enabled = enabled

    def stage_dir(self, stage: str, step: int | None) -> Path:
        """Directory holding one stage's trajectories for one step."""
        step_dir = f"step_{step:05d}" if step is not None else "step_unknown"
        return self.output_dir / step_dir / stage

    def write(self, trajectory: Trajectory, *, rollout_id: str, stage: str, step: int | None) -> Path:
        if not self.enabled:
            logger.debug("Trajectory writing is disabled; skipping write.")
            return Path()

        path = self.stage_dir(stage, step) / f"{rollout_id}.json"

        payload = {
            "rollout_id": rollout_id,
            "stage": stage,
            "step": step,
            "written_at": datetime.now().isoformat(),
            **trajectory.for_logging(),
        }

        # exist_ok guards against concurrent processes creating the same step/stage dir
        path.parent.mkdir(parents=True, exist_ok=True)
        save_object(payload, path, overwrite=True)
        return path

    async def write_async(self, trajectory: Trajectory, *, rollout_id: str, stage: str, step: int | None) -> Path:
        """Write without blocking the caller's event loop."""
        return await asyncio.to_thread(self.write, trajectory, rollout_id=rollout_id, stage=stage, step=step)
