from __future__ import annotations

from pathlib import Path

import torch

from models import UltraLiteDCE
from utils.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_save_and_restore(tmp_path: Path) -> None:
    source = UltraLiteDCE()
    optimizer = torch.optim.Adam(source.parameters(), lr=1e-4)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        {
            "epoch": 2,
            "model": source.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {"model": {"width": 8}},
        },
        path,
    )
    restored = UltraLiteDCE()
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-4)
    checkpoint = load_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
    )
    assert checkpoint["epoch"] == 2
    for expected, actual in zip(source.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)

