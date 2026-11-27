import sys
from pathlib import Path

import pandas as pd
from statistical_analysis_report import plot_cold_effect


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))



def test_plot_cold_effect_accepts_block_effects_dataframe(tmp_path: Path) -> None:
    results = {
        "block_effects": pd.DataFrame(
            [
                {
                    "block": 1,
                    "p_value": 0.01,
                    "log_crc_cold_minus_warm": 0.2,
                    "ci_low": 0.1,
                    "ci_high": 0.3,
                },
                {
                    "block": 2,
                    "p_value": 0.6,
                    "log_crc_cold_minus_warm": -0.1,
                    "ci_low": -0.2,
                    "ci_high": 0.1,
                },
            ]
        )
    }

    plot_cold_effect(results, tmp_path)

    assert (tmp_path / "CRC_cold_effects.png").exists()
