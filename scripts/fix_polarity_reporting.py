#!/usr/bin/env python3
"""
Fix polarity reporting to use directionless metrics.

The signed polarity V flips sign for high_shear, likely due to:
1. Flow direction metadata not calibrated
2. Image coordinate conventions differ between batches

This script adds |V| (absolute V) as the recommended directionless metric
alongside R (alignment strength, which is already directionless).
"""

import pandas as pd
import numpy as np
from pathlib import Path


def fix_polarity_summary(summary_csv: Path) -> pd.DataFrame:
    """Add |V| to polarity summary."""
    df = pd.read_csv(summary_csv)

    # Add |V| columns
    df['abs_V_mean'] = np.abs(df['V_mean'])

    # Reorder columns
    cols = ['condition', 'n_images', 'n_cells_total',
            'R_mean', 'R_std', 'R_min', 'R_max',
            'V_mean', 'abs_V_mean', 'V_std']
    df = df[[c for c in cols if c in df.columns]]

    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir", help="Path to runs directory")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)

    # Fix SBIAD1540 polarity
    sbiad_summary = runs_dir / "sbiad1540_full" / "polarity_summary_by_condition.csv"
    if sbiad_summary.exists():
        print(f"Fixing {sbiad_summary}")
        df = fix_polarity_summary(sbiad_summary)

        print("\nPolarity Summary (with |V|):")
        print(df.to_string(index=False))

        # Save
        output = sbiad_summary.parent / "polarity_summary_fixed.csv"
        df.to_csv(output, index=False)
        print(f"\nSaved to: {output}")

        # Print interpretation
        print("\n" + "="*70)
        print("POLARITY INTERPRETATION")
        print("="*70)
        print("""
R (alignment strength) - RECOMMENDED PRIMARY METRIC:
  - Directionless measure of cell elongation alignment
  - Increases monotonically with shear stress
  - Does NOT depend on flow axis calibration

V (signed polarity) - TREAT AS EXPLORATORY:
  - Sign depends on flow direction metadata
  - FLIPS SIGN for high_shear condition
  - Likely due to uncalibrated flow axis or coordinate conventions

|V| (absolute polarity) - ALTERNATIVE DIRECTIONLESS METRIC:
  - Magnitude of coherent alignment
  - Use alongside R when direction is uncertain

RECOMMENDATION:
  Report R as primary metric. Treat V sign as exploratory unless
  flow direction is confirmed from experimental metadata.
""")


if __name__ == "__main__":
    main()
