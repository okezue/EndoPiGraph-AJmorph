from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import tifffile

from .config import load_config
from .download import download_study
from .interfaces import extract_interfaces
from .io import read_image
from .manifest import make_manifest
from .ml import train_ajmorph_classifier
from .patches import export_interface_patches
from .pipeline import run_pipeline


def cmd_download(args: argparse.Namespace) -> int:
    download_study(accession=args.accession, out_dir=args.out, method=args.method, run_wget=args.run_wget)
    return 0


def cmd_make_manifest(args: argparse.Namespace) -> int:
    make_manifest(root=args.input, out_csv=args.out)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    run_pipeline(args.config)
    return 0


def cmd_export_patches(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"])
    qc_dir = out_dir / "patches"
    qc_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(cfg["manifest_csv"])
    if cfg.get("qc", {}).get("max_images"):
        manifest = manifest.head(int(cfg["qc"]["max_images"]))

    # Marker type to export (default: AJ)
    marker_type = args.marker_type
    if marker_type not in cfg.get("junction_markers", {}):
        raise SystemExit(f"marker_type {marker_type!r} not found in config junction_markers")

    marker_cfg = cfg["junction_markers"][marker_type]

    for _, row in manifest.iterrows():
        image_id = str(row["image_id"])
        img_path = str(row["path"])
        labels_path = out_dir / "masks" / f"{image_id}_labels.tif"
        if not labels_path.exists():
            print(f"[skip] missing labels for {image_id}: {labels_path}")
            continue

        arr, ch_names = read_image(img_path)
        ch_idx = None
        ch = marker_cfg.get("channel")
        if isinstance(ch, int):
            ch_idx = ch
        elif isinstance(ch, str):
            if ch in ch_names:
                ch_idx = ch_names.index(ch)
            else:
                # allow partial match
                lowered = [c.lower() for c in ch_names]
                if ch.lower() in lowered:
                    ch_idx = lowered.index(ch.lower())
        if ch_idx is None:
            raise SystemExit(
                f"Could not resolve marker channel for {marker_type}. Available channels: {ch_names}"
            )
        marker = arr[ch_idx]

        labels = tifffile.imread(str(labels_path))
        interfaces = extract_interfaces(labels)

        edges_df = interfaces.edges
        if args.max_edges:
            edges_df = edges_df.sample(n=min(args.max_edges, len(edges_df)), random_state=0)

        export_interface_patches(
            image_id=image_id,
            marker=marker,
            edges=edges_df,
            boundary_coords=interfaces.boundary_coords,
            out_dir=qc_dir,
            patch_size=args.patch_size,
            max_patches=args.max_patches,
        )

    print(f"Wrote patches to: {qc_dir}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    train_ajmorph_classifier(
        features_csv=args.features,
        labels_csv=args.labels,
        out_dir=args.out,
        label_col=args.label_col,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="endopigraph",
        description="EndoPiGraph-AJmorph v1: build typed endothelial cell contact graphs and AJ morphology features.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sub = subparsers.add_parser("download", help="Download a BioImage Archive accession")
    sub.add_argument("accession", help="BioImage Archive accession, e.g. S-BIAD1540")
    sub.add_argument("--out", default="data/raw", help="Output directory (default: data/raw)")
    sub.add_argument(
        "--method",
        default="print",
        choices=["print", "wget"],
        help="Download method: print commands or attempt wget",
    )
    sub.add_argument(
        "--run-wget",
        action="store_true",
        help="If method=wget, actually execute wget (otherwise just print the command)",
    )
    sub.set_defaults(func=cmd_download)

    sub = subparsers.add_parser("make-manifest", help="Scan a directory and write a manifest.csv")
    sub.add_argument("--input", required=True, help="Root directory containing image files")
    sub.add_argument("--out", required=True, help="Output CSV path")
    sub.set_defaults(func=cmd_make_manifest)

    sub = subparsers.add_parser("run", help="Run the full pipeline")
    sub.add_argument("--config", required=True, help="YAML config path")
    sub.set_defaults(func=cmd_run)

    sub = subparsers.add_parser("export-patches", help="Export interface crops for manual AJmorph labeling")
    sub.add_argument("--config", required=True, help="YAML config path")
    sub.add_argument("--marker-type", default="AJ", help="Which junction marker key to export (default: AJ)")
    sub.add_argument("--patch-size", type=int, default=96, help="Patch size in pixels (square)")
    sub.add_argument("--max-patches", type=int, default=300, help="Max patches per image")
    sub.add_argument("--max-edges", type=int, default=None, help="Optional cap: sample up to N edges per image")
    sub.set_defaults(func=cmd_export_patches)

    sub = subparsers.add_parser("train-ajmorph", help="Train a classifier from labeled interface patches")
    sub.add_argument("--features", required=True, help="Edge features CSV (from outputs/tables/*_edges.csv or merged)")
    sub.add_argument("--labels", required=True, help="CSV with columns: image_id, cell_i, cell_j, aj_label")
    sub.add_argument("--out", required=True, help="Output directory for model + metrics")
    sub.add_argument("--label-col", default="aj_label", help="Name of label column in labels CSV")
    sub.set_defaults(func=cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
