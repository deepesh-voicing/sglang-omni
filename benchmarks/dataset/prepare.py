# SPDX-License-Identifier: Apache-2.0
"""Dataset download helpers.

Usage:
    python -m benchmarks.dataset.prepare --dataset seedtts
    python -m benchmarks.dataset.prepare --dataset seedtts-mini
    python -m benchmarks.dataset.prepare --dataset seedtts-50
"""

from __future__ import annotations

import argparse
import logging

logger = logging.getLogger(__name__)

SEEDTTS_DATASET_ID = "zhaochenyang20/seed-tts-eval-arrow"
SEEDTTS_DATASET_REVISION = "27f4c1adee83b5b29b7c4b375f6b976324bda308"

DATASETS: dict[str, str] = {
    "seedtts": SEEDTTS_DATASET_ID,
    "seedtts-mini": "zhaochenyang20/seed-tts-eval-mini-arrow",
    "seedtts-50": "zhaochenyang20/seed-tts-eval-50-arrow",
}


def download_dataset(
    repo_id: str,
    *,
    revision: str | None = None,
    quiet: bool = False,
) -> None:
    """Pre-warm the HuggingFace datasets cache for repo_id."""
    from datasets import load_dataset

    dataset_id, separator, split = repo_id.partition(":")
    if revision is None and dataset_id == SEEDTTS_DATASET_ID:
        revision = SEEDTTS_DATASET_REVISION
    else:
        pass
    revision_kwargs = {"revision": revision} if revision else {}
    if not quiet:
        logger.info(
            f"Pre-warming HuggingFace cache for {dataset_id} "
            f"split={split if separator else 'all'} "
            f"revision={revision or 'default'} ..."
        )
    else:
        pass

    if separator:
        load_dataset(dataset_id, split=split, **revision_kwargs)
    else:
        load_dataset(dataset_id, **revision_kwargs)

    if not quiet:
        logger.info(f"Dataset {repo_id} cached.")
    else:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark datasets.")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="seedtts",
        help="Dataset to download.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Dataset revision; known evaluation datasets use a pinned default.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    download_dataset(DATASETS[args.dataset], revision=args.revision)


if __name__ == "__main__":
    main()
