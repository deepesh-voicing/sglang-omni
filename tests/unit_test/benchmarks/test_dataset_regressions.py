from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from benchmarks.dataset import prepare, seedtts


class FakeDataset:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.column_names = list(rows[0].keys()) if rows else []
        self.selected_indices: list[int] | None = None

    def cast_column(self, _name: str, _audio_spec) -> "FakeDataset":
        return self

    def select(self, indices: list[int]) -> "FakeDataset":
        self.selected_indices = list(indices)
        return FakeDataset([self.rows[i] for i in indices])

    def rename_columns(self, aliases: dict[str, str]) -> "FakeDataset":
        return FakeDataset(
            [
                {aliases.get(key, key): value for key, value in row.items()}
                for row in self.rows
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)


def test_download_seedtts_uses_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}

    def fake_load_dataset(repo_id: str, **kwargs):
        observed["repo_id"] = repo_id
        observed.update(kwargs)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            get_dataset_config_names=lambda *args, **_kwargs: [],
            load_dataset=fake_load_dataset,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=lambda *args, **_kwargs: None),
    )

    prepare.download_dataset(prepare.SEEDTTS_DATASET_ID, quiet=True)

    assert observed == {
        "repo_id": prepare.SEEDTTS_DATASET_ID,
        "revision": prepare.SEEDTTS_DATASET_REVISION,
    }


def test_load_seedtts_samples_accepts_local_meta_lst(tmp_path: Path) -> None:
    meta_dir = tmp_path / "en"
    meta_dir.mkdir()
    ref_audio = meta_dir / "ref.wav"
    ref_audio.write_bytes(b"wav")
    meta_path = meta_dir / "meta.lst"
    meta_path.write_text(
        "sample-1|hello|ref.wav|target one\nsample-2|world|ref.wav|target two\n"
    )

    samples = seedtts.load_seedtts_samples(str(meta_path), max_samples=1)

    assert len(samples) == 1
    assert samples[0].sample_id == "sample-1"
    assert samples[0].ref_text == "hello"
    assert samples[0].ref_audio == str(ref_audio)
    assert samples[0].target_text == "target one"


def test_load_seedtts_samples_stages_only_selected_rows(
    monkeypatch, tmp_path: Path
) -> None:
    seedtts._STAGED_CACHE.clear()  # noqa: leading-underscore  # production name

    rows = [
        {
            "sample_id": f"sample-{idx}",
            "ref_text": f"ref-{idx}",
            "ref_audio_path": f"audio/{idx}.wav",
            "target_text": f"target-{idx}",
            "ref_audio": {"bytes": f"audio-{idx}".encode()},
        }
        for idx in range(5)
    ]
    dataset = FakeDataset(rows)
    stage_dir = tmp_path / "seedtts_stage"
    stage_dir.mkdir()

    def fake_load_dataset(repo_id: str, split: str, revision: str | None = None):
        assert repo_id == "zhaochenyang20/seed-tts-eval-arrow"
        assert split == "en"
        assert revision == prepare.SEEDTTS_DATASET_REVISION
        return dataset

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            Audio=lambda **kwargs: ("Audio", kwargs),
            load_dataset=fake_load_dataset,
        ),
    )
    monkeypatch.setattr(seedtts.tempfile, "mkdtemp", lambda prefix: str(stage_dir))
    monkeypatch.setattr(seedtts.atexit, "register", lambda *args, **kwargs: None)

    samples = seedtts.load_seedtts_samples(
        "zhaochenyang20/seed-tts-eval-arrow",
        max_samples=2,
        split="en",
    )

    assert dataset.selected_indices == [0, 1]
    assert [sample.sample_id for sample in samples] == ["sample-0", "sample-1"]
    assert sorted(
        path.relative_to(stage_dir).as_posix() for path in stage_dir.rglob("*.wav")
    ) == [
        "audio/0.wav",
        "audio/1.wav",
    ]

    seedtts._STAGED_CACHE.clear()  # noqa: leading-underscore  # production name


@pytest.mark.parametrize(
    ("ref_audio_path", "outside_name"),
    [
        ("../escape.wav", "escape.wav"),
        (None, "absolute.wav"),
    ],
)
def test_load_seedtts_samples_rejects_unsafe_audio_paths(
    monkeypatch, tmp_path: Path, ref_audio_path: str | None, outside_name: str
) -> None:
    seedtts._STAGED_CACHE.clear()  # noqa: leading-underscore  # production name

    stage_dir = tmp_path / "seedtts_stage"
    stage_dir.mkdir()
    outside_path = tmp_path / outside_name
    rows = [
        {
            "sample_id": "sample-0",
            "ref_text": "ref-0",
            "ref_audio_path": (
                ref_audio_path if ref_audio_path is not None else str(outside_path)
            ),
            "target_text": "target-0",
            "ref_audio": {"bytes": b"audio-0"},
        }
    ]

    def fake_load_dataset(repo_id: str, split: str, revision: str):
        assert repo_id == prepare.SEEDTTS_DATASET_ID
        assert split == "en"
        assert revision == prepare.SEEDTTS_DATASET_REVISION
        return FakeDataset(rows)

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            Audio=lambda **kwargs: ("Audio", kwargs),
            load_dataset=fake_load_dataset,
        ),
    )
    monkeypatch.setattr(seedtts.tempfile, "mkdtemp", lambda prefix: str(stage_dir))
    monkeypatch.setattr(seedtts.atexit, "register", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="Invalid ref_audio_path"):
        seedtts.load_seedtts_samples(
            "zhaochenyang20/seed-tts-eval-arrow",
            max_samples=1,
            split="en",
        )

    assert not outside_path.exists()
    assert list(stage_dir.rglob("*.wav")) == []

    seedtts._STAGED_CACHE.clear()  # noqa: leading-underscore  # production name


def install_fake_datasets(monkeypatch: pytest.MonkeyPatch, load_dataset) -> None:
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(
            Audio=lambda **kwargs: ("Audio", kwargs),
            load_dataset=load_dataset,
            get_dataset_config_names=lambda repo_id: pytest.fail(
                f"Unexpected config enumeration for {repo_id}"
            ),
        ),
    )
