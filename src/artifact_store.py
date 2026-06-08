"""Tiny persistence helpers for artifacts (JSON, pickle, manifest, append-only CSV).

Centralises reading/writing of the files the pipeline produces. :func:`append_rows_to_csv`
is schema-tolerant: it migrates an existing CSV's header when new columns appear, so
appending newer report rows never corrupts older files.
"""
from __future__ import annotations

import json
import pickle
from csv import DictWriter, reader
from pathlib import Path
from typing import Any


def load_json_if_exists(path: Path):
    """Load JSON from ``path``, or return None if the file does not exist."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_json(path: Path, obj: Any):
    """Write ``obj`` as indented JSON, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_pickle(path: Path, obj: Any):
    """Pickle ``obj`` to ``path``, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle_if_exists(path: Path):
    """Unpickle from ``path``, or return None if the file does not exist."""
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def save_manifest(path: Path, manifest: dict):
    """Persist an experiment manifest (JSON) used for cache-compatibility checks."""
    save_json(path, manifest)


def append_rows_to_csv(path: Path, rows: list[dict]):
    """Append dict rows to a CSV, migrating the header if new columns appear."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming_fieldnames: list[str] = []
    for row in rows:
        for fieldname in row.keys():
            if fieldname not in incoming_fieldnames:
                incoming_fieldnames.append(fieldname)

    fieldnames = incoming_fieldnames
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        migrated_rows = None
        with open(path, "r", encoding="utf-8", newline="") as f:
            csv_reader = reader(f)
            existing_fieldnames = next(csv_reader, [])
            missing_fieldnames = [name for name in incoming_fieldnames if name not in existing_fieldnames]
            if missing_fieldnames:
                if set(existing_fieldnames).issubset(incoming_fieldnames):
                    fieldnames = incoming_fieldnames
                else:
                    fieldnames = existing_fieldnames + missing_fieldnames

                migrated_rows = []
                for values in csv_reader:
                    if not values:
                        continue
                    source_fieldnames = fieldnames if len(values) == len(fieldnames) else existing_fieldnames
                    migrated_rows.append({
                        source_fieldnames[i]: value
                        for i, value in enumerate(values)
                        if i < len(source_fieldnames)
                    })
            else:
                fieldnames = existing_fieldnames

        if migrated_rows is not None:
            with open(path, "w", encoding="utf-8", newline="") as f:
                writer = DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(migrated_rows)

    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
