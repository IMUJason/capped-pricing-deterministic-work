from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from pathlib import Path


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_listing_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        digest.update(str(child.relative_to(path)).encode("utf-8"))
        if child.is_file():
            digest.update(str(child.stat().st_size).encode("utf-8"))
    return digest.hexdigest()


def parse_dimacs_script_line(line: str) -> dict[str, object] | None:
    pattern = re.compile(
        r"Instances/(?P<family>Solomon|Homberger)/(?P<instance>[A-Za-z0-9_]+\.txt)\s+\$2\s+(?P<time_limit>\d+)\s+(?P<bks>[0-9.]+)\s+(?P<optimal>[01])\s+\$3"
    )
    match = pattern.search(line)
    if not match:
        return None
    return {
        "family": match.group("family"),
        "instance_name": Path(match.group("instance")).stem,
        "relative_path": f"Instances/{match.group('family')}/{match.group('instance')}",
        "time_limit_seconds": int(match.group("time_limit")),
        "best_known_solution": float(match.group("bks")),
        "is_optimal": bool(int(match.group("optimal"))),
    }


def prepare_vrptw_assets(project_root: Path) -> dict[str, object]:
    archive_path = project_root / "data" / "raw" / "vrptw" / "VRPTWController-master.zip"
    extract_root = project_root / "data" / "raw" / "vrptw" / "controller"
    provenance_root = project_root / "data" / "provenance"
    provenance_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(extract_root)

        bks_rows = []
        for script_name in ("genScript1.sh", "genScript2.sh", "genScript3.sh"):
            with archive.open(f"VRPTWController-master/{script_name}") as handle:
                for raw_line in handle.read().decode("utf-8").splitlines():
                    parsed = parse_dimacs_script_line(raw_line)
                    if parsed is not None:
                        parsed["script_name"] = script_name
                        bks_rows.append(parsed)

    bks_rows.sort(key=lambda row: (str(row["family"]), str(row["instance_name"])))
    bks_csv = provenance_root / "vrptw_dimacs_bks.csv"
    with bks_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["family", "instance_name", "relative_path", "time_limit_seconds", "best_known_solution", "is_optimal", "script_name"],
        )
        writer.writeheader()
        writer.writerows(bks_rows)

    manifest = {
        "source_type": "archive",
        "archive_path": str(archive_path),
        "archive_sha256": sha256_of_file(archive_path),
        "archive_origin": {
            "dimacs_page": "http://dimacs.rutgers.edu/programs/challenge/vrp/vrptw/",
            "controller_repository": "https://github.com/laser-ufpb/VRPTWController",
            "download_url": "https://github.com/laser-ufpb/VRPTWController/archive/refs/heads/master.zip",
        },
        "extracted_root": str(extract_root / "VRPTWController-master" / "Instances"),
        "bks_csv": str(bks_csv),
        "instance_counts": {
            "Solomon": len([row for row in bks_rows if row["family"] == "Solomon"]),
            "Homberger": len([row for row in bks_rows if row["family"] == "Homberger"]),
        },
    }
    manifest_path = provenance_root / "vrptw_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def index_local_miplib(project_root: Path) -> dict[str, object]:
    provenance_root = project_root / "data" / "provenance"
    provenance_root.mkdir(parents=True, exist_ok=True)

    benchmark_root = Path(project_root / "data" / "external" / "miplib_benchmark").resolve()
    collection_root = Path(project_root / "data" / "external" / "miplib_collection").resolve()
    solu_path = collection_root / "miplib2017-v36.solu"
    testscript_path = collection_root / "miplib2017-testscript-v1"

    def make_index(root: Path) -> list[dict[str, object]]:
        rows = []
        for path in sorted(root.glob("*.mps.gz")):
            rows.append(
                {
                    "file_name": path.name,
                    "absolute_path": str(path),
                    "size_bytes": path.stat().st_size,
                }
            )
        return rows

    benchmark_rows = make_index(benchmark_root)
    collection_rows = make_index(collection_root)

    for name, rows in (("benchmark", benchmark_rows), ("collection", collection_rows)):
        output_path = provenance_root / f"miplib_{name}_index.csv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["file_name", "absolute_path", "size_bytes"])
            writer.writeheader()
            writer.writerows(rows)

    manifest = {
        "benchmark_root": str(benchmark_root),
        "collection_root": str(collection_root),
        "benchmark_instance_count": len(benchmark_rows),
        "collection_instance_count": len(collection_rows),
        "solu_path": str(solu_path),
        "solu_sha256": sha256_of_file(solu_path) if solu_path.exists() else None,
        "testscript_path": str(testscript_path),
        "testscript_is_directory": testscript_path.is_dir(),
        "testscript_sha256": sha256_of_file(testscript_path) if testscript_path.is_file() else None,
        "testscript_listing_fingerprint": directory_listing_fingerprint(testscript_path) if testscript_path.is_dir() else None,
    }
    manifest_path = provenance_root / "miplib_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
