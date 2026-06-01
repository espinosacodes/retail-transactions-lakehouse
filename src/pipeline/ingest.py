"""Ingesta incremental (RF-8).

Detecta archivos nuevos en `data/landing/` y dispara el pipeline completo
(bronze → silver → gold → models). El bronze se ejecuta en modo `overwrite`,
de modo que reprocesar todos los archivos juntos garantiza idempotencia y evita
duplicados. Esta es la decisión de diseño explícita del documento de arquitectura:
"Append-only en Bronze, overwrite por partición en Silver/Gold".

Mecanismo:
- Se mantiene un manifest en `data/landing/_manifest.json` con `nombre → sha256`.
- En cada invocación se calcula el sha256 de todos los CSV bajo
  `data/landing/Transactions/` y `data/landing/Products/`.
- Si alguno cambió o es nuevo, se relanza el pipeline y se actualiza el manifest.
- Si no hay novedades, se reporta y se termina sin reprocesar.

Comandos CLI:
    python -m src.pipeline.ingest --check        # solo reporta qué cambió
    python -m src.pipeline.ingest --run          # ejecuta si hay cambios
    python -m src.pipeline.ingest --force        # ejecuta aunque no haya cambios
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from . import bronze, gold, models, silver
from .paths import LANDING, LANDING_PRODUCTS, LANDING_TX


MANIFEST = LANDING / "_manifest.json"
RUNS_LOG = LANDING / "_runs.jsonl"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _scan() -> Dict[str, str]:
    out: Dict[str, str] = {}
    for d in (LANDING_TX, LANDING_PRODUCTS):
        if not d.exists():
            continue
        for p in sorted(d.glob("*.csv")):
            rel = str(p.relative_to(LANDING))
            out[rel] = _sha256(p)
    return out


def _load_manifest() -> Dict[str, str]:
    if not MANIFEST.exists():
        return {}
    try:
        return json.loads(MANIFEST.read_text())
    except json.JSONDecodeError:
        return {}


def _save_manifest(m: Dict[str, str]) -> None:
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True))


def diff(current: Dict[str, str], previous: Dict[str, str]) -> Tuple[List[str], List[str], List[str]]:
    """Devuelve (nuevos, modificados, eliminados) por nombre relativo."""
    new = [k for k in current if k not in previous]
    changed = [k for k in current if k in previous and current[k] != previous[k]]
    removed = [k for k in previous if k not in current]
    return new, changed, removed


def _log_run(payload: dict) -> None:
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUNS_LOG.open("a") as f:
        f.write(json.dumps(payload) + "\n")


def _run_pipeline(skip_models: bool = False) -> Dict[str, float]:
    """Ejecuta el pipeline secuencial y devuelve los tiempos por etapa."""
    timings: Dict[str, float] = {}
    steps = [("bronze", bronze.run), ("silver", silver.run), ("gold", gold.run)]
    if not skip_models:
        steps.append(("models", models.run))
    for name, fn in steps:
        t0 = time.perf_counter()
        print(f"\n=== ingest: running {name} ===")
        fn()
        dt = time.perf_counter() - t0
        timings[name] = round(dt, 2)
        print(f"=== {name} done in {dt:.1f}s ===")
    return timings


def check() -> dict:
    current = _scan()
    previous = _load_manifest()
    new, changed, removed = diff(current, previous)
    report = {
        "files_seen": len(current),
        "new": new,
        "changed": changed,
        "removed": removed,
        "needs_run": bool(new or changed or removed),
    }
    print(json.dumps(report, indent=2))
    return report


def ingest(force: bool = False, skip_models: bool = False) -> dict:
    current = _scan()
    previous = _load_manifest()
    new, changed, removed = diff(current, previous)

    will_run = force or new or changed or removed
    started_at = datetime.utcnow().isoformat() + "Z"

    if not will_run:
        msg = "[ingest] no hay archivos nuevos ni modificados; pipeline omitido."
        print(msg)
        return {"started_at": started_at, "ran": False, "new": [], "changed": [], "removed": []}

    print(f"[ingest] cambios detectados — nuevos={new}, modificados={changed}, eliminados={removed}")
    timings = _run_pipeline(skip_models=skip_models)
    _save_manifest(current)

    payload = {
        "started_at": started_at,
        "finished_at": datetime.utcnow().isoformat() + "Z",
        "ran": True,
        "forced": force,
        "skip_models": skip_models,
        "new": new,
        "changed": changed,
        "removed": removed,
        "timings_s": timings,
        "files_count": len(current),
    }
    _log_run(payload)
    print(f"[ingest] OK — {payload['files_count']} archivos procesados")
    return payload


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingesta incremental")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="Reporta cambios sin ejecutar")
    g.add_argument("--run", action="store_true", help="Ejecuta si hay cambios")
    g.add_argument("--force", action="store_true", help="Ejecuta aunque no haya cambios")
    parser.add_argument("--skip-models", action="store_true",
                        help="Omite la etapa de modelos (útil para iterar rápido)")
    return parser


def main():
    args = _build_argparser().parse_args()
    if args.check:
        check()
    elif args.force:
        ingest(force=True, skip_models=args.skip_models)
    else:
        # Por defecto: ejecutar si hay cambios.
        ingest(force=False, skip_models=args.skip_models)


if __name__ == "__main__":
    main()
