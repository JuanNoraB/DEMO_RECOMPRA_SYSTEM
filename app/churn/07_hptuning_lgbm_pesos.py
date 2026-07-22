"""Corrección del HPT LightGBM de churn: scale_pos_weight como hiperparámetro.

Este script reutiliza toda la infraestructura de ``07_hptuning_lgbm.py`` y modifica
únicamente lo necesario para el segundo experimento:

- conserva exactamente el split estratificado 80/10/10;
- mantiene TEST aislado;
- conserva las 15 características y SEXO categórica;
- mantiene Successive Halving, PR-AUC, early stopping, paralelización y artefactos;
- convierte ``scale_pos_weight`` en hiperparámetro categórico;
- corrige la incompatibilidad de ``Axes.boxplot(labels=...)`` con Matplotlib reciente.

Por defecto se evalúan los pesos 1, 2, 4 y ``auto``. ``auto`` corresponde a la
relación negativos/positivos calculada exclusivamente en TRAIN.

Ejemplo:
    python app/churn/07_hptuning_lgbm_pesos.py \
        --workers 16 \
        --threads-per-worker 16

Opcional:
    python app/churn/07_hptuning_lgbm_pesos.py \
        --scale-pos-weights 1,2,4,auto \
        --experiment-name lgbm_churn_weight_search
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
BASE_SCRIPT = HERE / "07_hptuning_lgbm.py"
DEFAULT_EXPERIMENT = "lgbm_churn_weight_search"
DEFAULT_WEIGHT_SPEC = "1,2,4,auto"


def _load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(f"No se encontró el script base: {BASE_SCRIPT}")

    spec = importlib.util.spec_from_file_location("churn_hptuning_lgbm_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No fue posible cargar el script base: {BASE_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pop_cli_option(name: str, default: str) -> str:
    """Extrae una opción propia antes de delegar el resto al argparse del script base."""
    prefix = f"{name}="
    for index, value in enumerate(list(sys.argv[1:]), start=1):
        if value.startswith(prefix):
            result = value[len(prefix):]
            del sys.argv[index]
            return result
        if value == name:
            if index + 1 >= len(sys.argv):
                raise ValueError(f"{name} requiere un valor")
            result = sys.argv[index + 1]
            del sys.argv[index:index + 2]
            return result
    return default


def _has_cli_option(name: str) -> bool:
    return any(value == name or value.startswith(f"{name}=") for value in sys.argv[1:])


def _get_cli_option(name: str, default: str | None = None) -> str | None:
    prefix = f"{name}="
    for index, value in enumerate(sys.argv[1:], start=1):
        if value.startswith(prefix):
            return value[len(prefix):]
        if value == name and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
    return default


def _parse_weight_tokens(specification: str) -> list[str]:
    tokens = [token.strip().lower() for token in specification.split(",") if token.strip()]
    if not tokens:
        raise ValueError("--scale-pos-weights no puede estar vacío")

    validated: list[str] = []
    for token in tokens:
        if token == "auto":
            validated.append(token)
            continue
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(
                "--scale-pos-weights acepta números positivos o 'auto', separados por comas"
            ) from exc
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Todos los valores de --scale-pos-weights deben ser positivos")
        validated.append(str(value))

    return validated


def _resolve_weights(tokens: list[str], train_ratio: float) -> list[float]:
    resolved = [train_ratio if token == "auto" else float(token) for token in tokens]

    # Elimina duplicados conservando el orden y una tolerancia numérica razonable.
    unique: list[float] = []
    for value in resolved:
        if not any(np.isclose(value, existing, rtol=1e-10, atol=1e-12) for existing in unique):
            unique.append(float(value))
    if not unique:
        raise ValueError("No quedaron pesos válidos para evaluar")
    return unique


def _balanced_assignments(values: list[float], count: int, seed: int) -> list[float]:
    """Garantiza cobertura equilibrada de cada peso entre los candidatos."""
    assignments = [values[index % len(values)] for index in range(count)]
    rng = np.random.default_rng(seed + 7919)
    rng.shuffle(assignments)
    return assignments


def _patch_module(base, weight_tokens: list[str]) -> None:
    original_make_candidates = base.make_candidates
    original_save_json = base.save_json
    original_crear_logger = base.crear_logger

    state: dict[str, Any] = {
        "tokens": weight_tokens,
        "resolved": None,
        "train_ratio": None,
    }

    def make_candidates_with_weight_search(
        count: int,
        seed: int,
        train_ratio: float,
    ) -> list[dict[str, Any]]:
        candidates = original_make_candidates(count, seed, train_ratio)
        resolved = _resolve_weights(weight_tokens, float(train_ratio))
        assignments = _balanced_assignments(resolved, count, seed)

        for candidate, weight in zip(candidates, assignments):
            candidate["params"]["scale_pos_weight"] = float(weight)

        state["resolved"] = resolved
        state["train_ratio"] = float(train_ratio)
        return candidates

    def save_json_with_weight_metadata(path: Path, value: Any) -> None:
        payload = copy.deepcopy(value)

        if isinstance(payload, dict) and path.name == "run_definition.json":
            natural_ratio = payload.pop("scale_pos_weight", state.get("train_ratio"))
            payload["schema_version"] = max(int(payload.get("schema_version", 1)), 2)
            payload["scale_pos_weight_train_ratio"] = natural_ratio
            payload["scale_pos_weight_search"] = {
                "mode": "categorical_hyperparameter",
                "requested": state["tokens"],
                "resolved": state["resolved"],
                "assignment": "balanced_and_shuffled_across_candidates",
                "selection_metric": "PR-AUC validation",
            }

        if isinstance(payload, dict) and path.name in {"best_metrics.json", "run_summary.json"}:
            natural_ratio = payload.pop("scale_pos_weight_train", state.get("train_ratio"))
            best_params = payload.get("best_params", {})
            payload["scale_pos_weight_train_ratio"] = natural_ratio
            payload["scale_pos_weight_search"] = {
                "mode": "categorical_hyperparameter",
                "requested": state["tokens"],
                "resolved": state["resolved"],
                "selected": best_params.get("scale_pos_weight"),
                "selection_metric": "PR-AUC validation",
            }

        original_save_json(path, payload)

    def crear_logger_corrected(path: Path):
        raw_log = original_crear_logger(path)

        def log(message: str = "") -> None:
            corrected = message
            if corrected.startswith("   scale_pos_weight ="):
                corrected = corrected.replace(
                    "   scale_pos_weight =",
                    "   ratio natural TRAIN (negativos/positivos) =",
                    1,
                )
            corrected = corrected.replace(
                "   scale_pos_weight permanece fijo durante todos los candidatos",
                "   scale_pos_weight se optimiza como hiperparámetro categórico",
            )
            if corrected.startswith("[LGBM-HPT] Iteraciones finales="):
                corrected = corrected.replace("scale_pos_weight=", "ratio_train=", 1)
            raw_log(corrected)

        return log

    def plot_halving_compatible(all_results: pd.DataFrame, output: Path) -> None:
        if all_results.empty:
            return

        rounds = sorted(all_results["round"].astype(int).unique())
        values = [
            all_results.loc[
                all_results["round"].astype(int) == round_id,
                "pr_auc",
            ].to_numpy()
            for round_id in rounds
        ]
        best = [float(np.max(value)) for value in values]

        fig, ax = plt.subplots(figsize=(9, 5.5))
        labels = [f"R{round_id}" for round_id in rounds]
        try:
            ax.boxplot(values, tick_labels=labels, showfliers=False)
        except TypeError:
            # Compatibilidad con versiones antiguas de Matplotlib.
            ax.boxplot(values, labels=labels, showfliers=False)
        ax.plot(range(1, len(rounds) + 1), best, marker="o", label="Mejor PR-AUC")
        ax.set_xlabel("Ronda de Successive Halving")
        ax.set_ylabel("PR-AUC de validación")
        ax.set_title("LightGBM churn — evolución del Successive Halving")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=160, bbox_inches="tight")
        plt.close(fig)

    base.make_candidates = make_candidates_with_weight_search
    base.save_json = save_json_with_weight_metadata
    base.crear_logger = crear_logger_corrected
    base.plot_halving = plot_halving_compatible
    base._weight_search_state = state


def main() -> None:
    weight_specification = _pop_cli_option("--scale-pos-weights", DEFAULT_WEIGHT_SPEC)
    weight_tokens = _parse_weight_tokens(weight_specification)

    if not _has_cli_option("--experiment-name"):
        sys.argv.extend(["--experiment-name", DEFAULT_EXPERIMENT])
    if not _has_cli_option("--log"):
        sys.argv.extend(
            [
                "--log",
                str(HERE.parents[1] / "data" / "logs" / "07_hptuning_lgbm_churn_pesos.log"),
            ]
        )

    base = _load_base_module()
    _patch_module(base, weight_tokens)

    print(
        "[CHURN-WEIGHTS] scale_pos_weight será optimizado entre: "
        + ", ".join(weight_tokens),
        flush=True,
    )
    print(
        "[CHURN-WEIGHTS] La opción 'auto' se resolverá con negativos/positivos de TRAIN.",
        flush=True,
    )

    base.main()

    experiment_name = _get_cli_option("--experiment-name", DEFAULT_EXPERIMENT)
    output_dir_arg = _get_cli_option("--output-dir")
    if output_dir_arg:
        output_dir = Path(output_dir_arg).expanduser().resolve()
    else:
        output_dir = (
            base.REPO_ROOT
            / "data"
            / "models"
            / "churn"
            / "hpt"
            / str(experiment_name)
        )

    best_hparams_path = output_dir / "best_hparams.json"
    if best_hparams_path.exists():
        with best_hparams_path.open(encoding="utf-8") as handle:
            best_hparams = json.load(handle)
        print(
            "[CHURN-WEIGHTS] Peso seleccionado por PR-AUC de validación: "
            f"scale_pos_weight={best_hparams.get('scale_pos_weight')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
