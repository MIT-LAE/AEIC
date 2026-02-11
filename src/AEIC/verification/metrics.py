# TODO: Remove this when we migrate to Python 3.14+.
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from AEIC.types import Species, SpeciesValues


@dataclass(frozen=True)
class ComparisonMetrics:
    rmse: float = np.nan
    mae: float = np.nan
    mape_pct: float = np.nan
    max_error: float = np.nan
    corr: float = np.nan
    r2: float = np.nan
    n: int = 0

    @classmethod
    def compute(cls, y_true: np.ndarray, y_pred: np.ndarray) -> ComparisonMetrics:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            return ComparisonMetrics()

        y_true_m = y_true[mask]
        y_pred_m = y_pred[mask]
        diff = y_pred_m - y_true_m
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        max_error = float(np.max(np.abs(diff)))

        denom_mask = np.abs(y_true_m) > 1e-12
        if np.any(denom_mask):
            mape_pct = float(
                np.mean(np.abs(diff[denom_mask] / y_true_m[denom_mask])) * 100.0
            )
        else:
            mape_pct = np.nan

        if y_true_m.size < 2 or np.std(y_true_m) == 0 or np.std(y_pred_m) == 0:
            corr = np.nan
        else:
            corr = float(np.corrcoef(y_true_m, y_pred_m)[0, 1])

        ss_res = float(np.sum((y_true_m - y_pred_m) ** 2))
        ss_tot = float(np.sum((y_true_m - np.mean(y_true_m)) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        return ComparisonMetrics(
            rmse=rmse,
            mae=mae,
            mape_pct=mape_pct,
            max_error=max_error,
            corr=corr,
            r2=r2,
            n=int(y_true_m.size),
        )


ComparisonMetricsCollection = dict[
    str, ComparisonMetrics | SpeciesValues[ComparisonMetrics]
]


def out_of_tolerance(
    vals: ComparisonMetricsCollection, rtol: float = 1.0e-5, atol: float = 1.0e-8
) -> list[str | tuple[str, Species]]:
    bad = []
    for k, v in vals.items():
        if isinstance(v, ComparisonMetrics):
            if not np.isclose(v.rmse, 0.0, rtol, atol) or not np.isclose(
                v.mae, 0.0, rtol, atol
            ):
                bad.append(k)
        elif isinstance(v, SpeciesValues):
            for vs, vm in v.items():
                if not np.isclose(vm.rmse, 0.0, rtol, atol) or not np.isclose(
                    vm.mae, 0.0, rtol, atol
                ):
                    bad.append(f'{k} ({vs.name})')
        else:
            raise ValueError(f'Invalid type in ComparisonMetricsCollection: {k}')
    return bad
