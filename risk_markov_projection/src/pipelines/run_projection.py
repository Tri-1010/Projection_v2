from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config
from src.data.data_loader import load_raw_data
from src.data.schema import default_schema
from src.data.validators import validate_input
from src.models.markov.calibration import apply_scalar_to_distributions, fit_scalar_by_mob
from src.models.markov.projector import MarkovProjector
from src.models.markov.transition import TransitionModel
from src.utils.logger import get_logger
from src.utils.metrics import aggregate_indicator_by_mob, delinquency_indicator, mae, wape
from src.utils.cohort_report import export_cohort_del30_report


logger = get_logger(__name__)


def compute_actual_indicator_by_mob(df: pd.DataFrame) -> pd.Series:
    schema = default_schema()
    mob0 = df[df[schema.mob_col] == 0]
    total_ead0 = mob0[schema.ead_col].sum()
    if total_ead0 == 0:
        return pd.Series(dtype=float)
    indicators = {}
    for mob, group in df.groupby(schema.mob_col):
        ead_by_state = group.groupby(schema.state_col)[schema.ead_col].sum()
        dist_vec = []
        for state in config.STATE_ORDER:
            dist_vec.append(ead_by_state.get(state, 0.0) / total_ead0)
        indicators[mob] = delinquency_indicator(dist_vec, config.STATE_ORDER, config.BUCKETS_30P)
    return pd.Series(indicators)


def run(asof_date: str | None = None, target_mob: int | None = None, source: str | None = None, parquet_path: Path | None = None) -> pd.DataFrame:
    schema = default_schema()
    max_mob = target_mob if target_mob is not None else config.MAX_MOB
    logger.info("Loading data...")
    df = load_raw_data(schema=schema, source=source, parquet_path=parquet_path)
    logger.info("Validating input...")
    validate_input(df, schema=schema, max_mob=max_mob)

    logger.info("Building transitions...")
    transition_model = TransitionModel(df, schema=schema, state_order=config.STATE_ORDER, min_obs=config.MIN_OBS, min_ead=config.MIN_EAD)
    projector = MarkovProjector(transition_model, schema=schema, state_order=config.STATE_ORDER, max_mob=max_mob)

    logger.info("Running projection...")
    projection_df = projector.project(df)

    actual_series = compute_actual_indicator_by_mob(df)
    predicted_series = aggregate_indicator_by_mob(
        projection_df, indicator_col="DEL_30P_ON_EAD0", weight_col="EAD0", mob_col=schema.mob_col
    ) if not projection_df.empty else pd.Series(dtype=float)

    if config.CALIBRATION.get("enabled", False) and not projection_df.empty:
        logger.info("Applying calibration (scalar_by_mob)...")
        actual_aligned = (
            actual_series.reindex(predicted_series.index).ffill().bfill()
            if not actual_series.empty
            else pd.Series(index=predicted_series.index, dtype=float)
        )
        factors = fit_scalar_by_mob(
            actual_aligned,
            predicted_series,
            lower=config.CALIBRATION["lower_bound"],
            upper=config.CALIBRATION["upper_bound"],
        )
        factor_dict = {int(idx): float(val) for idx, val in factors.items()}
        projection_df = apply_scalar_to_distributions(
            projection_df,
            factors=factor_dict,
            target_buckets=config.BUCKETS_30P,
            state_order=config.STATE_ORDER,
            mob_col=schema.mob_col,
        )
        predicted_series = aggregate_indicator_by_mob(
            projection_df, indicator_col="DEL_30P_ON_EAD0", weight_col="EAD0", mob_col=schema.mob_col
        )

    if not predicted_series.empty:
        actual_aligned = (
            actual_series.reindex(predicted_series.index).ffill().bfill()
            if not actual_series.empty
            else pd.Series(index=predicted_series.index, dtype=float)
        )
        if not actual_aligned.empty:
            logger.info(
                "MAE/WAPE for DEL_30P_ON_EAD0 by MOB: mae=%.6f wape=%.6f",
                mae(actual_aligned, predicted_series),
                wape(actual_aligned, predicted_series),
            )

    if not projection_df.empty:
        fallback_rate = (projection_df["matrix_source"] != "segment_mob").mean()
        logger.info("Fallback usage (non-segment_mob matrices): %.2f%%", 100 * fallback_rate)

    output_dir = config.OUTPUT["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / config.OUTPUT["csv_name"]
    parquet_output = output_dir / config.OUTPUT["parquet_name"]
    projection_df.to_csv(csv_path, index=False)
    projection_df.to_parquet(parquet_output, index=False)

    # Report actual vs predicted DEL_30P_ON_EAD0 by MOB
    if not predicted_series.empty:
        actual_aligned = (
            actual_series.reindex(predicted_series.index).ffill().bfill()
            if not actual_series.empty
            else pd.Series(index=predicted_series.index, dtype=float)
        )
        report_df = pd.DataFrame(
            {
                schema.mob_col: predicted_series.index,
                "ACTUAL_DEL30P_ON_EAD0": actual_aligned.values,
                "PRED_DEL30P_ON_EAD0": predicted_series.values,
            }
        )
        report_df["ABS_ERR"] = (report_df["ACTUAL_DEL30P_ON_EAD0"] - report_df["PRED_DEL30P_ON_EAD0"]).abs()
        report_df["REL_ERR"] = report_df["ABS_ERR"] / report_df["ACTUAL_DEL30P_ON_EAD0"].abs()
        report_path = output_dir / config.OUTPUT.get("report_name", "indicator_report.csv")
        report_df.to_csv(report_path, index=False)
        logger.info("Saved indicator report to %s", report_path)

    # Cohort report (requires cohort_col in schema and data)
    cohort_col = getattr(schema, "cohort_col", None)
    if cohort_col and cohort_col in df.columns:
        cohort_report_path = output_dir / config.OUTPUT.get("cohort_report_name", "cohort_del30_report.csv")
        export_cohort_del30_report(
            df,
            projection_df,
            output_path=cohort_report_path,
            schema=schema,
            state_order=config.STATE_ORDER,
            buckets_30p=config.BUCKETS_30P,
            max_mob=max_mob,
        )
        logger.info("Saved cohort DEL30 report to %s", cohort_report_path)

    logger.info("Projection saved to %s and %s", csv_path, parquet_output)
    return projection_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run risk Markov projection pipeline.")
    parser.add_argument("--asof-date", required=False, help="As-of date for the projection (metadata only).")
    parser.add_argument("--target-mob", required=False, type=int, help="Projection horizon (default from config).")
    parser.add_argument("--source", required=False, choices=["parquet", "oracle"], help="Override DATA_SOURCE.")
    parser.add_argument("--parquet-path", required=False, type=str, help="Override parquet path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        asof_date=args.asof_date,
        target_mob=args.target_mob,
        source=args.source,
        parquet_path=Path(args.parquet_path) if args.parquet_path else None,
    )
