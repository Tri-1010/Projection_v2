import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    plt = None
    _HAS_MPL = False
from scipy.optimize import minimize

from src.config import CFG, BUCKETS_CANON

# Notes: inline comments map major blocks to the calibration guidance
# (one-step forecast, WLS k calibration, smoothing, optional alpha).

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except Exception:
    _HAS_CVXPY = False


def validate_transition_matrices(P_by_mob, tol=1e-6):
    """Validate each P_m is square and row-stochastic (rows sum to 1)."""
    for mob, mat in P_by_mob.items():
        mat = np.asarray(mat)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(f"P[{mob}] must be square.")
        rs = mat.sum(axis=1)
        if not np.allclose(rs, 1.0, atol=tol):
            raise ValueError(f"Row sums of P[{mob}] not ~1.")


def _assert_prob_vector(v, tol=1e-6, name="v"):
    """Guard: non-negative vector that sums to ~1 (probability simplex)."""
    v = np.asarray(v, dtype=float)
    if np.any(v < -tol):
        raise AssertionError(f"{name} has negative values (min={v.min():.6g}).")
    s = float(v.sum())
    if not np.isclose(s, 1.0, atol=tol):
        raise AssertionError(f"{name} sum={s:.6g} not ~1 (tol={tol}).")


def _assert_row_stochastic(P, tol=1e-6, name="P"):
    """Guard: each row of P sums to ~1 (valid transition matrix)."""
    P = np.asarray(P, dtype=float)
    rs = P.sum(axis=1)
    if not np.allclose(rs, 1.0, atol=tol):
        raise AssertionError(f"{name} row sums not ~1 (min={rs.min():.6g}, max={rs.max():.6g}).")


def weighted_median(values, weights):
    """Weighted median used to aggregate per-vintage k values (robust to outliers)."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0:
        return np.nan
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w)
    cutoff = 0.5 * np.sum(w)
    return float(v[np.searchsorted(cum, cutoff)])


def del30_from_v(v, s30_idx):
    """Compute DEL30 as sum of the 30+ state probabilities (or shares)."""
    return float(np.sum(v[s30_idx]))


def build_state_vector(
    df: pd.DataFrame,
    vintage_id,
    mob: int,
    states=None,
    vintage_cols=None,
    state_col=None,
    ead_col=None,
    mob_col=None,
    return_total: bool = False,
    tol: float = 1e-6,
):
    """
    Build a normalized state vector v_obs for a given vintage and MOB.
    Returns (v, total) if return_total=True, else v only.
    """
    if states is None:
        states = list(BUCKETS_CANON)
    if vintage_cols is None:
        vintage_cols = ["PRODUCT_TYPE", "RISK_SCORE", "VINTAGE_DATE"]
    if state_col is None:
        state_col = CFG["state"]
    if ead_col is None:
        ead_col = CFG["ead"]
    if mob_col is None:
        mob_col = CFG["mob"]

    if not isinstance(vintage_id, tuple):
        vintage_id = (vintage_id,)
    if len(vintage_id) != len(vintage_cols):
        raise ValueError("vintage_id length must match vintage_cols.")

    mask = pd.Series(True, index=df.index)
    for col, val in zip(vintage_cols, vintage_id):
        mask &= (df[col] == val)
    mask &= (df[mob_col] == mob)

    df_seg = df.loc[mask]
    if df_seg.empty:
        return (None, 0.0) if return_total else None

    ead_vec = (
        df_seg.groupby(state_col)[ead_col].sum()
        .reindex(states, fill_value=0.0)
    )
    total = float(ead_vec.sum())
    if total <= 0:
        return (None, 0.0) if return_total else None

    v = (ead_vec / total).values
    _assert_prob_vector(v, tol=tol, name="v")
    return (v, total) if return_total else v


def _get_P_for_segment(matrices_by_mob, parent_fallback, product, score, mob, states):
    """
    Select P_m for (product, score, mob) with fallbacks:
    exact mob -> last available mob -> parent fallback -> identity.
    """
    prod_str = str(product)
    score_str = str(score)

    mob_dict = matrices_by_mob.get(prod_str, {})
    P_df = None

    if mob in mob_dict and score_str in mob_dict[mob]:
        P_df = mob_dict[mob][score_str]["P"]
    else:
        if mob_dict:
            last_mob = max(mob_dict.keys())
            if score_str in mob_dict[last_mob]:
                P_df = mob_dict[last_mob][score_str]["P"]

    if P_df is None and parent_fallback is not None:
        key_exact = (prod_str, score_str)
        if key_exact in parent_fallback:
            P_df = parent_fallback[key_exact]
        else:
            candidate = [k for k in parent_fallback.keys() if k[0] == prod_str]
            if candidate:
                P_df = parent_fallback[candidate[0]]

    if P_df is None:
        eye = np.eye(len(states))
        P_df = pd.DataFrame(eye, index=states, columns=states)

    return P_df.reindex(index=states, columns=states, fill_value=0.0)


def _build_state_vector_from_series(ead_series, states):
    """Normalize an EAD series into a state vector and return total exposure."""
    v = ead_series.reindex(states, fill_value=0.0).astype(float)
    total = float(v.sum())
    if total <= 0:
        return None, 0.0
    return v, total


def fit_k_raw_segmented(
    actual_results,
    matrices_by_mob,
    parent_fallback,
    states,
    s30_states,
    include_co=True,
    eps=1e-8,
    min_obs=5,
    method="ratio",
    agg="weighted_median",
    min_denom=1e-10,
    fallback_k=1.0,
    fallback_weight=0.0,
    lambda_k=0.0,
    k_prior=0.0,
    denom_mode="ead",
    disb_total_by_vintage=None,
    min_disb=1e-10,
    weight_mode="ead",
    tol=1e-6,
    check_P=True,
    return_detail=False,
):
    """
    Fit k_m per MOB using observed v_m and one-step Markov forecast v_hat.
    method="wls": k_m = sum(w*a*d) / sum(w*a^2), with guards and clipping.
    method="wls_reg": penalized WLS toward k_prior:
        k_m = (sum(w*a*d) + lambda_k * k_prior) / (sum(w*a^2) + lambda_k)
    method="ratio": per-vintage k = d/a, then aggregate (legacy behavior).
    """
    method = str(method).lower()
    if method not in ("ratio", "wls", "wls_reg"):
        raise ValueError(f"Unknown method: {method}")
    if lambda_k < 0:
        raise ValueError("lambda_k must be >= 0.")
    state_set = set(states)
    s30_states = [s for s in s30_states if s in state_set]
    if (not include_co) and ("CO" in s30_states):
        s30_states = [s for s in s30_states if s != "CO"]
    s30_idx = np.array([states.index(s) for s in s30_states], dtype=int)

    denom_mode = str(denom_mode).lower()
    if denom_mode not in ("ead", "disb"):
        raise ValueError(f"Unknown denom_mode: {denom_mode}")
    weight_mode = str(weight_mode).lower()
    if weight_mode not in ("ead", "equal"):
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    rows = []
    for (prod, score, _vintage), mob_dict in actual_results.items():
        mobs = sorted(mob_dict.keys())
        mob_set = set(mobs)
        for m in mobs:
            if (m + 1) not in mob_set:
                continue

            # Step 2/3: use observed v_m to forecast one-step v_hat with P_m.
            v_m, w_m = _build_state_vector_from_series(mob_dict[m], states)
            v_m1, _w_m1 = _build_state_vector_from_series(mob_dict[m + 1], states)
            if v_m is None or v_m1 is None:
                continue

            P_df = _get_P_for_segment(matrices_by_mob, parent_fallback, prod, score, m, states)
            if check_P:
                _assert_row_stochastic(P_df.values, tol=tol, name=f"P[{prod},{score},mob={m}]")

            if denom_mode == "ead":
                v_m_norm = v_m / w_m
                v_m1_norm = v_m1 / v_m1.sum()
                v_hat = v_m_norm.values @ P_df.values
                _assert_prob_vector(v_m_norm.values, tol=tol, name="v_m_norm")
                _assert_prob_vector(v_m1_norm.values, tol=tol, name="v_m1_norm")
                _assert_prob_vector(v_hat, tol=tol, name="v_hat")

                # DEL30 shares on EAD (share on total EAD at each MOB).
                y_vm = del30_from_v(v_m_norm.values, s30_idx)
                y_tar = del30_from_v(v_m1_norm.values, s30_idx)
                y_hat = del30_from_v(v_hat, s30_idx)
                y_vm_amt = y_tar_amt = y_hat_amt = disb_total = None
            else:
                if disb_total_by_vintage is None:
                    raise ValueError("denom_mode='disb' requires disb_total_by_vintage.")
                key = (prod, score, _vintage)
                disb_total = float(disb_total_by_vintage.get(key, 0.0))
                if disb_total <= min_disb:
                    continue

                v_hat_amt = v_m.values @ P_df.values
                y_vm_amt = float(np.sum(v_m.values[s30_idx]))
                y_tar_amt = float(np.sum(v_m1.values[s30_idx]))
                y_hat_amt = float(np.sum(v_hat_amt[s30_idx]))

                # DEL30 scaled by DISB_TOTAL (cohort-based).
                y_vm = y_vm_amt / disb_total
                y_tar = y_tar_amt / disb_total
                y_hat = y_hat_amt / disb_total

            # a = Markov increment, d = actual increment (Step 3 in guidance).
            a = float(y_hat - y_vm)
            d = float(y_tar - y_vm)
            is_small_a = abs(a) < eps

            w_row = 1.0 if weight_mode == "equal" else float(w_m)
            row = {"mob": int(m), "weight": w_row}
            if method == "ratio":
                if is_small_a:
                    # Guardrail: skip near-zero denominators for ratio.
                    continue
                # Legacy per-vintage ratio, then aggregate across vintages.
                k_unclipped = d / a
                k_raw = float(np.clip(k_unclipped, 0.0, 1.0))
                if not (0.0 - tol <= k_raw <= 1.0 + tol):
                    raise AssertionError(f"k_raw out of bounds: {k_raw}")
                row.update({
                    "k_raw": k_raw,
                    "denom": float(a),
                })
                if return_detail:
                    row.update({
                        "product": str(prod),
                        "score": str(score),
                        "vintage": _vintage,
                        "y_vm": float(y_vm),
                        "y_tar": float(y_tar),
                        "y_hat": float(y_hat),
                        "k_raw_unclipped": float(k_unclipped),
                    })
                    if denom_mode == "disb":
                        row.update({
                            "y_vm_amt": y_vm_amt,
                            "y_tar_amt": y_tar_amt,
                            "y_hat_amt": y_hat_amt,
                            "disb_total": disb_total,
                        })
                rows.append(row)
            elif method in ("wls", "wls_reg"):
                # WLS collects a and d; closed form is computed per MOB later.
                row.update({
                    "a": float(a),
                    "d": float(d),
                })
                if return_detail:
                    if is_small_a:
                        k_ratio_unclipped = np.nan
                        k_ratio = np.nan
                    else:
                        k_ratio_unclipped = d / a
                        k_ratio = float(np.clip(k_ratio_unclipped, 0.0, 1.0))
                    row.update({
                        "product": str(prod),
                        "score": str(score),
                        "vintage": _vintage,
                        "y_vm": float(y_vm),
                        "y_tar": float(y_tar),
                        "y_hat": float(y_hat),
                        "k_ratio_unclipped": float(k_ratio_unclipped),
                        "k_ratio": k_ratio,
                    })
                    if denom_mode == "disb":
                        row.update({
                            "y_vm_amt": y_vm_amt,
                            "y_tar_amt": y_tar_amt,
                            "y_hat_amt": y_hat_amt,
                            "disb_total": disb_total,
                        })
                rows.append(row)
            else:
                raise ValueError(f"Unknown method: {method}")

    df_k = pd.DataFrame(rows)
    if df_k.empty:
        return {}, {}, df_k

    k_raw_by_mob = {}
    weight_by_mob = {}

    if method == "ratio":
        for m, grp in df_k.groupby("mob"):
            if len(grp) < min_obs:
                continue
            vals = grp["k_raw"].values
            wts = grp["weight"].values
            if agg == "weighted_mean":
                k_m = float(np.average(vals, weights=wts))
            else:
                k_m = weighted_median(vals, wts)
            k_raw_by_mob[int(m)] = float(np.clip(k_m, 0.0, 1.0))
            weight_by_mob[int(m)] = float(np.sum(wts))
    else:
        mob_stats = []
        for m, grp in df_k.groupby("mob"):
            n_obs = int(len(grp))
            wts = grp["weight"].values
            a_vals = grp["a"].values
            d_vals = grp["d"].values
            w_sum = float(np.sum(wts))
            # Step 4.2: WLS numerator/denominator for k_m.
            denom = float(np.sum(wts * (a_vals ** 2)))
            numer = float(np.sum(wts * a_vals * d_vals))

            # Guardrails: require enough obs and stable denom.
            denom_eff = denom + float(lambda_k) if method == "wls_reg" else denom
            use_wls = (n_obs >= min_obs) and (denom_eff >= min_denom)
            if use_wls:
                if method == "wls_reg":
                    k_m = (numer + float(lambda_k) * float(k_prior)) / denom_eff
                else:
                    k_m = numer / denom
                w_used = w_sum
            else:
                # Fallback when denom is too small or obs are insufficient.
                k_m = float(fallback_k)
                w_used = float(fallback_weight)

            # Clip to [0, 1] as required by governance.
            k_m = float(np.clip(k_m, 0.0, 1.0))
            k_raw_by_mob[int(m)] = k_m
            weight_by_mob[int(m)] = float(w_used)

            if return_detail:
                mob_stats.append({
                    "mob": int(m),
                    "n_obs": n_obs,
                    "w_sum": w_sum,
                    "denom_wls": denom,
                    "numer_wls": numer,
                    "denom_eff": denom_eff,
                    "lambda_k": float(lambda_k),
                    "k_prior": float(k_prior),
                    "k_wls": k_m,
                    "w_used": float(w_used),
                    "wls_used": bool(use_wls),
                })

        if return_detail and mob_stats:
            # Attach per-MOB WLS diagnostics to each row for reporting.
            df_k = df_k.merge(pd.DataFrame(mob_stats), on="mob", how="left")

    return k_raw_by_mob, weight_by_mob, df_k


def fit_k_raw(*args, **kwargs):
    """Public wrapper; keeps backward-compatible name."""
    return fit_k_raw_segmented(*args, **kwargs)


def smooth_k(
    k_raw_by_mob,
    weight_by_mob,
    mob_min,
    mob_max,
    gamma=10.0,
    monotone=False,
    use_cvxpy=True,
    default_k=1.0,
    tol=1e-6,
):
    """
    Smooth k across MOB with a second-difference penalty (Step 6).
    Optional monotone constraint; CVXPY if available, else scipy.
    """
    mobs = np.arange(mob_min, mob_max + 1)
    n = len(mobs)
    k_raw = np.array([k_raw_by_mob.get(m, default_k) for m in mobs], dtype=float)
    w = np.array([weight_by_mob.get(m, 0.0) for m in mobs], dtype=float)

    if use_cvxpy and _HAS_CVXPY:
        k = cp.Variable(n)
        # Second-difference penalty encourages smooth curves.
        diff2 = k[2:] - 2 * k[1:-1] + k[:-2]
        obj = cp.sum(cp.multiply(w, cp.square(k - k_raw))) + gamma * cp.sum_squares(diff2)
        cons = [k >= 0, k <= 1]
        if monotone:
            cons.append(k[1:] >= k[:-1])
        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver=cp.OSQP)
        k_val = np.array(k.value).reshape(-1)
    else:
        def obj(x):
            d2 = x[2:] - 2 * x[1:-1] + x[:-2]
            return np.sum(w * (x - k_raw) ** 2) + gamma * np.sum(d2 ** 2)

        bounds = [(0.0, 1.0)] * n
        constraints = []
        if monotone:
            for i in range(1, n):
                constraints.append({"type": "ineq", "fun": lambda x, i=i: x[i] - x[i - 1]})

        method = "SLSQP" if monotone else "L-BFGS-B"
        res = minimize(obj, x0=k_raw, method=method, bounds=bounds, constraints=constraints)
        k_val = res.x

    if np.any(k_val < -tol) or np.any(k_val > 1.0 + tol):
        raise AssertionError("k_smooth out of bounds before clipping.")

    # Clip again to keep k in [0, 1] after optimization.
    k_smooth_by_mob = {int(mobs[i]): float(np.clip(k_val[i], 0.0, 1.0)) for i in range(n)}
    return k_smooth_by_mob, mobs, k_val


def forecast_segment_partial_step(
    matrices_by_mob,
    parent_fallback,
    product,
    score,
    start_mob,
    initial_ead,
    max_mob,
    k_by_mob,
    states,
    check_P=False,
    tol=1e-6,
):
    """
    Forecast with partial-step k: v_{m+1} = v_m + k_m * (v_hat - v_m).
    """
    results = {}
    cur = initial_ead.reindex(states, fill_value=0.0).astype(float)
    results[start_mob] = cur.copy()

    for mob in range(start_mob, max_mob):
        P_df = _get_P_for_segment(matrices_by_mob, parent_fallback, product, score, mob, states)
        if check_P:
            _assert_row_stochastic(P_df.values, tol=tol, name=f"P[{product},{score},mob={mob}]")
        v_hat = cur.values @ P_df.values
        k_m = float(np.clip(k_by_mob.get(mob, 1.0), 0.0, 1.0))
        # Apply k on the state vector (Step 8.2).
        cur = cur + k_m * (pd.Series(v_hat, index=states) - cur)
        results[mob + 1] = cur.copy()

    return results


def forecast_all_vintages_partial_step(
    actual_results,
    matrices_by_mob,
    parent_fallback,
    max_mob,
    k_by_mob,
    states,
):
    """Apply partial-step forecast for every (product, score, vintage)."""
    results = {}
    for (prod, score, vintage), mob_dict in actual_results.items():
        if not mob_dict:
            continue
        start_mob = int(max(mob_dict.keys()))
        init_ead = mob_dict[start_mob].reindex(states, fill_value=0.0)
        fc = forecast_segment_partial_step(
            matrices_by_mob=matrices_by_mob,
            parent_fallback=parent_fallback,
            product=prod,
            score=score,
            start_mob=start_mob,
            initial_ead=init_ead,
            max_mob=max_mob,
            k_by_mob=k_by_mob,
            states=states,
        )
        results[(prod, score, vintage)] = fc
    return results


def fit_alpha_segmented(
    actual_results,
    matrices_by_mob,
    parent_fallback,
    states,
    s30_states,
    k_smooth_by_mob,
    mob_target,
    include_co=True,
    alpha_grid=None,
    val_frac=0.2,
    denom_mode="ead",
    disb_total_by_vintage=None,
    min_disb=1e-10,
    weight_mode="ead",
    tol=1e-6,
    check_P=True,
):
    """
    Optional Step 7: scale k by alpha to best fit a long-horizon MOB target.
    Alpha is selected by weighted MAE on a validation slice of vintages.
    """
    state_set = set(states)
    s30_states = [s for s in s30_states if s in state_set]
    if (not include_co) and ("CO" in s30_states):
        s30_states = [s for s in s30_states if s != "CO"]
    s30_idx = np.array([states.index(s) for s in s30_states], dtype=int)

    denom_mode = str(denom_mode).lower()
    if denom_mode not in ("ead", "disb"):
        raise ValueError(f"Unknown denom_mode: {denom_mode}")
    weight_mode = str(weight_mode).lower()
    if weight_mode not in ("ead", "equal"):
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    keys = sorted(actual_results.keys(), key=lambda x: x[2])
    if not keys:
        return 1.0, k_smooth_by_mob, pd.DataFrame()

    # Use the most recent vintages as validation by default.
    n_val = max(1, int(len(keys) * val_frac))
    val_keys = keys[-n_val:]

    if alpha_grid is None:
        alpha_grid = np.arange(0.5, 1.5 + 0.01, 0.01)

    actual_cache = {}
    weight_cache = {}
    start_cache = {}

    for key in val_keys:
        mob_dict = actual_results[key]
        if mob_target not in mob_dict:
            continue
        mob_start = int(min(mob_dict.keys()))
        v0, _w0 = _build_state_vector_from_series(mob_dict[mob_start], states)
        vT, wT = _build_state_vector_from_series(mob_dict[mob_target], states)
        if v0 is None or vT is None:
            continue
        if denom_mode == "ead":
            v0_norm = (v0 / v0.sum()).values
            vT_norm = (vT / vT.sum()).values
            _assert_prob_vector(v0_norm, tol=tol, name="v0_norm")
            _assert_prob_vector(vT_norm, tol=tol, name="vT_norm")
            actual_cache[key] = del30_from_v((vT / vT.sum()).values, s30_idx)
            weight_cache[key] = 1.0 if weight_mode == "equal" else wT
            start_cache[key] = (v0, mob_start, None)
        else:
            if disb_total_by_vintage is None:
                raise ValueError("denom_mode='disb' requires disb_total_by_vintage.")
            disb_total = float(disb_total_by_vintage.get(key, 0.0))
            if disb_total <= min_disb:
                continue
            actual_cache[key] = float(np.sum(vT.values[s30_idx])) / disb_total
            weight_cache[key] = 1.0 if weight_mode == "equal" else wT
            start_cache[key] = (v0, mob_start, disb_total)

    best_alpha = None
    best_score = np.inf
    score_rows = []

    for alpha in alpha_grid:
        # Scale k curve; clip to [0, 1].
        k_final = {m: float(np.clip(alpha * k, 0.0, 1.0)) for m, k in k_smooth_by_mob.items()}
        errs = []
        wts = []
        for key in actual_cache.keys():
            v0, mob_start, disb_total = start_cache[key]
            prod, score, _vintage = key
            fc = forecast_segment_partial_step(
                matrices_by_mob=matrices_by_mob,
                parent_fallback=parent_fallback,
                product=prod,
                score=score,
                start_mob=mob_start,
                initial_ead=v0,
                max_mob=mob_target,
                k_by_mob=k_final,
                states=states,
                check_P=check_P,
                tol=tol,
            )
            vT_hat = fc[mob_target]
            if denom_mode == "ead":
                vT_hat_norm = (vT_hat / vT_hat.sum()).values
                _assert_prob_vector(vT_hat_norm, tol=tol, name="vT_hat_norm")
                pred = del30_from_v(vT_hat_norm, s30_idx)
            else:
                pred = float(np.sum(vT_hat.values[s30_idx])) / disb_total
            actual = actual_cache[key]
            w = weight_cache[key]
            errs.append(abs(pred - actual))
            wts.append(w)

        if not errs:
            continue
        # Weighted MAE on DEL30 at mob_target.
        score = float(np.average(errs, weights=wts))
        score_rows.append({"alpha": float(alpha), "score": score})

        if score < best_score:
            best_score = score
            best_alpha = float(alpha)

    score_df = pd.DataFrame(score_rows)
    if not score_df.empty and "score" in score_df.columns:
        score_df = score_df.sort_values("score")
    else:
        score_df = pd.DataFrame(columns=["alpha", "score"])
    if best_alpha is None:
        best_alpha = 1.0
    k_final = {m: float(np.clip(best_alpha * k, 0.0, 1.0)) for m, k in k_smooth_by_mob.items()}
    return best_alpha, k_final, score_df


def fit_alpha(*args, **kwargs):
    """Public wrapper; keeps backward-compatible name."""
    return fit_alpha_segmented(*args, **kwargs)


def forecast_vintage(
    actual_results,
    matrices_by_mob,
    parent_fallback,
    vintage_key,
    states,
    s30_states,
    k_by_mob,
    mob_target,
    include_co=True,
    start_mob=None,
    tol=1e-6,
    check_P=True,
):
    """Forecast a single vintage and compare adjusted vs Markov DEL30."""
    state_set = set(states)
    s30_states = [s for s in s30_states if s in state_set]
    if (not include_co) and ("CO" in s30_states):
        s30_states = [s for s in s30_states if s != "CO"]
    s30_idx = np.array([states.index(s) for s in s30_states], dtype=int)

    if not isinstance(vintage_key, tuple) or len(vintage_key) < 2:
        raise ValueError("vintage_key must be a tuple like (product, score, vintage).")

    mob_dict = actual_results.get(vintage_key)
    if not mob_dict:
        raise KeyError(f"No actual_results for vintage_key={vintage_key}")

    if start_mob is None:
        start_mob = int(max(mob_dict.keys()))

    v0_series = mob_dict[start_mob].reindex(states, fill_value=0.0).astype(float)
    total = float(v0_series.sum())
    if total <= 0:
        raise ValueError("Initial exposure sum <= 0.")

    v_adj = (v0_series / total).values
    v_mkv = v_adj.copy()
    _assert_prob_vector(v_adj, tol=tol, name="v_adj")

    rows = []
    for mob in range(start_mob, mob_target + 1):
        # Track DEL30 path for diagnostics.
        y_adj = del30_from_v(v_adj, s30_idx)
        y_mkv = del30_from_v(v_mkv, s30_idx)
        rows.append({
            "vintage_id": vintage_key,
            "mob": int(mob),
            "del30_hat": float(y_adj),
            "del30_markov": float(y_mkv),
            "diff": float(y_adj - y_mkv),
        })

        if mob == mob_target:
            break

        P_df = _get_P_for_segment(matrices_by_mob, parent_fallback, vintage_key[0], vintage_key[1], mob, states)
        if check_P:
            _assert_row_stochastic(P_df.values, tol=tol, name=f"P[{vintage_key[0]},{vintage_key[1]},mob={mob}]")

        v_hat = v_adj @ P_df.values
        v_mkv = v_mkv @ P_df.values

        k_m = float(np.clip(k_by_mob.get(mob, 1.0), 0.0, 1.0))
        if not (0.0 - tol <= k_m <= 1.0 + tol):
            raise AssertionError(f"k_m out of bounds: {k_m}")

        # Apply partial step on probability vectors.
        v_adj = v_adj + k_m * (v_hat - v_adj)

        v_adj = np.clip(v_adj, 0.0, None)
        v_mkv = np.clip(v_mkv, 0.0, None)
        v_adj = v_adj / v_adj.sum()
        v_mkv = v_mkv / v_mkv.sum()
        _assert_prob_vector(v_adj, tol=tol, name="v_adj")
        _assert_prob_vector(v_mkv, tol=tol, name="v_mkv")

    return pd.DataFrame(rows)


def plot_k_curves(k_raw_by_mob, k_smooth_by_mob, k_final_by_mob):
    """Quick visual check: raw vs smooth vs final k across MOB."""
    if not _HAS_MPL:
        print("matplotlib is not available; skipping plot.")
        return
    mobs = sorted(set(k_raw_by_mob) | set(k_smooth_by_mob) | set(k_final_by_mob))
    kr = [k_raw_by_mob.get(m, np.nan) for m in mobs]
    ks = [k_smooth_by_mob.get(m, np.nan) for m in mobs]
    kf = [k_final_by_mob.get(m, np.nan) for m in mobs]

    plt.figure(figsize=(8, 4))
    plt.plot(mobs, kr, marker="o", label="k_raw")
    plt.plot(mobs, ks, marker="o", label="k_smooth")
    plt.plot(mobs, kf, marker="o", label="k_final")
    plt.xlabel("MOB")
    plt.ylabel("k")
    plt.legend()
    plt.tight_layout()
    plt.show()


def backtest_error_by_mob(
    actual_results,
    matrices_by_mob,
    parent_fallback,
    states,
    s30_states,
    k_by_mob,
    include_co=True,
    denom_mode="ead",
    disb_total_by_vintage=None,
    min_disb=1e-10,
    weight_mode="ead",
):
    """One-step backtest MAE by MOB: adjusted vs pure Markov."""
    denom_mode = str(denom_mode).lower()
    if denom_mode not in ("ead", "disb"):
        raise ValueError(f"Unknown denom_mode: {denom_mode}")
    weight_mode = str(weight_mode).lower()
    if weight_mode not in ("ead", "equal"):
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    state_set = set(states)
    s30_states = [s for s in s30_states if s in state_set]
    if (not include_co) and ("CO" in s30_states):
        s30_states = [s for s in s30_states if s != "CO"]
    s30_idx = np.array([states.index(s) for s in s30_states], dtype=int)

    rows = []
    for (prod, score, _vintage), mob_dict in actual_results.items():
        mobs = sorted(mob_dict.keys())
        mob_set = set(mobs)
        for m in mobs:
            if (m + 1) not in mob_set:
                continue
            v_m, _w_m = _build_state_vector_from_series(mob_dict[m], states)
            v_m1, w_m1 = _build_state_vector_from_series(mob_dict[m + 1], states)
            if v_m is None or v_m1 is None:
                continue

            P_df = _get_P_for_segment(matrices_by_mob, parent_fallback, prod, score, m, states)
            v_hat = v_m.values @ P_df.values
            k_m = float(np.clip(k_by_mob.get(m, 1.0), 0.0, 1.0))
            # Apply k on the state vector, then compare DEL30 errors.
            v_adj = v_m + k_m * (pd.Series(v_hat, index=states) - v_m)

            if denom_mode == "ead":
                y_tar = del30_from_v((v_m1 / v_m1.sum()).values, s30_idx)
                y_adj = del30_from_v((v_adj / v_adj.sum()).values, s30_idx)
                y_mkv = del30_from_v((v_hat / v_hat.sum()), s30_idx)
            else:
                if disb_total_by_vintage is None:
                    raise ValueError("denom_mode='disb' requires disb_total_by_vintage.")
                key = (prod, score, _vintage)
                disb_total = float(disb_total_by_vintage.get(key, 0.0))
                if disb_total <= min_disb:
                    continue
                y_tar = float(np.sum(v_m1.values[s30_idx])) / disb_total
                y_adj = float(np.sum(v_adj.values[s30_idx])) / disb_total
                y_mkv = float(np.sum(v_hat[s30_idx])) / disb_total

            rows.append({
                "mob": int(m + 1),
                "err_adj": abs(y_adj - y_tar),
                "err_markov": abs(y_mkv - y_tar),
                "weight": 1.0 if weight_mode == "equal" else float(w_m1),
            })

    df_err = pd.DataFrame(rows)
    if df_err.empty:
        return df_err

    out = (
        df_err.groupby("mob")
        .apply(lambda g: pd.Series({
            "mae_adj": np.average(g["err_adj"], weights=g["weight"]),
            "mae_markov": np.average(g["err_markov"], weights=g["weight"]),
        }))
        .reset_index()
    )
    return out
