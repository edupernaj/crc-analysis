"""
Statistical analysis of block-level CPT CRC results.

Experimental design
-------------------
Only the condition:

    CPT mit PB ohne Thermometer + Pumpe

is analysed.

Each participant has four experimental pairs. Each pair contains:

    warm / non-pain -> pain / cold

The statistical variable ``block`` therefore takes values 1-4 and represents
the four paired experimental blocks.

Primary outcome
---------------
``crc_log_median``

This is the natural logarithm of the positive block-level CRC median.

Why log CRC?
------------
CRC is an energy-like positive quantity and can have a strongly skewed
distribution. Modelling the raw CRC can therefore produce very large
standard errors and make the residual distribution poorly behaved.

The log transformation makes multiplicative differences additive:

    log(CRC_pain) - log(CRC_warm)

Each block-specific cold-minus-warm contrast can therefore be exponentiated:

    exp(beta_block)

to obtain the estimated multiplicative CRC ratio.

For example:

    beta = 0.10  -> pain CRC approximately 1.11 times warm CRC
    beta = -0.10 -> pain CRC approximately 0.90 times warm CRC

Primary models
--------------
Random intercept:

    CRC_log ~ pain * C(block) + (1 | participant)

Random slope:

    CRC_log ~ pain * C(block) + (pain | participant)

The block factor is categorical, matching the period treatment in the
original LMM analysis. This allows the warm/cold effect to differ by
experimental block instead of imposing a linear time trend.

The default ``MODEL_CHOICE = "auto"`` compares the random-intercept and
random-slope models using ML, selects the random-slope model only when the
diagnostic LRT is significant and the fit converges, and refits the selected
model with REML. Both models can also be fitted as a sensitivity analysis by
setting:

    MODEL_CHOICE = "both"

Alternatively:

    MODEL_CHOICE = "random_intercept"
    MODEL_CHOICE = "random_slope"

The random-slope model allows the pain effect to vary between participants.

The primary fixed effect is ``pain``.

Interpretation
--------------
pain = 0:
    warm / non-pain

pain = 1:
    cold / pain

With the interaction model, the ``pain`` coefficient is the cold-versus-warm
effect in block 1. The block-specific cold-versus-warm effects are calculated
as the appropriate linear contrasts of ``pain`` and the interaction terms.
Their standard errors and p-values use the full fixed-effect covariance matrix.

The analysis deliberately does not include tau or DI.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import build_design_matrices
from scipy import stats

# ============================================================================
# Constants
# ============================================================================

PRIMARY_OUTCOME = "crc_log_median"
RAW_OUTCOME = "crc_median"

# "auto" selects one primary model after an ML comparison. "both" keeps both
# REML fits as a sensitivity analysis while still identifying the selected
# primary model in the output.
MODEL_CHOICE = "random_intercept"

MODEL_SPECS = {
    "random_intercept": {
        "label": "Random intercept",
        "formula": "crc_log_median ~ pain * C(block)",
        "re_formula": "1",
    },
    "random_slope": {
        "label": "Random intercept + pain slope",
        "formula": "crc_log_median ~ pain * C(block)",
        "re_formula": "1 + pain",
    },
}


OUTPUT_FOLDER = Path("output")
OUTPUT_FOLDER.mkdir(exist_ok=True)
PLOTS_FOLDER = OUTPUT_FOLDER.joinpath("plots")
PLOTS_FOLDER.mkdir(exist_ok=True)
RESULTS_FOLDER = OUTPUT_FOLDER.joinpath("results")
RESULTS_FOLDER.mkdir(exist_ok=True)


# ============================================================================
# Result container
# ============================================================================
@dataclass
class FittedModel:
    """Container for a fitted mixed-effects model and metadata."""

    name: str
    formula: str
    re_formula: str
    result: object
    reml: bool
    fit_warnings: list[str]


# ============================================================================
# Loading and validation
# ============================================================================
def load_results(path: Path) -> pd.DataFrame:
    """
    Load and validate the generated CPT CRC dataset.

    The primary outcome must be ``crc_log_median``. Raw CRC is retained only
    for descriptive/reference purposes.
    """
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    df = pd.read_csv(path)

    required = {
        "participant",
        "block",
        "pain",
        "crc_median",
        "crc_log_median",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()

    df["participant"] = df["participant"].astype(str)
    df["block"] = pd.to_numeric(df["block"], errors="coerce")
    df["pain"] = pd.to_numeric(df["pain"], errors="coerce")
    df["crc_median"] = pd.to_numeric(df["crc_median"], errors="coerce")
    df["crc_log_median"] = pd.to_numeric(df["crc_log_median"], errors="coerce")

    if not df["pain"].dropna().isin([0, 1]).all():
        raise ValueError("pain must contain only 0 (warm/non-pain) and 1 (cold/pain).")
    if not df["block"].dropna().isin([1, 2, 3, 4]).all():
        raise ValueError("block must contain only paired experimental blocks 1-4.")
    if (df["crc_median"].dropna() <= 0).any():
        raise ValueError(
            "crc_median contains non-positive values. "
            "The log-transformed CRC outcome cannot be used safely."
        )

    df = df.dropna(
        subset=["participant", "block", "pain", "crc_median", "crc_log_median"]
    )
    if df.empty:
        raise ValueError("No complete observations remain.")

    df["block"] = df["block"].astype(int)
    df["pain"] = df["pain"].astype(int)

    return df.sort_values(["participant", "block", "pain"]).reset_index(drop=True)


# ============================================================================
# Analysis dataset preparation
# ============================================================================
def prepare_analysis_data(
    df: pd.DataFrame,
    require_both_conditions: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare the dataset for within-participant mixed-effects analysis.

    Participants without both warm and cold observations are excluded by
    default because the primary question concerns the within-participant
    pain effect.
    """
    df = df.copy()
    if not require_both_conditions:
        return df, pd.DataFrame(columns=["participant", "reason"])

    counts = (
        df.groupby("participant")["pain"]
        .agg(n_observations="size", n_conditions="nunique")
        .reset_index()
    )
    excluded = counts[counts["n_conditions"] < 2].copy()
    excluded["reason"] = (
        "Participant does not have both warm/non-pain and cold/pain observations."
    )
    keep = counts.loc[counts["n_conditions"] == 2, "participant"]
    analysis_df = df[df["participant"].isin(keep)].copy()
    return (analysis_df, excluded[["participant", "reason"]])


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate descriptive statistics for the log-transformed CRC outcome.
    """
    summary = (
        df.groupby("pain", observed=True)[PRIMARY_OUTCOME]
        .agg(n="count", mean="mean", median="median", sd="std", min="min", max="max")
        .reset_index()
    )
    summary["condition"] = summary["pain"].map({0: "Warm / non-pain", 1: "Cold / pain"})
    summary["geometric_mean_crc"] = np.exp(summary["mean"])
    return summary[
        [
            "condition",
            "pain",
            "n",
            "mean",
            "median",
            "sd",
            "min",
            "max",
            "geometric_mean_crc",
        ]
    ]


# ============================================================================
# Within-subject effects
# ============================================================================
def within_subject_changes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate participant-level pain-minus-warm log CRC changes.

    Repeated observations are averaged within participant and condition.

    The primary contrast is:

        mean(log CRC pain) - mean(log CRC warm)

    Exponentiating the contrast gives the corresponding CRC ratio.
    """
    wide = (
        df.groupby(["participant", "pain"], observed=True)[PRIMARY_OUTCOME]
        .mean()
        .unstack("pain")
        .rename(columns={0: "warm_log_crc", 1: "pain_log_crc"})
        .reset_index()
    )
    wide["pain_minus_warm_log_crc"] = wide["pain_log_crc"] - wide["warm_log_crc"]
    wide["pain_warm_crc_ratio"] = np.exp(wide["pain_minus_warm_log_crc"])
    wide["percent_change"] = 100.0 * (wide["pain_warm_crc_ratio"] - 1.0)
    return wide


# ============================================================================
# Simple sensitivity test
# ============================================================================
def paired_descriptive_test(df: pd.DataFrame) -> dict:
    """
    Run a paired participant-level t-test on log CRC.

    This is NOT the primary inferential analysis.
    It is included as a simple sensitivity analysis to provide an intuitive
    participant-level comparison alongside the LMM.
    """
    changes = within_subject_changes(df)["pain_minus_warm_log_crc"].dropna()

    if len(changes) < 2:
        return {
            "n_participants": int(len(changes)),
            "mean_log_change": np.nan,
            "geometric_crc_ratio": np.nan,
            "t_statistic": np.nan,
            "p_value": np.nan,
        }
    test = stats.ttest_1samp(changes, popmean=0.0)
    mean_change = float(changes.mean())
    return {
        "n_participants": int(len(changes)),
        "mean_log_change": mean_change,
        "geometric_crc_ratio": float(np.exp(mean_change)),
        "t_statistic": float(test.statistic),
        "p_value": float(test.pvalue),
    }


# ============================================================================
# Mixed-effects model fitting
# ============================================================================
def _fit_mixedlm_once(
    df: pd.DataFrame,
    model_name: str,
    reml: bool,
    method: str,
):
    """Fit one MixedLM with a specified optimizer."""
    specification = MODEL_SPECS[model_name]

    model = smf.mixedlm(
        specification["formula"],
        data=df,
        groups=df["participant"],
        re_formula=specification["re_formula"],
    )
    return model.fit(
        reml=reml,
        method=method,
        maxiter=2000,
        disp=False,
    )


def fit_mixed_model(
    df: pd.DataFrame,
    model_name: str,
    reml: bool = True,
) -> FittedModel:
    """Fit one of the two predefined CRC LMMs.

    Parameters
    ----------
    df
        Analysis dataframe.
    model_name
        ``random_intercept`` or ``random_slope``.
    reml
        Whether to use REML.
        Use ``reml=False`` for comparison of candidate models.
        Use ``reml=True`` for final parameter estimation.

    Returns
    -------
    FittedModel
        Fitted model and metadata.

    Notes
    -----
    The random-intercept model corresponds to:
        CRC ~ pain + block + (1 | participant)
    The random-slope model corresponds to:
        CRC ~ pain + block + (1 + pain | participant)
    ``statsmodels`` implements these random-effects structures through
    ``re_formula``. The default random effect is a group-specific intercept,
    while ``re_formula='1 + pain'`` adds a participant-specific pain slope.
    """
    if model_name not in MODEL_SPECS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from {list(MODEL_SPECS)}."
        )
    errors: list[str] = []
    fit_warnings: list[str] = []

    # Mixed models can be sensitive to the optimizer, especially the
    # random-slope model. Try a small set rather than silently accepting
    # a failed first attempt.
    for method in ("lbfgs", "bfgs", "powell", "cg"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = _fit_mixedlm_once(
                    df=df, model_name=model_name, reml=reml, method=method
                )
            warning_messages = [str(w.message) for w in caught]
            converged = bool(getattr(result, "converged", False))
            if converged:
                fit_warnings.extend(warning_messages)
                return FittedModel(
                    name=model_name,
                    formula=MODEL_SPECS[model_name]["formula"],
                    re_formula=MODEL_SPECS[model_name]["re_formula"],
                    result=result,
                    reml=reml,
                    fit_warnings=fit_warnings,
                )
            errors.append(f"{method}: optimizer did not converge")
        except Exception as exc:
            errors.append(f"{method}: {type(exc).__name__}: {exc}")
    raise RuntimeError(f"Could not fit {model_name}.\n\n" + "\n".join(errors))


# ============================================================================
# Model results
# ============================================================================
def model_parameter_table(fitted: FittedModel) -> pd.DataFrame:
    """Return fixed-effect estimates and confidence intervals."""
    result = fitted.result
    confidence_intervals = result.conf_int()
    rows = []
    for name in result.fe_params.index:
        rows.append(
            {
                "model": fitted.name,
                "term": name,
                "estimate": float(result.fe_params[name]),
                "std_error": float(result.bse[name]),
                "z_or_t": float(result.tvalues[name]),
                "p_value": float(result.pvalues[name]),
                "ci_low": float(confidence_intervals.loc[name, 0]),
                "ci_high": float(confidence_intervals.loc[name, 1]),
            }
        )
    return pd.DataFrame(rows)


def block_effect_table(fitted: FittedModel) -> pd.DataFrame:
    """Return correct cold-minus-warm contrasts for each experimental block.

    For the interaction model, the block 2--4 effects are not the interaction
    coefficients by themselves. Each effect is a contrast between the fitted
    cold and warm means at that block, with uncertainty calculated from the
    complete fixed-effect covariance matrix.
    """
    result = fitted.result
    fixed_names = list(result.fe_params.index)
    design_info = result.model.data.design_info
    rows = []

    for block in [1, 2, 3, 4]:
        new_data = pd.DataFrame({"pain": [0, 1], "block": [block, block]})
        design = build_design_matrices(
            [design_info], new_data, return_type="dataframe"
        )[0]
        design = design.reindex(columns=fixed_names)
        contrast = design.iloc[1].to_numpy() - design.iloc[0].to_numpy()

        covariance = result.cov_params().loc[fixed_names, fixed_names].to_numpy()
        estimate = float(contrast @ result.fe_params.loc[fixed_names].to_numpy())
        variance = float(contrast @ covariance @ contrast)
        standard_error = float(np.sqrt(max(variance, 0.0)))
        z_value = estimate / standard_error if standard_error > 0 else np.nan
        p_value = (
            float(2.0 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
        )
        ci_low = estimate - 1.96 * standard_error
        ci_high = estimate + 1.96 * standard_error

        rows.append(
            {
                "model": fitted.name,
                "block": block,
                "log_crc_cold_minus_warm": estimate,
                "std_error": standard_error,
                "z_or_t": z_value,
                "p_value": p_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "crc_ratio_cold_warm": float(np.exp(estimate)),
                "crc_ratio_ci_low": float(np.exp(ci_low)),
                "crc_ratio_ci_high": float(np.exp(ci_high)),
            }
        )

    return pd.DataFrame(rows)


def df_group_count(result) -> int:
    """Return the number of independent participant groups."""
    return int(len(result.model.group_labels))


def extract_primary_effect(fitted: FittedModel) -> dict:
    """
    Extract and interpret the block-1 cold-minus-warm effect.

    ``pain`` is coded:

        0 = warm/non-pain
        1 = cold/pain

    Therefore beta_pain is the log CRC ratio. ``exp(beta_pain)`` is the
    estimated multiplicative CRC ratio.
    """
    effects = block_effect_table(fitted)
    row = effects.loc[effects["block"] == 1].iloc[0]
    beta = float(row["log_crc_cold_minus_warm"])
    ci_low = float(row["ci_low"])
    ci_high = float(row["ci_high"])
    return {
        "model": fitted.name,
        "reml": fitted.reml,
        "formula": fitted.formula,
        "random_effects": fitted.re_formula,
        "pain_log_estimate": beta,
        "pain_std_error": float(row["std_error"]),
        "pain_log_ci_low": ci_low,
        "pain_log_ci_high": ci_high,
        "pain_crc_ratio": float(np.exp(beta)),
        "pain_crc_ratio_ci_low": float(np.exp(ci_low)),
        "pain_crc_ratio_ci_high": float(np.exp(ci_high)),
        "pain_p_value": float(row["p_value"]),
        "n_observations": int(fitted.result.nobs),
        "n_participants": int(df_group_count(fitted.result)),
        "aic": float(fitted.result.aic),
        "bic": float(fitted.result.bic),
        "log_likelihood": float(fitted.result.llf),
        "scale": float(fitted.result.scale),
        "converged": bool(fitted.result.converged),
    }


# ============================================================================
# Model comparison
# ============================================================================
def compare_models(
    df: pd.DataFrame,
) -> tuple[FittedModel, FittedModel, pd.DataFrame, dict]:
    """Compare random-intercept and random-slope models using maximum likelihood.

    Both models have identical fixed effects. They differ only in their random
    effects.
    The comparison table contains:
    - log likelihood
    - AIC
    - BIC
    - convergence status
    A likelihood-ratio statistic is also calculated.

    IMPORTANT:
    The random-slope variance is tested at a boundary of the parameter space,
    so the ordinary chi-square reference distribution for the LRT is only a
    rough diagnostic. It should not be treated as the sole criterion for
    selecting the model.
    """
    random_intercept = fit_mixed_model(df=df, model_name="random_intercept", reml=False)
    random_slope = fit_mixed_model(df=df, model_name="random_slope", reml=False)
    comparison = pd.DataFrame(
        [
            {
                "model": random_intercept.name,
                "reml": False,
                "log_likelihood": float(random_intercept.result.llf),
                "aic": float(random_intercept.result.aic),
                "bic": float(random_intercept.result.bic),
                "converged": bool(random_intercept.result.converged),
            },
            {
                "model": random_slope.name,
                "reml": False,
                "log_likelihood": float(random_slope.result.llf),
                "aic": float(random_slope.result.aic),
                "bic": float(random_slope.result.bic),
                "converged": bool(random_slope.result.converged),
            },
        ]
    )

    # Likelihood-ratio statistic.
    lr_statistic = 2.0 * (
        float(random_slope.result.llf) - float(random_intercept.result.llf)
    )
    df_difference = int(
        len(random_slope.result.params) - len(random_intercept.result.params)
    )
    if df_difference > 0:
        lr_p_value = float(stats.chi2.sf(max(lr_statistic, 0.0), df_difference))
    else:
        lr_p_value = np.nan

    comparison_test = {
        "likelihood_ratio": lr_statistic,
        "df_difference": df_difference,
        "chi_square_reference_p_value": lr_p_value,
        "warning": (
            "The random-slope variance is a boundary parameter. "
            "The ordinary chi-square LRT p-value is therefore only "
            "a rough diagnostic and should not be used alone for "
            "model selection."
        ),
    }
    return (
        random_intercept,
        random_slope,
        comparison,
        comparison_test,
    )


# ============================================================================
# REML refit
# ============================================================================
def refit_selected_model_with_reml(df: pd.DataFrame, model_name: str) -> FittedModel:
    """Refit a selected model with REML for final parameter estimation."""
    return fit_mixed_model(df=df, model_name=model_name, reml=True)


# ============================================================================
# Exploratory plots
# ============================================================================
def plot_condition_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot log CRC by condition.

    The y-axis explicitly states that the outcome is log CRC. The subtitle-like
    annotation explains the coding.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    groups = [
        df.loc[df["pain"] == 0, PRIMARY_OUTCOME].dropna().to_numpy(),
        df.loc[df["pain"] == 1, PRIMARY_OUTCOME].dropna().to_numpy(),
    ]
    ax.boxplot(groups, label=["Warm / non-pain", "Cold / pain"], showmeans=True)
    ax.set_ylabel("Natural-log CRC median")
    ax.set_xlabel("Experimental condition")
    ax.set_title("CRC by experimental condition")
    ax.text(
        0.5,
        -0.18,
        "Lower values indicate lower CRC. Pain = cold condition.",
        transform=ax.transAxes,
        ha="center",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_participant_trajectories(df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot participant trajectories across the four paired experimental blocks.

    Within each block, warm is plotted at the left and cold at the right.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    for participant, participant_df in df.groupby("participant"):
        participant_df = participant_df.sort_values(["block", "pain"])
        x = participant_df["block"].to_numpy(dtype=float)
        x = x + np.where(participant_df["pain"].to_numpy() == 1, 0.12, -0.12)
        ax.plot(
            x, participant_df[PRIMARY_OUTCOME], marker="o", linewidth=0.8, alpha=0.6
        )
    ax.set_xlabel("Experimental pair/block")
    ax.set_ylabel("Natural-log CRC median")
    ax.set_title("Participant CRC trajectories")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(
        [
            "1\nwarm → cold",
            "2\nwarm → cold",
            "3\nwarm → cold",
            "4\nwarm → cold",
        ]
    )
    ax.text(
        0.5,
        -0.15,
        "Each pair contains warm/non-pain followed by cold/pain.",
        transform=ax.transAxes,
        ha="center",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_within_subject_changes(df: pd.DataFrame, output_path: Path) -> None:
    """
    Plot participant-level pain-minus-warm log CRC changes.

    Values below zero indicate lower CRC during cold/pain.
    """
    changes = (
        within_subject_changes(df)
        .dropna(subset=["pain_minus_warm_log_crc"])
        .sort_values("pain_minus_warm_log_crc")
    )
    fig, ax = plt.subplots(figsize=(8, max(4, 0.22 * len(changes))))
    y = np.arange(len(changes))
    ax.scatter(changes["pain_minus_warm_log_crc"], y)
    ax.axvline(0.0, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(changes["participant"])
    ax.set_xlabel("Log CRC: cold/pain − warm/non-pain")
    ax.set_ylabel("Participant")
    ax.set_title("Within-participant CRC changes")
    ax.text(
        0.5,
        -0.12,
        "Left of zero = lower CRC during cold/pain.",
        transform=ax.transAxes,
        ha="center",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ============================================================================
# Model diagnostics
# ============================================================================
def plot_model_residuals(fitted: FittedModel, output_path: Path) -> None:
    """Save residual-vs-fitted and Q-Q diagnostic plots."""
    result = fitted.result
    fitted_values = np.asarray(result.fittedvalues, dtype=float)
    residuals = np.asarray(result.resid, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Residuals versus fitted values.
    axes[0].scatter(fitted_values, residuals, alpha=0.7)
    axes[0].axhline(0.0, linestyle="--")
    axes[0].set_xlabel("Fitted log CRC")
    axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs fitted")

    # Normal Q-Q plot.
    stats.probplot(residuals, dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q plot")
    fig.suptitle(f"Diagnostics: {fitted.name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ============================================================================
# Complete analysis
# ============================================================================
def write_interpretation(
    primary_df: pd.DataFrame,
    block_effects: pd.DataFrame,
    selected_model: str,
    comparison_test: dict | None,
    output_path: Path,
) -> None:
    """
    Write a human-readable interpretation of the primary LMM results.
    """
    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write("CRC statistical analysis\n")
        handle.write("========================\n\n")

        handle.write(
            "Experimental condition:\n  CPT mit PB ohne Thermometer + Pumpe\n\n"
        )

        handle.write(
            "Experimental coding:\n"
            "  pain=0 -> warm / non-pain\n"
            "  pain=1 -> cold / pain\n"
            "  block=1-4 -> paired experimental blocks\n"
            "  segment 9 -> excluded\n\n"
        )

        handle.write("Primary outcome:\n  natural-log transformed block CRC median\n\n")

        handle.write(
            "Primary fixed effects:\n"
            "  pain * categorical(block)\n"
            "  The pain coefficient is the block-1 cold-versus-warm effect.\n\n"
        )

        handle.write(
            "Interpretation:\n"
            "  Each block effect is a cold-minus-warm contrast on the log scale.\n"
            "  exp(effect) is the estimated cold/warm CRC ratio.\n"
            "  ratio < 1 means lower CRC during cold/pain.\n"
            "  ratio > 1 means higher CRC during cold/pain.\n"
            "  Block-specific p-values use the full contrast covariance.\n\n"
        )

        handle.write(f"Selected primary model: {selected_model}\n\n")
        if comparison_test is not None:
            handle.write(
                "Random-effects comparison (ML; diagnostic only):\n"
                f"  likelihood-ratio statistic = {comparison_test['likelihood_ratio']:.6g}\n"
                f"  df difference = {comparison_test['df_difference']}\n"
                f"  chi-square reference p = "
                f"{comparison_test['chi_square_reference_p_value']:.6g}\n"
                f"  note: {comparison_test['warning']}\n\n"
            )

        handle.write("Primary block-specific effects:\n")
        handle.write(block_effects.to_string(index=False))
        handle.write("\n\nPrimary model summary row:\n")
        handle.write(primary_df.to_string(index=False))
        handle.write("\n\n")

        for _, row in block_effects.iterrows():
            ratio = row["crc_ratio_cold_warm"]

            if ratio < 1:
                direction = "lower"
            elif ratio > 1:
                direction = "higher"
            else:
                direction = "equal"

            handle.write(
                f"Block {int(row['block'])} ({row['model']}):\n"
                f"  Estimated cold/warm CRC ratio: {ratio:.4f}\n"
                f"  This corresponds to {direction} CRC during "
                f"cold/pain relative to warm/non-pain.\n"
                f"  p = {row['p_value']:.6g}\n\n"
            )


def run_analysis(
    input_path: Path,
    output_dir: Path,
    model_choice: str,
) -> dict:
    """
    Run the complete statistical analysis.

    ``model_choice`` may be:

        random_intercept
        random_slope
        both

    When ``both`` is selected, the models are compared using ML and then
    both are refitted using REML for final parameter estimation.
    """
    if model_choice not in {"auto", "random_intercept", "random_slope", "both"}:
        raise ValueError(
            "model_choice must be auto, random_intercept, random_slope, or both."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(input_path)
    analysis_df, exclusions = prepare_analysis_data(df)
    if analysis_df["participant"].nunique() < 2:
        raise ValueError("At least two participants are required for the LMM.")
    descriptive = descriptive_statistics(analysis_df)
    descriptive.to_csv(output_dir / "descriptive_statistics.csv", index=False)
    changes = within_subject_changes(analysis_df)
    changes.to_csv(output_dir / "within_subject_changes.csv", index=False)
    if not exclusions.empty:
        exclusions.to_csv(output_dir / "excluded_participants.csv", index=False)
    paired = paired_descriptive_test(analysis_df)
    pd.DataFrame([paired]).to_csv(
        output_dir / "paired_descriptive_test.csv", index=False
    )
    plot_condition_distribution(
        analysis_df, output_dir / "crc_condition_distribution.png"
    )
    plot_participant_trajectories(
        analysis_df, output_dir / "crc_participant_trajectories.png"
    )
    plot_within_subject_changes(
        analysis_df, output_dir / "crc_within_subject_changes.png"
    )

    # Determine the primary model and the models to refit with REML.
    comparison_test = None
    if model_choice in {"auto", "both"}:
        (
            _ml_intercept,
            ml_slope,
            comparison,
            comparison_test,
        ) = compare_models(analysis_df)
        comparison.to_csv(output_dir / "model_comparison.csv", index=False)
        pd.DataFrame([comparison_test]).to_csv(
            output_dir / "model_comparison_test.csv", index=False
        )
        if (
            ml_slope.result.converged
            and np.isfinite(comparison_test["chi_square_reference_p_value"])
            and comparison_test["chi_square_reference_p_value"] < 0.05
        ):
            selected_model_name = "random_slope"
        else:
            selected_model_name = "random_intercept"

        selected_names = (
            ["random_intercept", "random_slope"]
            if model_choice == "both"
            else [selected_model_name]
        )
    else:
        selected_names = [model_choice]
        selected_model_name = model_choice

    (output_dir / "selected_primary_model.txt").write_text(
        selected_model_name + "\n",
        encoding="utf-8",
    )
    final_models: dict[str, FittedModel] = {}
    primary_results: list[dict] = []
    effect_tables: list[pd.DataFrame] = []

    for model_name in selected_names:
        fitted = refit_selected_model_with_reml(analysis_df, model_name)
        final_models[model_name] = fitted

        parameter_table = model_parameter_table(fitted)
        parameter_table.to_csv(output_dir / f"{model_name}_parameters.csv", index=False)
        effects = block_effect_table(fitted)
        effects.to_csv(output_dir / f"{model_name}_block_effects.csv", index=False)
        effect_tables.append(effects)

        primary_result = extract_primary_effect(fitted)
        primary_result["is_primary"] = model_name == selected_model_name
        primary_results.append(primary_result)

        plot_model_residuals(fitted, output_dir / f"{model_name}_diagnostics.png")

        with open(
            output_dir / f"{model_name}_summary.txt", "w", encoding="utf-8"
        ) as handle:
            handle.write(fitted.result.summary().as_text())
            if fitted.fit_warnings:
                handle.write("\n\nWarnings captured during fitting:\n")
                for message in fitted.fit_warnings:
                    handle.write(f"- {message}\n")

    primary_df = pd.DataFrame(primary_results)
    primary_df.to_csv(output_dir / "primary_lmm_results.csv", index=False)
    block_effects_df = pd.concat(effect_tables, ignore_index=True)
    block_effects_df.to_csv(output_dir / "block_effects.csv", index=False)
    primary_block_effects = block_effects_df.loc[
        block_effects_df["model"] == selected_model_name
    ].reset_index(drop=True)
    write_interpretation(
        primary_df,
        primary_block_effects,
        selected_model_name,
        comparison_test,
        output_dir / "analysis_report.txt",
    )

    return {
        "data": analysis_df,
        "exclusions": exclusions,
        "descriptive": descriptive,
        "within_subject_changes": changes,
        "paired_descriptive_test": paired,
        "models": final_models,
        "primary_results": primary_df,
        "block_effects": block_effects_df,
        "selected_model": selected_model_name,
    }


def plot_cold_effect(results: pd.DataFrame, output_path):
    # Cold effect (Δ log units) with CI per period
    fig = plt.figure(figsize=(16, 8))
    ax = plt.axes()

    block_effects = results["block_effects"].copy()
    blocks = block_effects["block"].to_numpy()
    cold_effects = block_effects["log_crc_cold_minus_warm"].to_numpy()
    ci_low = block_effects["ci_low"].to_numpy()
    ci_high = block_effects["ci_high"].to_numpy()
    bar_colors = [
        "#27ae60" if p < 0.05 else "#e74c3c" for p in block_effects["p_value"]
    ]

    ax.bar(
        blocks,
        cold_effects,
        color=bar_colors,
        alpha=0.75,
        edgecolor="black",
        linewidth=0.8,
        width=0.5,
    )
    ax.errorbar(
        blocks,
        cold_effects,
        yerr=[cold_effects - ci_low, ci_high - cold_effects],
        fmt="none",
        color="black",
        capsize=5,
        linewidth=1.5,
    )
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)

    # Annotate percentage change on each bar using the exact block contrast.
    for i, cold_effect in enumerate(cold_effects):
        y_pos = cold_effect + (0.02 if cold_effect >= 0 else -0.04)
        ax.text(
            blocks[i],
            y_pos,
            f"{float(100.0 * (np.exp(cold_effect) - 1.0)):+.0f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title(
        "Cold water effect on CRC per period\n(green = p < .05, red = p ≥ .05)",
        fontweight="bold",
    )
    ax.set_xlabel("Period")
    ax.set_ylabel("Δ CRC_log (cold − warm)")
    ax.set_xticks(blocks)
    ax.set_xticklabels([f"P{p}" for p in blocks])
    ax.grid(True, alpha=0.25, axis="y")

    plt.tight_layout()
    plt.savefig(output_path / "CRC_cold_effects.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the statistical analysis from the generated CSV."""
    input_path = RESULTS_FOLDER / "crc_cpt_blocks.csv"
    output_dir = RESULTS_FOLDER / "statistics"

    results = run_analysis(
        input_path=input_path,
        output_dir=output_dir,
        model_choice=MODEL_CHOICE,
    )

    plot_cold_effect(results=results, output_path=output_dir)
    print("\nStatistical analysis completed.")
    print("Participants analysed:", results["data"]["participant"].nunique())
    print("Observations analysed:", len(results["data"]))

    print("\nPrimary LMM results:")
    print(results["primary_results"].to_string(index=False))
    print("\nSelected model block-specific cold-versus-warm effects:")
    print(
        results["block_effects"]
        .loc[results["block_effects"]["model"] == results["selected_model"]]
        .to_string(index=False)
    )

    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
