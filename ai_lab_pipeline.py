"""
AI Lab Pipeline
Single-script deliverable for scraping, cleaning, analysis, and output generation.

How to run:
    python ai_lab_pipeline.py

Outputs created in the same folder as this script:
    - firm_profiles.csv
    - clean_firm_profiles.csv
    - propensity_score_distribution.png
    - smd_comparison.csv
    - matched_data.csv
    - analysis_summary.txt
"""

# GitHub Copilot Prompt History:
# "Scrape the firm profiles from the unstructured HTML table, extract the firm name from the nested <strong> tag,
# build headers from the first row, and save the raw dataset to CSV in the script folder."

# GitHub Copilot Prompt History:
# "Clean Annual Rev., Revenue Growth, R&D Spend, Team Size, Digital Sales, and Customer Accts into numeric form.
# Replace N/A, --, Unknown, and blanks with missing values. Standardize AI Program into a binary variable."

# GitHub Copilot Prompt History:
# "Run a naive OLS regression of Revenue Growth on AI adoption, estimate propensity scores, plot common support,
# compute SMD before and after matching, perform nearest-neighbor matching, and re-estimate the AI effect."

# GitHub Copilot Prompt History:
# "Save all deliverables in the script folder and print the key results needed for the written interpretation."

from pathlib import Path
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from bs4 import BeautifulSoup
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RAW_CSV_PATH = BASE_DIR / "firm_profiles.csv"
CLEAN_CSV_PATH = BASE_DIR / "clean_firm_profiles.csv"
PLOT_PATH = BASE_DIR / "propensity_score_distribution.png"
SMD_CSV_PATH = BASE_DIR / "smd_comparison.csv"
MATCHED_CSV_PATH = BASE_DIR / "matched_data.csv"
SUMMARY_TXT_PATH = BASE_DIR / "analysis_summary.txt"

URL = "https://bana290-assignment1.netlify.app/"


# ============================================================
# Helper Functions
# ============================================================

def normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Replace common string placeholders for missing values."""
    missing_tokens = [
        "N/A", "n/a", "NA", "na", "--", "Unknown", "unknown", "", " ", "None", "none"
    ]
    return df.replace(missing_tokens, np.nan)


def parse_human_number(value):
    """
    Convert human-readable numeric strings to floats.

    Examples:
        "$141,638,144" -> 141638144
        "USD 110.70M" -> 110700000
        "104.62 million" -> 104620000
        "$15.31 mn" -> 15310000
        "12.3% rev" -> 12.3
        "73%" -> 73
        "339.9K" -> 339900
        "1.07M" -> 1070000
        "2,000" -> 2000
    """
    if pd.isna(value):
        return np.nan

    s = str(value).strip().lower()
    if s in {"", "nan", "none", "n/a", "--", "unknown"}:
        return np.nan

    s = s.replace(",", "")
    s = s.replace("$", "")
    s = s.replace("usd", "")
    s = s.strip()

    multiplier = 1.0
    if "billion" in s:
        multiplier = 1_000_000_000
        s = s.replace("billion", "")
    elif "million" in s:
        multiplier = 1_000_000
        s = s.replace("million", "")
    elif re.search(r"\bmn\b", s):
        multiplier = 1_000_000
        s = re.sub(r"\bmn\b", "", s)
    elif re.search(r"\bk\b", s):
        multiplier = 1_000
        s = re.sub(r"\bk\b", "", s)
    elif re.search(r"\bm\b", s):
        multiplier = 1_000_000
        s = re.sub(r"\bm\b", "", s)
    elif re.search(r"\bb\b", s):
        multiplier = 1_000_000_000
        s = re.sub(r"\bb\b", "", s)

    # keep the first numeric pattern
    match = re.search(r"-?\d+\.?\d*", s)
    if not match:
        return np.nan

    return float(match.group()) * multiplier


def standardize_ai_program(value):
    """
    Convert AI program labels to binary.
    1 = more advanced / adopted / active AI usage
    0 = no meaningful AI adoption
    """
    if pd.isna(value):
        return np.nan

    s = str(value).strip().lower()

    yes_values = {
        "yes", "adopted", "ai enabled", "enabled", "production", "live"
    }
    no_values = {
        "no", "not yet", "manual only", "legacy only"
    }
    uncertain_values = {
        "pilot", "in review"
    }

    if s in yes_values:
        return 1
    if s in no_values:
        return 0
    if s in uncertain_values:
        return np.nan

    return np.nan


def smd_numeric(data: pd.DataFrame, treatment_col: str, covariate_col: str):
    """Compute standardized mean difference for a numeric covariate."""
    treated = data.loc[data[treatment_col] == 1, covariate_col].dropna()
    control = data.loc[data[treatment_col] == 0, covariate_col].dropna()

    if len(treated) == 0 or len(control) == 0:
        return np.nan

    mean_t = treated.mean()
    mean_c = control.mean()
    var_t = treated.var(ddof=1)
    var_c = control.var(ddof=1)

    pooled_sd = np.sqrt((var_t + var_c) / 2)
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return 0.0

    return (mean_t - mean_c) / pooled_sd


def run_ols(data: pd.DataFrame, y_col: str, x_cols: list[str]):
    """Fit an OLS model with a constant."""
    X = sm.add_constant(data[x_cols].copy())
    y = data[y_col].copy()
    return sm.OLS(y, X).fit()


# ============================================================
# Stage 1 — Scrape
# ============================================================

def scrape_directory_table(url: str) -> pd.DataFrame:
    """Scrape the unstructured HTML table from the assignment site."""
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("No HTML table found on the page.")

    rows = table.find_all("tr")
    if not rows:
        raise ValueError("No table rows found.")

    # Extract headers from first row
    header_cells = rows[0].find_all(["th", "td"])
    headers_out = [cell.get_text(" ", strip=True) for cell in header_cells]

    data_rows = []
    for row in rows[1:]:
        cols = row.find_all("td")
        if not cols:
            continue

        row_data = []

        # Firm name is nested in <strong>, with extra metadata in the same cell.
        firm_cell = cols[0]
        strong_tag = firm_cell.find("strong")
        if strong_tag:
            firm_name = strong_tag.get_text(" ", strip=True)
        else:
            firm_name = firm_cell.get_text(" ", strip=True)

        row_data.append(firm_name)

        # Remaining cells
        for col in cols[1:]:
            row_data.append(col.get_text(" ", strip=True))

        # Keep only rows that match the header length
        if len(row_data) == len(headers_out):
            data_rows.append(row_data)

    if not data_rows:
        raise ValueError("No valid data rows were extracted.")

    df_raw = pd.DataFrame(data_rows, columns=headers_out)
    return df_raw


# ============================================================
# Stage 2 — Clean
# ============================================================

def clean_scraped_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean the scraped dataset and return an analysis-ready DataFrame."""
    df = df_raw.copy()
    df = normalize_missing(df)

    # Clean numeric-like columns
    numeric_cols = [
        "Founded", "Team Size", "Annual Rev.", "Rev Growth (YoY)",
        "R&D Spend", "Digital Sales", "Customer Accts"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(parse_human_number)

    # Binary treatment column
    if "AI Program" in df.columns:
        df["AI Program"] = df["AI Program"].apply(standardize_ai_program)

    return df


# ============================================================
# Stage 3 — Analyze
# ============================================================

def analyze_data(df_clean: pd.DataFrame):
    """
    Run baseline OLS, propensity score estimation, common support plot,
    SMD before/after matching, nearest-neighbor matching, and post-match OLS.
    """
    required_cols = [
        "Rev Growth (YoY)", "AI Program",
        "Founded", "Team Size", "Annual Rev.", "Digital Sales", "Customer Accts"
    ]
    missing_required = [c for c in required_cols if c not in df_clean.columns]
    if missing_required:
        raise ValueError(f"Missing required columns for analysis: {missing_required}")

    analysis_cols = required_cols.copy()
    analysis_df = df_clean[analysis_cols].copy()

    # Restrict to clean treatment indicator and observed analysis columns
    analysis_df = analysis_df[analysis_df["AI Program"].isin([0, 1])]
    analysis_df = analysis_df.dropna().copy()

    if analysis_df.empty:
        raise ValueError("No rows available for analysis after cleaning and filtering.")

    # Baseline OLS
    baseline_model = run_ols(analysis_df, "Rev Growth (YoY)", ["AI Program"])
    baseline_coef = baseline_model.params["AI Program"]
    baseline_p = baseline_model.pvalues["AI Program"]

    # Propensity scores
    covariates = ["Founded", "Team Size", "Annual Rev.", "Digital Sales", "Customer Accts"]
    X_ps = analysis_df[covariates].copy()
    y_ps = analysis_df["AI Program"].copy()

    logit = LogisticRegression(max_iter=5000)
    logit.fit(X_ps, y_ps)
    analysis_df["propensity_score"] = logit.predict_proba(X_ps)[:, 1]

    # Common support plot
    treated_ps = analysis_df.loc[analysis_df["AI Program"] == 1, "propensity_score"]
    control_ps = analysis_df.loc[analysis_df["AI Program"] == 0, "propensity_score"]

    plt.figure(figsize=(8, 5))
    plt.hist(treated_ps, bins=12, alpha=0.6, label="AI Program = 1", density=True)
    plt.hist(control_ps, bins=12, alpha=0.6, label="AI Program = 0", density=True)
    plt.xlabel("Propensity Score")
    plt.ylabel("Density")
    plt.title("Distribution of Propensity Scores by Treatment Group")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=300)
    plt.close()

    # SMD before matching
    smd_before_records = []
    for cov in covariates:
        smd_before_records.append({
            "Covariate": cov,
            "SMD_Before": smd_numeric(analysis_df, "AI Program", cov)
        })
    smd_before_df = pd.DataFrame(smd_before_records)

    # Nearest-neighbor matching without replacement on the propensity score
    treated = analysis_df[analysis_df["AI Program"] == 1].copy().reset_index(drop=True)
    control = analysis_df[analysis_df["AI Program"] == 0].copy().reset_index(drop=True)

    if treated.empty or control.empty:
        raise ValueError("Both treatment and control groups are required for matching.")

    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[["propensity_score"]])

    distances, indices = nn.kneighbors(treated[["propensity_score"]])

    used_control = set()
    matched_pairs = []
    for i, control_idx in enumerate(indices.flatten()):
        if control_idx in used_control:
            continue
        used_control.add(control_idx)
        matched_pairs.append((i, int(control_idx), float(distances[i][0])))

    if not matched_pairs:
        raise ValueError("No matched pairs were formed.")

    matched_treated_rows = []
    matched_control_rows = []
    for pair_id, (t_idx, c_idx, dist) in enumerate(matched_pairs, start=1):
        t_row = treated.iloc[t_idx].copy()
        c_row = control.iloc[c_idx].copy()

        t_row["matched_pair_id"] = pair_id
        c_row["matched_pair_id"] = pair_id
        t_row["match_distance"] = dist
        c_row["match_distance"] = dist

        matched_treated_rows.append(t_row)
        matched_control_rows.append(c_row)

    matched_df = pd.concat(
        [pd.DataFrame(matched_treated_rows), pd.DataFrame(matched_control_rows)],
        ignore_index=True
    )
    matched_df.to_csv(MATCHED_CSV_PATH, index=False)

    # SMD after matching
    smd_after_records = []
    for cov in covariates:
        smd_after_records.append({
            "Covariate": cov,
            "SMD_After": smd_numeric(matched_df, "AI Program", cov)
        })
    smd_after_df = pd.DataFrame(smd_after_records)

    smd_compare = smd_before_df.merge(smd_after_df, on="Covariate", how="outer")
    smd_compare["Abs_SMD_Before"] = smd_compare["SMD_Before"].abs()
    smd_compare["Abs_SMD_After"] = smd_compare["SMD_After"].abs()
    smd_compare.to_csv(SMD_CSV_PATH, index=False)

    # Post-matching OLS
    matched_model = run_ols(matched_df, "Rev Growth (YoY)", ["AI Program"])
    matched_coef = matched_model.params["AI Program"]
    matched_p = matched_model.pvalues["AI Program"]

    # Summary text
    summary_lines = []
    summary_lines.append("=== ANALYSIS SUMMARY ===\n")
    summary_lines.append("1) Baseline Correlation: Naive OLS\n")
    summary_lines.append(f"Coefficient on AI Program: {baseline_coef:.6f}\n")
    summary_lines.append(f"P-value: {baseline_p:.6f}\n\n")
    summary_lines.append("2) Propensity Score Model Covariates\n")
    summary_lines.append(f"{covariates}\n\n")
    summary_lines.append("3) Matching\n")
    summary_lines.append(f"Number of treated observations before matching: {len(treated)}\n")
    summary_lines.append(f"Number of control observations before matching: {len(control)}\n")
    summary_lines.append(f"Number of matched pairs: {len(matched_pairs)}\n\n")
    summary_lines.append("4) Post-Matching OLS\n")
    summary_lines.append(f"Coefficient on AI Program: {matched_coef:.6f}\n")
    summary_lines.append(f"P-value: {matched_p:.6f}\n\n")
    summary_lines.append("5) SMD Comparison\n")
    summary_lines.append(smd_compare.to_string(index=False))
    summary_lines.append("\n")

    with open(SUMMARY_TXT_PATH, "w", encoding="utf-8") as f:
        f.writelines(summary_lines)

    return {
        "analysis_df": analysis_df,
        "matched_df": matched_df,
        "baseline_coef": baseline_coef,
        "baseline_p": baseline_p,
        "matched_coef": matched_coef,
        "matched_p": matched_p,
        "matched_pairs": len(matched_pairs),
        "treated_n": len(treated),
        "control_n": len(control),
        "smd_compare": smd_compare,
    }


# ============================================================
# Main Pipeline
# ============================================================

def main():
    print("Stage 1 — Scrape")
    df_raw = scrape_directory_table(URL)
    df_raw.to_csv(RAW_CSV_PATH, index=False)
    print(f"Saved raw file: {RAW_CSV_PATH}")

    print("\nStage 2 — Clean")
    df_clean = clean_scraped_data(df_raw)
    df_clean.to_csv(CLEAN_CSV_PATH, index=False)
    print(f"Saved cleaned file: {CLEAN_CSV_PATH}")

    print("\nStage 3 — Analyze")
    results = analyze_data(df_clean)
    print(f"Saved common support plot: {PLOT_PATH}")
    print(f"Saved SMD table: {SMD_CSV_PATH}")
    print(f"Saved matched data: {MATCHED_CSV_PATH}")
    print(f"Saved analysis summary: {SUMMARY_TXT_PATH}")

    print("\nKey Results")
    print(f"Baseline AI coefficient: {results['baseline_coef']:.6f}")
    print(f"Baseline p-value: {results['baseline_p']:.6f}")
    print(f"Post-match AI coefficient: {results['matched_coef']:.6f}")
    print(f"Post-match p-value: {results['matched_p']:.6f}")
    print(f"Matched pairs: {results['matched_pairs']}")


if __name__ == "__main__":
    main()
