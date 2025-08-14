import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, brunnermunzel, ttest_rel, ttest_ind, wilcoxon
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

st.set_page_config(page_title="Nonparam P-Values (Active vs Placebo)", layout="wide")
st.title("Nonparametric P-Values — Active vs Placebo")
st.caption(
    "Upload an Excel with either an **Input** sheet (raw Baseline & Day 28/56/84 columns, "
    "e.g., QOL/IPSS/etc.) or a **Change Wide** sheet (Day28_change/Day56_change/Day84_change). "
    "The app auto-detects layout, builds CHANGE_WIDE & SUMMARY, renders a chart, and computes "
    "Mann–Whitney, Brunner–Munzel, effect sizes (Cliff’s δ, Hodges–Lehmann), "
    "paired Wilcoxon & paired t (within-group), Welch t on changes (between-group), "
    "rank-based repeated-measures (Group, Time, Group×Time), and adds a PRIMARY ENDPOINT row: "
    "Final-visit Mann–Whitney (A vs P)."
)

# ------------------ Settings ------------------
DEFAULT_VISITS = ["Day28", "Day56", "Day84"]
with st.sidebar:
    st.header("Settings")
    visits_text = st.text_input("Visits (comma-separated)", ",".join(DEFAULT_VISITS))
    VISITS = [v.strip() for v in visits_text.split(",") if v.strip()]
    st.caption("E.g., Day28, Day56, Day84")

# ------------------ Helpers -------------------
def norm(s): return str(s).strip().lower()

def infer_col(df, must_have=(), any_of=()):
    """Find first column whose name contains all 'must_have' and any of 'any_of' (if provided)."""
    for c in df.columns:
        lc = norm(c)
        if all(m in lc for m in must_have) and (not any_of or any(a in lc for a in any_of)):
            return c
    return None

def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m, n = len(a), len(b)
    if m == 0 or n == 0: return np.nan
    wins = 0
    for x in a:
        wins += np.sum(x > b) - np.sum(x < b)
    return wins / (m * n)

def hodges_lehmann(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0: return np.nan
    diffs = a.reshape(-1,1) - b.reshape(1,-1)
    return float(np.median(diffs))

# ---------- Input -> CHANGE_WIDE ----------
def build_long_from_input(df, visits):
    df = df.rename(columns=lambda x: str(x).strip())

    # Subject detection (lenient)
    subj = (infer_col(df, ("subject",)) or
            infer_col(df, ("participant","id")) or
            infer_col(df, ("subject","id")) or
            infer_col(df, ("id",)) or
            "Subject")
    grp  = infer_col(df, ("group",)) or "Group"

    # Baseline detection: accept baseline/base/BL_ variants
    base = (infer_col(df, ("baseline",)) or
            infer_col(df, ("base",)) or
            infer_col(df, ("bl_",)) or
            infer_col(df, ("bl ",)) or
            infer_col(df, (" bl",)) or
            infer_col(df, ("bl",)))
    if not base:
        return None, None, None, "Baseline column not found (looked for baseline/base/BL_)."

    # Visit columns: look for day markers; trailing metric text can be anything
    visit_map = {}
    for v in visits:
        aliases = [v.lower(), v.lower().replace("day","day ")]  # day28 / day 28
        col = None
        for c in df.columns:
            lc = norm(c)
            if any(a in lc for a in aliases):
                col = c; break
        if not col:
            return None, None, None, f"Raw column for {v} not found."
        visit_map[v] = col

    inp_echo = df[[subj, grp, base] + list(visit_map.values())].copy()
    inp_echo.columns = ["Subject","Group","Baseline"] + visits
    inp_echo = inp_echo.dropna(subset=["Subject","Group"])
    inp_echo["Group"] = inp_echo["Group"].astype(str).str.strip()
    inp_echo = inp_echo[inp_echo["Group"].isin(["Active","Placebo"])]

    # change-from-baseline
    chg_wide = inp_echo[["Subject","Group"]].copy()
    for v in visits:
        chg_wide[f"{v}_change"] = inp_echo[v] - inp_echo["Baseline"]

    # long format for RM and per-visit tests
    rows = []
    for _, r in chg_wide.iterrows():
        for v in visits:
            val = r.get(f"{v}_change")
            if pd.notna(val):
                rows.append({"Subject": r["Subject"], "Group": r["Group"], "Time": v, "Change": float(val)})
    long_change = pd.DataFrame(rows)
    if long_change.empty:
        return None, None, None, "No change values computed."
    long_change["Time"] = pd.Categorical(long_change["Time"], categories=visits, ordered=True)
    return long_change, chg_wide.copy(), inp_echo.copy(), None

# ---------- CHANGE_WIDE (already) ----------
def build_long_from_changewide(df, visits):
    df = df.rename(columns=lambda x: str(x).strip())

    subj = (infer_col(df, ("subject",)) or
            infer_col(df, ("participant","id")) or
            infer_col(df, ("subject","id")) or
            infer_col(df, ("id",)) or
            "Subject")
    grp  = infer_col(df, ("group",)) or "Group"

    change_words = ("change","delta","diff")
    vmap = {}
    for v in visits:
        aliases = [v.lower(), v.lower().replace("day","day ")]
        col = None
        for c in df.columns:
            lc = norm(c)
            if any(a in lc for a in aliases) and any(w in lc for w in change_words):
                col = c; break
        if not col:
            exact = f"{v.lower()}_change"
            for c in df.columns:
                if norm(c) == exact:
                    col = c; break
        if not col:
            return None, None, f"Change column for {v} not found."
        vmap[v] = col

    use = df[[subj, grp] + list(vmap.values())].copy()
    use.columns = ["Subject","Group"] + visits
    use["Group"] = use["Group"].astype(str).str.strip()
    use = use[use["Group"].isin(["Active","Placebo"])]

    rows = []
    for _, r in use.iterrows():
        for v in visits:
            val = r[v]
            if pd.notna(val):
                rows.append({"Subject": r["Subject"], "Group": r["Group"], "Time": v, "Change": float(val)})
    long_change = pd.DataFrame(rows)
    if long_change.empty:
        return None, None, "No change values found."
    long_change["Time"] = pd.Categorical(long_change["Time"], categories=visits, ordered=True)
    return long_change, use.copy(), None

# ---------- Core stats with robust RM + FINAL MW row ----------
def compute_nonparam_between_and_rm(long_change, visits):
    # Working copy
    lc = long_change.copy()

    # Keep only visits present AND with both groups represented
    observed_times = lc["Time"].dropna().astype(str).unique().tolist()
    observed_times = [v for v in visits if v in observed_times]
    valid_visits = []
    for v in observed_times:
        nA = lc[(lc["Time"] == v) & (lc["Group"] == "Active")].shape[0]
        nP = lc[(lc["Time"] == v) & (lc["Group"] == "Placebo")].shape[0]
        if nA > 0 and nP > 0:
            valid_visits.append(v)

    # Filter to valid visits only
    lc = lc[lc["Time"].isin(valid_visits)].copy()
    lc["Time"] = pd.Categorical(lc["Time"], categories=valid_visits, ordered=True)

    # Rank-based RM (subject-blocked) — only if ≥2 valid time points
    if len(valid_visits) < 2:
        group_p = time_p = inter_p = np.nan
    else:
        lc["rank_change"] = lc["Change"].rank(method="average")
        model = ols("rank_change ~ C(Subject) + C(Group) * C(Time)", data=lc).fit()
        an = anova_lm(model, typ=3)
        group_p = float(an.loc["C(Group)", "PR(>F)"])
        time_p  = float(an.loc["C(Time)", "PR(>F)"])
        inter_p = float(an.loc["C(Group):C(Time)", "PR(>F)"])

    # Per-visit tests (use same filtered visits)
    rows = []
    sumrows = []
    for v in valid_visits:
        a = lc.loc[(lc["Group"]=="Active") & (lc["Time"]==v), "Change"]
        p = lc.loc[(lc["Group"]=="Placebo") & (lc["Time"]==v), "Change"]

        # Mann–Whitney
        U, mw_p = (np.nan, np.nan)
        if len(a)>0 and len(p)>0:
            U, mw_p = mannwhitneyu(a, p, alternative="two-sided")

        # Brunner–Munzel (≥4 per group typical)
        bm_stat, bm_p = (np.nan, np.nan)
        if len(a)>3 and len(p)>3:
            bm_stat, bm_p = brunnermunzel(a, p, alternative="two-sided")

        # Effect sizes
        d_cliff = cliffs_delta(a, p)
        hl = hodges_lehmann(a, p)

        # Rank-biserial r (direction by median diff)
        r_rb = np.nan
        if len(a)>0 and len(p)>0 and not np.isnan(U):
            n1, n2 = len(a), len(p)
            r_rb = 1 - 2 * U / (n1 * n2)
            med_diff = (np.median(a) - np.median(p)) if (len(a) and len(p)) else 0.0
            if np.sign(r_rb) == 0 and med_diff != 0:
                r_rb = np.sign(med_diff) * abs(r_rb)

        rows.append({
            "Visit": v,
            "N Active": int(len(a)), "N Placebo": int(len(p)),
            "Active median Δ": float(np.median(a)) if len(a)>0 else np.nan,
            "Placebo median Δ": float(np.median(p)) if len(p)>0 else np.nan,
            "Mann–Whitney U": float(U) if U==U else np.nan,
            "Mann–Whitney p": float(mw_p) if mw_p==mw_p else np.nan,
            "Brunner–Munzel stat": float(bm_stat) if bm_stat==bm_stat else np.nan,
            "Brunner–Munzel p": float(bm_p) if bm_p==bm_p else np.nan,
            "Cliff's delta": float(d_cliff) if d_cliff==d_cliff else np.nan,
            "Hodges–Lehmann (Δ A−P)": float(hl) if hl==hl else np.nan,
            "Rank-biserial r": float(r_rb) if r_rb==r_rb else np.nan,
        })

        # Summary for chart (median & IQR)
        def qstats(x):
            if len(x)==0: return (np.nan, np.nan, np.nan, np.nan, np.nan)
            med = np.median(x); q1 = np.percentile(x, 25); q3 = np.percentile(x, 75)
            return med, q1, q3, (q3-med), (med-q1)

        amed, aq1, aq3, aplus, aminus = qstats(a)
        pmed, pq1, pq3, pplus, pminus = qstats(p)
        sumrows.append({
            "Time": v,
            "Active_median": amed, "Active_Q1": aq1, "Active_Q3": aq3,
            "Placebo_median": pmed, "Placebo_Q1": pq1, "Placebo_Q3": pq3,
            "Active_plus": aplus, "Active_minus": aminus,
            "Placebo_plus": pplus, "Placebo_minus": pminus
        })

    pv_df = pd.DataFrame(rows)

    # --- RM (global) summary table ---
    rm_df = pd.DataFrame({
        "Effect": ["Group (Active vs Placebo)",
                   f"Time ({', '.join(valid_visits)})",
                   "Group × Time interaction"],
        "p-value": [group_p, time_p, inter_p]
    })

    # --- PRIMARY ENDPOINT: Final visit Mann–Whitney (A vs P) ---
    final_visit = valid_visits[-1] if len(valid_visits) > 0 else None
    if final_visit is not None:
        a_final = lc.loc[(lc["Group"]=="Active") & (lc["Time"]==final_visit), "Change"].dropna()
        p_final = lc.loc[(lc["Group"]=="Placebo") & (lc["Time"]==final_visit), "Change"].dropna()
        if len(a_final) > 0 and len(p_final) > 0:
            _, mw_p_final = mannwhitneyu(a_final, p_final, alternative="two-sided")
            rm_df = pd.concat([
                rm_df,
                pd.DataFrame({
                    "Effect": [f"Final visit Mann–Whitney (A vs P) at {final_visit}"],
                    "p-value": [float(mw_p_final)]
                })
            ], ignore_index=True)

    summary_df = pd.DataFrame(sumrows)
    return pv_df, rm_df, summary_df, valid_visits

def compute_within_group_tests(input_echo, visits):
    """Within each group, paired Wilcoxon & paired t: Baseline vs each visit."""
    if input_echo is None or input_echo.empty:
        return pd.DataFrame()
    rows = []
    for grp in ["Active", "Placebo"]:
        g = input_echo[input_echo["Group"] == grp]
        base = g["Baseline"]
        for v in visits:
            follow = g[v] if v in g.columns else None
            if follow is None:
                rows.append({"Group": grp, "Visit": v, "N (paired)": 0,
                             "Wilcoxon stat": np.nan, "Wilcoxon p": np.nan,
                             "Paired t stat": np.nan, "Paired t p": np.nan})
                continue
            paired = pd.concat([base, follow], axis=1).dropna()
            if paired.shape[0] == 0:
                w_stat = w_p = t_stat = t_p = np.nan
            else:
                try:
                    w_stat, w_p = wilcoxon(paired.iloc[:,0], paired.iloc[:,1],
                                           alternative="two-sided", zero_method="wilcox")
                except Exception:
                    w_stat, w_p = (np.nan, np.nan)
                try:
                    t_stat, t_p = ttest_rel(paired.iloc[:,0], paired.iloc[:,1])
                except Exception:
                    t_stat, t_p = (np.nan, np.nan)
            rows.append({
                "Group": grp, "Visit": v, "N (paired)": int(paired.shape[0]),
                "Wilcoxon stat": float(w_stat) if w_stat==w_stat else np.nan,
                "Wilcoxon p": float(w_p) if w_p==w_p else np.nan,
                "Paired t stat": float(t_stat) if t_stat==t_stat else np.nan,
                "Paired t p": float(t_p) if t_p==t_p else np.nan,
            })
    return pd.DataFrame(rows)

def compute_ttests(long_change, visits):
    """Between-group Welch t-tests on change-from-baseline per visit."""
    rows = []
    for v in visits:
        a = long_change.loc[(long_change["Group"]=="Active") & (long_change["Time"]==v), "Change"].dropna()
        p = long_change.loc[(long_change["Group"]=="Placebo") & (long_change["Time"]==v), "Change"].dropna()
        if len(a)>1 and len(p)>1:
            t_stat, t_p = ttest_ind(a, p, equal_var=False)
        else:
            t_stat, t_p = (np.nan, np.nan)
        rows.append({"Visit": v, "N Active": int(len(a)), "N Placebo": int(len(p)),
                     "Welch t stat": float(t_stat) if t_stat==t_stat else np.nan,
                     "Welch t p": float(t_p) if t_p==t_p else np.nan})
    return pd.DataFrame(rows)

def make_chart(summary_df, visits):
    # Guard: if summary is empty or missing required cols, return None
    required = {
        "Active_median","Active_minus","Active_plus",
        "Placebo_median","Placebo_minus","Placebo_plus","Time"
    }
    if summary_df is None or summary_df.empty or not required.issubset(set(summary_df.columns)):
        return None

    # Ensure correct order / subset to visits actually present
    summary_df = summary_df.copy().set_index("Time").reindex(visits).reset_index()

    fig, ax = plt.subplots(figsize=(8,5))
    x = np.arange(len(summary_df))

    a_med = summary_df["Active_median"].to_numpy()
    a_lo  = a_med - summary_df["Active_minus"].to_numpy()
    a_hi  = a_med + summary_df["Active_plus"].to_numpy()

    p_med = summary_df["Placebo_median"].to_numpy()
    p_lo  = p_med - summary_df["Placebo_minus"].to_numpy()
    p_hi  = p_med + summary_df["Placebo_plus"].to_numpy()

    ax.plot(x, a_med, marker="o", label="Active (median Δ)")
    ax.fill_between(x, a_lo, a_hi, alpha=0.3, label="Active IQR")
    ax.plot(x, p_med, marker="o", label="Placebo (median Δ)")
    ax.fill_between(x, p_lo, p_hi, alpha=0.3, label="Placebo IQR")

    ax.set_xticks(x, visits)
    ax.set_xlabel("Visit"); ax.set_ylabel("Change from Baseline")
    ax.set_title("Median Change with IQR (Active vs Placebo)"); ax.legend()
    fig.tight_layout()

    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150); buf.seek(0)
    return buf

# ------------------ UI -------------------
st.subheader("1) Upload your Excel (.xlsx)")
uploaded = st.file_uploader(
    "Upload a workbook that contains either an 'Input' sheet (raw) or a 'Change Wide' sheet (changes).",
    type=["xlsx"]
)

if uploaded:
    try:
        xl = pd.ExcelFile(uploaded)
    except Exception as e:
        st.error(f"Could not read Excel: {e}")
        st.stop()

    long_change = None; change_wide = None; input_echo = None
    used_mode = None; used_sheet = None

    # Try Change Wide first
    for s in xl.sheet_names:
        try:
            df = pd.read_excel(uploaded, sheet_name=s)
            lc, chg_wide, err = build_long_from_changewide(df, VISITS)
            if lc is not None and err is None and not lc.empty:
                long_change = lc; change_wide = chg_wide; used_mode, used_sheet = "CHANGE", s; break
        except Exception:
            pass

    # If not found, try Input (raw)
    if long_change is None:
        for s in xl.sheet_names:
            try:
                df = pd.read_excel(uploaded, sheet_name=s)
                lc, chg_wide, inp_echo, err = build_long_from_input(df, VISITS)
                if lc is not None and err is None and not lc.empty:
                    long_change = lc; change_wide = chg_wide; input_echo = inp_echo; used_mode, used_sheet = "RAW", s; break
            except Exception:
                pass

    if long_change is None or long_change.empty:
        st.error(
            "Could not auto-detect a usable sheet.\n\n"
            "Expected either:\n"
            " • A Change Wide sheet with change columns (headers containing day 28/56/84 + change/delta/diff), OR\n"
            " • An Input sheet with Baseline and Day 28/56/84 raw scores (QOL/IPSS/etc.)."
        ); st.stop()

    st.success(f"Detected: **{used_mode}** on sheet **{used_sheet}**")

    st.subheader("2) Preview (long-format change)")
    st.dataframe(long_change.head(10))

    st.subheader("3) Compute tests")
    pv_df, rm_df, summary_df, valid_visits = compute_nonparam_between_and_rm(long_change, VISITS)
    within_df = compute_within_group_tests(input_echo, valid_visits if input_echo is not None else valid_visits)
    ttests_df = compute_ttests(long_change, valid_visits)

    c1, c2 = st.columns(2)
    with c1: st.markdown("**P_VALUES (between-group, nonparam + effect sizes)**"); st.dataframe(pv_df)
    with c2: st.markdown("**RM_NONPARAM_SUMMARY**"); st.dataframe(rm_df)

    st.markdown("**WITHIN_GROUP_TESTS (Baseline → each visit)**")
    st.dataframe(within_df)

    st.markdown("**TTESTS (between-group Welch on change)**")
    st.dataframe(ttests_df)

    # Per-visit counts to diagnose missing data
    with st.expander("Per-visit N by group (after filtering)"):
        counts = []
        for v in valid_visits:
            nA = long_change[(long_change["Time"]==v) & (long_change["Group"]=="Active")]["Change"].notna().sum()
            nP = long_change[(long_change["Time"]==v) & (long_change["Group"]=="Placebo")]["Change"].notna().sum()
            counts.append({"Visit": v, "N Active": int(nA), "N Placebo": int(nP)})
        st.dataframe(pd.DataFrame(counts))

    st.subheader("4) Chart")
    chart_buf = make_chart(summary_df, valid_visits)
    if chart_buf is None:
        st.info("Chart skipped: not enough data to compute medians/IQR for both groups at any visit.")
    else:
        st.image(chart_buf, caption="Median change with IQR (Active vs Placebo)", use_column_width=True)

    # Build Excel in memory
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        if input_echo is not None: input_echo.to_excel(writer, sheet_name="INPUT_ECHO", index=False)
        change_wide.to_excel(writer, sheet_name="CHANGE_WIDE", index=False)
        summary_df.to_excel(writer, sheet_name="SUMMARY", index=False)
        pv_df.to_excel(writer, sheet_name="P_VALUES", index=False)
        rm_df.to_excel(writer, sheet_name="RM_NONPARAM_SUMMARY", index=False)
        within_df.to_excel(writer, sheet_name="WITHIN_GROUP_TESTS", index=False)
        ttests_df.to_excel(writer, sheet_name="TTESTS", index=False)

        # CHART sheet
        workbook  = writer.book
        ws_chart  = workbook.add_worksheet("CHARTS")
        writer.sheets["CHARTS"] = ws_chart
        if chart_buf is not None:
            ws_chart.insert_image("B2", "chart.png", {"image_data": chart_buf})
        else:
            ws_chart.write("B2", "Chart unavailable: insufficient data in both groups at the selected visits.")

    out.seek(0)

    st.subheader("5) Download Excel with all tabs")
    base = uploaded.name.rsplit(".",1)[0]
    st.download_button(
        label="⬇️ Download Excel (INPUT/CHANGE/SUMMARY/CHARTS/P_VALUES/RM/WITHIN/TTESTS)",
        data=out,
        file_name=f"{base}_RESULTS.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.caption("Notes: Group labels must be exactly 'Active' and 'Placebo'. Brunner–Munzel needs ≥4 observations per group per visit.")
