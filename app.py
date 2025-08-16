import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, brunnermunzel, ttest_rel, ttest_ind, wilcoxon
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

st.set_page_config(page_title="Nonparam P-Values — Pairwise + Repeated Measures", layout="wide")
st.title("Nonparametric Analysis — Pairwise (USPlus vs Placebo) + Repeated Measures")
st.caption(
    "Upload an Excel with raw columns (Baseline & Day 28/56/85 or Final). "
    "Pick the outcome (e.g., Ejaculation/Erection). Map columns even if headers differ. "
    "Pairwise is forced to USPlus vs Placebo when both are present."
)

# ---------- Helpers ----------
def norm(s): return str(s).strip().lower()
def to_num(s): return pd.to_numeric(s, errors="coerce")

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

def guess_col(df_cols, keywords):
    """Return first column whose lowercase name contains ALL keywords."""
    for c in df_cols:
        lc = norm(c)
        if all(k in lc for k in keywords):
            return c
    return None

def compute_rm_all_groups(long_change, visits):
    lc = long_change.copy()
    lc["Change"] = to_num(lc["Change"])
    lc = lc.dropna(subset=["Group","Time","Change"]).copy()
    observed = [v for v in visits if v in lc["Time"].astype(str).unique().tolist()]
    # Need at least 2 time points to do RM-style effects
    if len(observed) < 2:
        return pd.DataFrame({
            "Effect": ["Group (all groups)", f"Time ({', '.join(observed) if observed else '—'})", "Group × Time"],
            "p-value": [np.nan, np.nan, np.nan]
        }), observed

    lc["rank_change"] = lc["Change"].rank(method="average")
    model = ols("rank_change ~ C(Subject) + C(Group) * C(Time)", data=lc).fit()
    an = anova_lm(model, typ=3)
    rm_df = pd.DataFrame({
        "Effect": ["Group (all groups)", f"Time ({', '.join(observed)})", "Group × Time"],
        "p-value": [
            float(an.loc["C(Group)","PR(>F)"]),
            float(an.loc["C(Time)","PR(>F)"]),
            float(an.loc["C(Group):C(Time)","PR(>F)"]),
        ]
    })
    return rm_df, observed

def compute_pairwise_between(long_change, visits, group_a, group_b):
    lc = long_change[long_change["Group"].isin([group_a, group_b])].copy()
    lc["Change"] = to_num(lc["Change"])
    lc = lc.dropna(subset=["Change","Time","Group"]).copy()

    valid, diag = [], []
    for v in [vv for vv in visits if vv in lc["Time"].astype(str).unique().tolist()]:
        nA = lc[(lc["Time"]==v) & (lc["Group"]==group_a)]["Change"].notna().sum()
        nB = lc[(lc["Time"]==v) & (lc["Group"]==group_b)]["Change"].notna().sum()
        if nA>0 and nB>0: valid.append(v)
        diag.append({"Visit": v, f"N {group_a}": int(nA), f"N {group_b}": int(nB),
                     "Status": "kept" if (nA>0 and nB>0) else "dropped: need both groups"})

    rows, sumrows = [], []
    for v in valid:
        a = lc[(lc["Time"]==v) & (lc["Group"]==group_a)]["Change"]
        b = lc[(lc["Time"]==v) & (lc["Group"]==group_b)]["Change"]

        U = mw_p = bm_stat = bm_p = np.nan
        if len(a)>0 and len(b)>0:
            U, mw_p = mannwhitneyu(a, b, alternative="two-sided")
            if len(a)>3 and len(b)>3:
                bm_stat, bm_p = brunnermunzel(a, b, alternative="two-sided")

        d = cliffs_delta(a, b) if len(a)>0 and len(b)>0 else np.nan
        hl = hodges_lehmann(a, b) if len(a)>0 and len(b)>0 else np.nan
        r_rb = np.nan
        if len(a)>0 and len(b)>0 and not np.isnan(U):
            r_rb = 1 - 2 * U / (len(a)*len(b))

        rows.append({
            "Visit": v,
            f"N {group_a}": int(len(a)),
            f"N {group_b}": int(len(b)),
            f"{group_a} median Δ": float(np.median(a)) if len(a)>0 else np.nan,
            f"{group_b} median Δ": float(np.median(b)) if len(b)>0 else np.nan,
            "Mann–Whitney U": float(U) if U==U else np.nan,
            "Mann–Whitney p": float(mw_p) if mw_p==mw_p else np.nan,
            "Brunner–Munzel stat": float(bm_stat) if bm_stat==bm_stat else np.nan,
            "Brunner–Munzel p": float(bm_p) if bm_p==bm_p else np.nan,
            "Cliff's delta": float(d) if d==d else np.nan,
            "Hodges–Lehmann (Δ A−B)": float(hl) if hl==hl else np.nan,
            "Rank-biserial r": float(r_rb) if r_rb==r_rb else np.nan,
        })

        def qstats(x):
            if len(x)==0: return (np.nan, np.nan, np.nan, np.nan, np.nan)
            med = np.median(x); q1 = np.percentile(x,25); q3 = np.percentile(x,75)
            return med, q1, q3, (q3-med), (med-q1)

        amed, aq1, aq3, aplus, aminus = qstats(a)
        bmed, bq1, bq3, bplus, bminus = qstats(b)
        sumrows.append({
            "Time": v,
            f"{group_a}_median": amed, f"{group_a}_Q1": aq1, f"{group_a}_Q3": aq3,
            f"{group_b}_median": bmed, f"{group_b}_Q1": bq1, f"{group_b}_Q3": bq3,
            f"{group_a}_plus": aplus, f"{group_a}_minus": aminus,
            f"{group_b}_plus": bplus, f"{group_b}_minus": bminus
        })

    return pd.DataFrame(rows), pd.DataFrame(sumrows), valid, pd.DataFrame(diag)

def compute_within_group_tests(input_echo, visits):
    if input_echo is None or input_echo.empty:
        return pd.DataFrame()
    groups = sorted(input_echo["Group"].dropna().astype(str).str.strip().unique().tolist())
    rows = []
    for grp in groups:
        g = input_echo[input_echo["Group"] == grp]
        base = to_num(g["Baseline"])
        for v in visits:
            if v not in g.columns:
                rows.append({"Group": grp, "Visit": v, "N (paired)": 0,
                             "Wilcoxon stat": np.nan, "Wilcoxon p": np.nan,
                             "Paired t stat": np.nan, "Paired t p": np.nan})
                continue
            follow = to_num(g[v])
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

def compute_welch_on_changes(long_change, visits, group_a, group_b):
    rows = []
    lc = long_change[long_change["Group"].isin([group_a, group_b])].copy()
    for v in visits:
        a = lc[(lc["Time"]==v) & (lc["Group"]==group_a)]["Change"].dropna()
        b = lc[(lc["Time"]==v) & (lc["Group"]==group_b)]["Change"].dropna()
        if len(a)>1 and len(b)>1:
            t_stat, t_p = ttest_ind(a, b, equal_var=False)
        else:
            t_stat, t_p = (np.nan, np.nan)
        rows.append({"Visit": v, f"N {group_a}": int(len(a)), f"N {group_b}": int(len(b)),
                     "Welch t stat": float(t_stat) if t_stat==t_stat else np.nan,
                     "Welch t p": float(t_p) if t_p==t_p else np.nan})
    return pd.DataFrame(rows)

def make_chart(summary_df, visits, group_a, group_b):
    needed = {f"{group_a}_median", f"{group_a}_minus", f"{group_a}_plus",
              f"{group_b}_median", f"{group_b}_minus", f"{group_b}_plus", "Time"}
    if summary_df is None or summary_df.empty or not needed.issubset(set(summary_df.columns)):
        return None
    df = summary_df.copy().set_index("Time").reindex(visits).reset_index()
    fig, ax = plt.subplots(figsize=(8,5))
    x = np.arange(len(df))

    a_med = df[f"{group_a}_median"].to_numpy()
    a_lo  = a_med - df[f"{group_a}_minus"].to_numpy()
    a_hi  = a_med + df[f"{group_a}_plus"].to_numpy()
    b_med = df[f"{group_b}_median"].to_numpy()
    b_lo  = b_med - df[f"{group_b}_minus"].to_numpy()
    b_hi  = b_med + df[f"{group_b}_plus"].to_numpy()

    ax.plot(x, a_med, marker="o", label=f"{group_a} (median Δ)")
    ax.fill_between(x, a_lo, a_hi, alpha=0.3, label=f"{group_a} IQR")
    ax.plot(x, b_med, marker="o", label=f"{group_b} (median Δ)")
    ax.fill_between(x, b_lo, b_hi, alpha=0.3, label=f"{group_b} IQR")

    ax.set_xticks(x, visits); ax.set_xlabel("Visit"); ax.set_ylabel("Change from Baseline")
    ax.set_title(f"Median Change with IQR — {group_a} vs {group_b}")
    ax.legend(); fig.tight_layout()

    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=150); buf.seek(0)
    return buf

# ---------- UI ----------
st.subheader("1) Upload your Excel (.xlsx)")
uploaded = st.file_uploader("Workbook with an 'Input-like' sheet (raw BL + follow-ups).", type=["xlsx"])
if not uploaded:
    st.stop()

try:
    xl = pd.ExcelFile(uploaded)
except Exception as e:
    st.error(f"Could not read Excel: {e}")
    st.stop()

sheet = st.selectbox("Choose sheet", xl.sheet_names, index=0)
df = pd.read_excel(uploaded, sheet_name=sheet)
st.write("Preview:")
st.dataframe(df.head(8))

# Basic IDs (guessers)
subj_guess = None
for cand in ["Participant ID","Subject ID","Subject","ID","Participant","Patient ID","Patient"]:
    if cand in df.columns: subj_guess = cand; break
if subj_guess is None: subj_guess = df.columns[0]

grp_guess = None
for cand in ["GROUP","Group","Arm","Treatment","group"]:
    if cand in df.columns: grp_guess = cand; break

with st.sidebar:
    st.header("Column mapping")
    subj_col = st.selectbox("Subject ID column", df.columns.tolist(), index=df.columns.get_loc(subj_guess))
    grp_col = st.selectbox("Group column", df.columns.tolist(), index=(df.columns.get_loc(grp_guess) if grp_guess in df.columns else 0))

# Outcome focus
with st.sidebar:
    st.header("Outcome")
    quick = st.selectbox("Pick outcome keyword", ["Ejaculation", "Erection", "Custom"], index=0)
    outcome_kw = quick if quick != "Custom" else st.text_input("Header keyword (e.g., IPSS, Frequency, QoL)", "Ejaculation")

# Guess columns (Baseline + Day 28/56/84/85/Final)
cols = df.columns.tolist()
bl_guess   = (guess_col(cols, ["bl", norm(outcome_kw)]) or guess_col(cols, ["baseline", norm(outcome_kw)]))
d28_guess  = guess_col(cols, ["day", "28", norm(outcome_kw)])
d56_guess  = guess_col(cols, ["day", "56", norm(outcome_kw)])
d84_guess  = guess_col(cols, ["day", "84", norm(outcome_kw)])
d85_guess  = guess_col(cols, ["day", "85", norm(outcome_kw)])
final_guess = (guess_col(cols, ["final", norm(outcome_kw)]) or guess_col(cols, ["end", norm(outcome_kw)]) )

# UI mapping
st.subheader("2) Map outcome columns")
col1, col2 = st.columns(2)
with col1:
    bl_col  = st.selectbox("Baseline column", ["<None>"] + cols, index=(cols.index(bl_guess)+1 if bl_guess in cols else 0))
with col2:
    d28_col = st.selectbox("Day 28 column", ["<None>"] + cols, index=(cols.index(d28_guess)+1 if d28_guess in cols else 0))

col3, col4 = st.columns(2)
with col3:
    d56_col = st.selectbox("Day 56 column", ["<None>"] + cols, index=(cols.index(d56_guess)+1 if d56_guess in cols else 0))
with col4:
    d85_col = st.selectbox("Day 85 (or 84) column", ["<None>"] + cols,
                           index=(cols.index(d85_guess)+1 if d85_guess in cols else (cols.index(d84_guess)+1 if d84_guess in cols else 0)))

# New: Final/Last Visit mapping (for BL+FINAL datasets)
final_col = st.selectbox("Final / Last Visit column (optional)", ["<None>"] + cols,
                         index=(cols.index(final_guess)+1 if final_guess in cols else 0))

# Build mapping and visit list (order matters)
mapped = {
    "Baseline": None if bl_col=="<None>" else bl_col,
    "Day28": None if d28_col=="<None>" else d28_col,
    "Day56": None if d56_col=="<None>" else d56_col,
    "Day85": None if d85_col=="<None>" else d85_col,
    "Final": None if final_col=="<None>" else final_col,
}
# Keep order: Day28, Day56, Day85, Final
VISITS = [v for v in ["Day28","Day56","Day85","Final"] if mapped[v] is not None]

if mapped["Baseline"] is None or len(VISITS)==0:
    st.error("Please map at least a Baseline and one follow-up (Day 28/56/85 or Final).")
    st.stop()

# Build INPUT_ECHO and CHANGE_WIDE
inp = df[[subj_col, grp_col, mapped["Baseline"]] + [mapped[v] for v in VISITS]].copy()
rename_map = {subj_col:"Subject", grp_col:"Group", mapped["Baseline"]:"Baseline"}
for v in VISITS:
    rename_map[mapped[v]] = v
inp = inp.rename(columns=rename_map)
inp["Group"] = inp["Group"].astype(str).str.strip()
for c in ["Baseline"] + VISITS:
    inp[c] = to_num(inp[c])

change_wide = inp[["Subject","Group"]].copy()
for v in VISITS:
    change_wide[f"{v}_change"] = inp[v] - inp["Baseline"]

# Long format
rows = []
for _, r in change_wide.iterrows():
    for v in VISITS:
        val = r.get(f"{v}_change")
        if pd.notna(val):
            rows.append({"Subject": r["Subject"], "Group": r["Group"], "Time": v, "Change": float(val)})
long_change = pd.DataFrame(rows)
long_change["Time"] = pd.Categorical(long_change["Time"], categories=VISITS, ordered=True)

# Restrict pair to USPlus vs Placebo when possible
all_groups = sorted(long_change["Group"].dropna().astype(str).str.strip().unique().tolist())
if {"USPlus","Placebo"}.issubset(set(all_groups)):
    group_a, group_b = "USPlus", "Placebo"
else:
    group_a, group_b = (all_groups + all_groups)[:2]  # safe even if 1 group

with st.sidebar:
    st.header("Pairwise comparison")
    st.write("Locked to **USPlus vs Placebo** when both exist.")
    group_a = st.selectbox("Group A", all_groups, index=all_groups.index(group_a) if group_a in all_groups else 0)
    group_b = st.selectbox("Group B", all_groups, index=all_groups.index(group_b) if group_b in all_groups else (1 if len(all_groups)>1 else 0))
    if group_a == group_b:
        st.warning("Pick two different groups for pairwise tests.")

st.subheader("3) Long-format change (preview)")
st.dataframe(long_change.head(10))

# RM across ALL groups (skips automatically if only 1 visit)
st.subheader("4) Repeated-measures across ALL groups")
rm_df, observed = compute_rm_all_groups(long_change, VISITS)
st.dataframe(rm_df)

# Pairwise between-group tests
st.subheader(f"5) Pairwise between-group tests — {group_a} vs {group_b}")
pv_df, summary_df, valid_visits, diag_df = compute_pairwise_between(long_change, VISITS, group_a, group_b)
st.markdown("**P_VALUES (per visit)**")
st.dataframe(pv_df)
with st.expander("Diagnostics (per-visit Ns for selected pair)"):
    st.dataframe(diag_df)

# Add final-visit Mann–Whitney to RM table (works for any single visit incl. Final)
if len(valid_visits) > 0:
    final_visit = valid_visits[-1]
    a_final = long_change[(long_change["Group"]==group_a) & (long_change["Time"]==final_visit)]["Change"].dropna()
    b_final = long_change[(long_change["Group"]==group_b) & (long_change["Time"]==final_visit)]["Change"].dropna()
    if len(a_final)>0 and len(b_final)>0:
        _, mw_p_final = mannwhitneyu(a_final, b_final, alternative="two-sided")
        rm_df = pd.concat([rm_df, pd.DataFrame({
            "Effect": [f"Final visit Mann–Whitney ({group_a} vs {group_b}) at {final_visit}"],
            "p-value": [float(mw_p_final)]
        })], ignore_index=True)
        st.markdown("**RM_NONPARAM_SUMMARY (with final-visit MW)**")
        st.dataframe(rm_df)

# Within-group tests (Baseline → each visit) — includes Final-only case
st.subheader("6) Within-group tests (Baseline → each visit)")
within_df = compute_within_group_tests(inp, observed or VISITS)
st.dataframe(within_df)

# Welch t on change (between groups)
st.subheader(f"7) Welch t-tests on change — {group_a} vs {group_b}")
ttests_df = compute_welch_on_changes(long_change, valid_visits or observed or VISITS, group_a, group_b)
st.dataframe(ttests_df)

# Chart
st.subheader("8) Chart (selected pair)")
if not summary_df.empty and (valid_visits or observed):
    chart_buf = make_chart(summary_df, valid_visits or observed, group_a, group_b)
    if chart_buf is None:
        st.info("Chart skipped: not enough data for selected pair.")
    else:
        st.image(chart_buf, caption=f"Median Δ with IQR — {group_a} vs {group_b}", use_column_width=True)
else:
    chart_buf = None
    st.info("Chart available when two distinct groups and summary stats are present.")

# Build Excel
st.subheader("9) Download results")
out = io.BytesIO()
with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
    inp.to_excel(writer, sheet_name="INPUT_ECHO", index=False)
    change_wide.to_excel(writer, sheet_name="CHANGE_WIDE", index=False)
    if not summary_df.empty: summary_df.to_excel(writer, sheet_name="SUMMARY_PAIR", index=False)
    if not pv_df.empty: pv_df.to_excel(writer, sheet_name="P_VALUES_PAIR", index=False)
    if not rm_df.empty: rm_df.to_excel(writer, sheet_name="RM_NONPARAM_SUMMARY", index=False)
    if not within_df.empty: within_df.to_excel(writer, sheet_name="WITHIN_GROUP_TESTS", index=False)
    if not ttests_df.empty: ttests_df.to_excel(writer, sheet_name="TTESTS_PAIR", index=False)
    ws_chart = writer.book.add_worksheet("CHARTS")
    if chart_buf is not None:
        ws_chart.insert_image("B2", "chart.png", {"image_data": chart_buf})
    else:
        ws_chart.write("B2", "Chart unavailable for selected pair.")

out.seek(0)
base = uploaded.name.rsplit(".",1)[0]
st.download_button(
    "⬇️ Download Excel",
    data=out,
    file_name=f"{base}_RESULTS_PAIRWISE.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
