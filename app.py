import io, re
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, brunnermunzel, ttest_rel, ttest_ind, wilcoxon
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

st.set_page_config(page_title="Flexible Nonparametrics — Pairwise + Repeated Measures", layout="wide")
st.title("Flexible Nonparametric Analysis — Pairwise (A vs B) + Repeated Measures")
st.caption(
    "Upload an Excel workbook. The app auto-detects: (1) the outcome series (e.g., Ejaculation score, AM/PM), "
    "(2) Baseline column, and (3) any number of follow-up timepoints (Days/Weeks/Months/Visits, including Day 85). "
    "If change columns (e.g., 'Change AM') exist, they will be used directly; otherwise change is computed from baseline."
)

# ---------- Helpers ----------
def norm(s): return str(s).strip().lower()

def infer_col(df, must_have=(), any_of=()):
    for c in df.columns:
        lc = norm(c)
        if all(m in lc for m in must_have) and (not any_of or any(a in lc for a in any_of)):
            return c
    return None

SUBJ_CANDIDATES = [
    ("participant","id"), ("subject","id"), ("subject",), ("participant",), ("patient","id"), ("id",)
]
GRP_CANDIDATES = [("group",), ("arm",), ("treatment",), ("grp",), ("group_norm",)]

BASELINE_TOKENS = [
    "baseline", "base", "bl", "bl_", " bl", "bl "
]

# Regex recognizers for visits/time
TIME_PATTERNS = [
    (re.compile(r"\bday\s*(\d+)\b", re.I), lambda m: ("Day", int(m.group(1)), f"Day{m.group(1)}")),
    (re.compile(r"\bweek\s*(\d+)\b", re.I), lambda m: ("Week", int(m.group(1)), f"Week{m.group(1)}")),
    (re.compile(r"\b(\d+)\s*month(s)?\b", re.I), lambda m: ("Month", int(m.group(1)), f"Month{m.group(1)}")),
    (re.compile(r"\bvisit\s*(\d+)\b", re.I), lambda m: ("Visit", int(m.group(1)), f"Visit{m.group(1)}")),
    (re.compile(r"\bm(\d+)\b", re.I),     lambda m: ("Month", int(m.group(1)), f"Month{m.group(1)}")),  # e.g., M1
    (re.compile(r"\bv(\d+)\b", re.I),     lambda m: ("Visit", int(m.group(1)), f"Visit{m.group(1)}")),
]

CHANGE_TOKENS = ["change", "delta", "diff"]

def detect_subject_col(df):
    for keys in SUBJ_CANDIDATES:
        c = infer_col(df, keys)
        if c: return c
    # last resort
    return df.columns[0]

def detect_group_col(df):
    for keys in GRP_CANDIDATES:
        c = infer_col(df, keys)
        if c: return c
    # fallback if missing
    return None

def is_change_col(colname):
    lc = norm(colname)
    return any(tok in lc for tok in CHANGE_TOKENS)

def is_baseline_col(colname):
    lc = norm(colname)
    return any(tok in lc for tok in BASELINE_TOKENS)

def parse_time_label(text):
    """Return (order_key_tuple, canonical_label) or (None, None)."""
    s = str(text)
    for pat, maker in TIME_PATTERNS:
        m = pat.search(s)
        if m:
            kind, num, label = maker(m)
            # order as (priority, number) where Day < Week < Month < Visit for stability
            priority = {"Day":1, "Week":2, "Month":3, "Visit":4}.get(kind, 9)
            return (priority, num), label
    return (None, None)

def stem_for_series(colname):
    """Produce a series stem by removing baseline/time tokens, keeping the 'measure' part.
       E.g. 'Day 56 Ejaculation Score (0-8)' -> 'Ejaculation Score (0-8)' ;
            '1 Month AM Average' -> 'AM Average' ; 'Change AM' -> 'AM' with flag.
    """
    s = colname
    s = re.sub(r"\bday\s*\d+\b", "", s, flags=re.I)
    s = re.sub(r"\bweek\s*\d+\b", "", s, flags=re.I)
    s = re.sub(r"\b(\d+)\s*month(s)?\b", "", s, flags=re.I)
    s = re.sub(r"\bvisit\s*\d+\b", "", s, flags=re.I)
    s = re.sub(r"\bm\d+\b", "", s, flags=re.I)
    s = re.sub(r"\bv\d+\b", "", s, flags=re.I)
    s = re.sub(r"\bbaseline\b", "", s, flags=re.I)
    s = re.sub(r"\bbase\b", "", s, flags=re.I)
    s = re.sub(r"\bbl_?\b", "", s, flags=re.I)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def collect_series(df):
    """Group columns into 'series' candidates: each series has optional baseline, zero+ follow-ups, and/or change cols."""
    series = {}
    for c in df.columns:
        if c is None: continue
        s_stem = stem_for_series(str(c))
        if s_stem == "": continue
        series.setdefault(s_stem, {"baseline": [], "followups": [], "changes": []})
        if is_baseline_col(c):
            series[s_stem]["baseline"].append(c)
        elif is_change_col(c):
            series[s_stem]["changes"].append(c)
        else:
            order, label = parse_time_label(c)
            if order is not None:
                series[s_stem]["followups"].append((order, label, c))
            else:
                # leave non-time, non-baseline, non-change columns alone
                pass
    # sort follow-ups
    for s_stem in series:
        series[s_stem]["followups"].sort(key=lambda t: (t[0][0], t[0][1]))
    return series

def to_numeric_safe(s):
    return pd.to_numeric(s, errors="coerce")

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

# ---------- Builders ----------
def build_from_series(df, subj_col, group_col, series_entry, chosen_series_name):
    """
    Return long_change, change_wide, input_echo (if baseline path), visits(list), diag_msg(str or None)
    Logic:
      - If change columns exist and no clear baseline/follow-ups: use change columns directly (single timepoint 'Change').
      - Else compute change per follow-up: change = followup - baseline.
    """
    # sanitize
    df = df.rename(columns=lambda x: str(x).strip())
    subj = subj_col
    grp  = group_col

    # validate subj/group existence
    cols_needed = [subj] + ([grp] if grp else [])
    for c in cols_needed:
        if c not in df.columns:
            return None, None, None, [], f"Column '{c}' not found."

    # figure out baseline + followups + changes
    baselist = series_entry.get("baseline", [])
    followups = series_entry.get("followups", [])
    changes = series_entry.get("changes", [])

    # Strategy A: direct changes available (e.g., 'Change AM', 'Change PM')
    if len(changes) >= 1 and len(baselist) == 0 and len(followups) == 0:
        # Build change-wide with those columns as timepoints (label them as provided)
        cw = df[[subj] + ([grp] if grp else []) + changes].copy()
        cw.columns = ["Subject"] + (["Group"] if grp else []) + [c for c in changes]
        # long
        rows = []
        for _, r in cw.iterrows():
            for c in changes:
                v = r.get(c)
                if pd.notna(v):
                    rows.append({
                        "Subject": r["Subject"],
                        "Group": r["Group"] if grp else "All",
                        "Time": c,  # treat each change column name as a 'time' label
                        "Change": float(v)
                    })
        long_change = pd.DataFrame(rows)
        visits = [c for c in changes]  # preserve order
        long_change["Time"] = pd.Categorical(long_change["Time"], categories=visits, ordered=True)
        return long_change, cw, None, visits, None

    # Strategy B: compute change from baseline across available follow-ups
    # pick ONE baseline if multiple candidates; prefer the one that best matches the series name
    baseline_col = None
    if baselist:
        # heuristic: choose baseline whose stem is closest to chosen_series_name
        candidates = []
        for b in baselist:
            bstem = stem_for_series(b)
            # score: shorter distance = better
            score = 0 if norm(bstem) == norm(chosen_series_name) else 1
            candidates.append((score, b))
        candidates.sort(key=lambda x: x[0])
        baseline_col = candidates[0][1]
    else:
        return None, None, None, [], "No baseline column detected for the chosen series."

    if not followups:
        return None, None, None, [], "No follow-up timepoints detected for the chosen series."

    # Build input_echo: Baseline + followups (raw)
    use_cols = [subj] + ([grp] if grp else []) + [baseline_col] + [c for _,_,c in followups]
    inp = df[use_cols].copy()
    rename_map = {subj:"Subject"}
    if grp: rename_map[grp] = "Group"
    rename_map[baseline_col] = "Baseline"
    # canonical visit names (labels from parser)
    visits = [lbl for _,lbl,_ in followups]
    for (_, lbl, c) in followups:
        rename_map[c] = lbl
    inp = inp.rename(columns=rename_map)
    if grp:
        inp["Group"] = inp["Group"].astype(str).str.strip()

    # numeric force
    for c in ["Baseline"] + visits:
        inp[c] = to_numeric_safe(inp[c])

    # change-wide
    chg_wide = inp[["Subject"] + (["Group"] if grp else [])].copy()
    for v in visits:
        chg_wide[f"{v}_change"] = inp[v] - inp["Baseline"]

    # long
    rows = []
    for _, r in chg_wide.iterrows():
        for v in visits:
            val = r.get(f"{v}_change")
            if pd.notna(val):
                rows.append({
                    "Subject": r["Subject"],
                    "Group": r["Group"] if grp else "All",
                    "Time": v,
                    "Change": float(val)
                })
    long_change = pd.DataFrame(rows)
    long_change["Time"] = pd.Categorical(long_change["Time"], categories=visits, ordered=True)
    return long_change, chg_wide, inp, visits, None

# ---------- Stats ----------
def compute_rm_all_groups(long_change, visits):
    lc = long_change.copy()
    lc["Change"] = to_numeric_safe(lc["Change"])
    lc = lc.dropna(subset=["Group","Time","Change"]).copy()
    observed = [v for v in visits if v in lc["Time"].astype(str).unique().tolist()]
    if len(observed) < 2:
        return pd.DataFrame({
            "Effect": ["Group (all groups)", f"Time ({', '.join(observed) if observed else '—'})", "Group × Time"],
            "p-value": [np.nan, np.nan, np.nan]
        }), observed

    lc["rank_change"] = lc["Change"].rank(method="average")
    # subject random effect approximated via fixed effect in rank-based ANOVA
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
    lc["Change"] = to_numeric_safe(lc["Change"])
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

    pv_df = pd.DataFrame(rows)
    summary_df = pd.DataFrame(sumrows)
    diag_df = pd.DataFrame(diag)
    return pv_df, summary_df, valid, diag_df

def compute_within_group_tests(input_echo, visits):
    if input_echo is None or input_echo.empty:
        return pd.DataFrame()
    groups = (input_echo["Group"].dropna().astype(str).str.strip().unique().tolist()
              if "Group" in input_echo.columns else ["All"])
    rows = []
    for grp in groups:
        g = input_echo if "Group" not in input_echo.columns else input_echo[input_echo["Group"] == grp]
        base = to_numeric_safe(g["Baseline"])
        for v in visits:
            if v not in g.columns:
                rows.append({"Group": grp, "Visit": v, "N (paired)": 0,
                             "Wilcoxon stat": np.nan, "Wilcoxon p": np.nan,
                             "Paired t stat": np.nan, "Paired t p": np.nan})
                continue
            follow = to_numeric_safe(g[v])
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
uploaded = st.file_uploader("Upload a workbook (any sheet). The app will scan columns and infer series/visits.", type=["xlsx"])

if not uploaded:
    st.stop()

try:
    # Read first sheet just to list columns; we will actually scan all sheets
    xl = pd.ExcelFile(uploaded)
except Exception as e:
    st.error(f"Could not read Excel: {e}")
    st.stop()

# Pick a sheet
sheet = st.selectbox("Choose sheet", xl.sheet_names, index=0)
df = pd.read_excel(uploaded, sheet_name=sheet)
st.write("Preview:")
st.dataframe(df.head(8))

# Detect subject / group
subj_col = detect_subject_col(df)
grp_col  = detect_group_col(df)
with st.sidebar:
    st.header("Column mapping")
    subj_col = st.selectbox("Subject ID column", df.columns.tolist(), index=df.columns.get_loc(subj_col) if subj_col in df.columns else 0)
    grp_col = st.selectbox("Group column (or 'None' if absent)", ["<None>"] + df.columns.tolist(), index=(0 if (not grp_col or grp_col not in df.columns) else df.columns.get_loc(grp_col)+1))
    if grp_col == "<None>": grp_col = None

# Build series options
series = collect_series(df)
if not series:
    st.error("Could not detect any outcome series (no baseline/visit/change patterns found).")
    st.stop()

series_names = sorted(series.keys())
with st.sidebar:
    st.header("Outcome series")
    chosen_series = st.selectbox("Pick the outcome series to analyze", series_names, index=0)

series_entry = series[chosen_series]
st.markdown(f"**Detected series:** `{chosen_series}`")
st.write({
    "Baseline candidates": series_entry.get("baseline", []),
    "Follow-ups": [f"{lbl} ← {col}" for _,lbl,col in series_entry.get("followups", [])],
    "Change cols": series_entry.get("changes", []),
})

# Build long/change from the chosen series
long_change, change_wide, input_echo, VISITS, err = build_from_series(df, subj_col, grp_col, series_entry, chosen_series)
if err:
    st.error(err)
    st.stop()

# Available groups
all_groups = sorted(long_change["Group"].dropna().astype(str).str.strip().unique().tolist())
with st.sidebar:
    st.header("Pairwise comparison")
    if len(all_groups) < 1:
        st.error("No groups detected.")
        st.stop()
    if len(all_groups) == 1:
        group_a = all_groups[0]
        group_b = all_groups[0]
        st.info(f"Only one group detected: {group_a}. Pairwise tests will be limited.")
    else:
        default_pair = (all_groups[0], all_groups[1])
        group_a = st.selectbox("Group A", all_groups, index=all_groups.index(default_pair[0]))
        group_b = st.selectbox("Group B", all_groups, index=all_groups.index(default_pair[1]))
        if group_a == group_b:
            st.warning("Pick two different groups for pairwise tests.")

st.subheader("2) Long-format 'Change' (preview)")
st.dataframe(long_change.head(10))

# RM across ALL groups (only if >=2 timepoints)
st.subheader("3) Repeated-measures across ALL groups")
rm_df, observed = compute_rm_all_groups(long_change, VISITS)
st.dataframe(rm_df)

# Pairwise between-group tests
if len(all_groups) >= 2 and group_a != group_b:
    st.subheader(f"4) Pairwise between-group tests — {group_a} vs {group_b}")
    pv_df, summary_df, valid_visits, diag_df = compute_pairwise_between(long_change, VISITS, group_a, group_b)
    st.markdown("**P_VALUES (per visit)**")
    st.dataframe(pv_df)

    with st.expander("Diagnostics (per-visit Ns for selected pair)"):
        st.dataframe(diag_df)

    # Final-visit MW added to RM table (primary endpoint for selected pair)
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
            st.markdown("**RM_NONPARAM_SUMMARY (with final-visit MW for selected pair)**")
            st.dataframe(rm_df)
        else:
            st.info("Final-visit Mann–Whitney not added: one of the groups has no data at the final visit.")
else:
    pv_df = pd.DataFrame()
    summary_df = pd.DataFrame()
    valid_visits = []
    st.info("Pairwise tests require two distinct groups.")

# Within-group for ALL groups (only when baseline path used)
st.subheader("5) Within-group tests (Baseline → each visit, all groups)")
within_df = compute_within_group_tests(input_echo if input_echo is not None else pd.DataFrame(), observed or VISITS)
st.dataframe(within_df)

# Welch t (pairwise on selected pair)
if len(all_groups) >= 2 and group_a != group_b:
    st.subheader(f"6) Welch t-tests on change (pairwise: {group_a} vs {group_b})")
    ttests_df = compute_welch_on_changes(long_change, valid_visits or observed or VISITS, group_a, group_b)
    st.dataframe(ttests_df)
else:
    ttests_df = pd.DataFrame()

# Chart for selected pair
st.subheader("7) Chart (selected pair)")
if len(all_groups) >= 2 and group_a != group_b and not summary_df.empty and (valid_visits or observed):
    chart_buf = make_chart(summary_df, valid_visits or observed, group_a, group_b)
    if chart_buf is None:
        st.info("Chart skipped: not enough data for selected pair.")
    else:
        st.image(chart_buf, caption=f"Median Δ with IQR — {group_a} vs {group_b}", use_column_width=True)
else:
    chart_buf = None
    st.info("Chart available when two distinct groups and summary stats are present.")

# Build Excel
st.subheader("8) Download results")
out = io.BytesIO()
with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
    # Echo raw selections
    if input_echo is not None and not input_echo.empty:
        input_echo.to_excel(writer, sheet_name="INPUT_ECHO", index=False)
    if change_wide is not None and not change_wide.empty:
        change_wide.to_excel(writer, sheet_name="CHANGE_WIDE", index=False)
    if not summary_df.empty:
        summary_df.to_excel(writer, sheet_name="SUMMARY_PAIR", index=False)
    if 'pv_df' in locals() and not pv_df.empty:
        pv_df.to_excel(writer, sheet_name="P_VALUES_PAIR", index=False)
    if not rm_df.empty:
        rm_df.to_excel(writer, sheet_name="RM_NONPARAM_SUMMARY", index=False)
    if not within_df.empty:
        within_df.to_excel(writer, sheet_name="WITHIN_GROUP_TESTS", index=False)
    if 'ttests_df' in locals() and not ttests_df.empty:
        ttests_df.to_excel(writer, sheet_name="TTESTS_PAIR", index=False)

    # chart
    workbook = writer.book
    ws_chart = workbook.add_worksheet("CHARTS")
    writer.sheets["CHARTS"] = ws_chart
    if chart_buf is not None:
        ws_chart.insert_image("B2", "chart.png", {"image_data": chart_buf})
    else:
        ws_chart.write("B2", "Chart unavailable for selected pair.")

out.seek(0)
base = uploaded.name.rsplit(".",1)[0]
st.download_button(
    "⬇️ Download Excel (includes INPUT/CHANGE/SUMMARY_PAIR/P_VALUES_PAIR/RM/WITHIN/TTESTS/CHARTS as available)",
    data=out,
    file_name=f"{base}_RESULTS_FLEX.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
