import io
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import mannwhitneyu, brunnermunzel
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

st.set_page_config(page_title="Nonparam P-values (Active vs Placebo)", layout="wide")

st.title("Nonparametric P-values — Active vs Placebo (QOL/IPSS-style)")
st.caption("Uploads your Excel, auto-detects layout (Input or Change Wide), runs Mann–Whitney, Brunner–Munzel, and rank-based RM on change-from-baseline.")

# ---------- Settings ----------
DEFAULT_VISITS = ["Day28", "Day56", "Day84"]

with st.sidebar:
    st.header("Settings")
    visits_text = st.text_input("Visits (comma-separated)", ",".join(DEFAULT_VISITS))
    VISITS = [v.strip() for v in visits_text.split(",") if v.strip()]
    st.caption("E.g., Day28, Day56, Day84")

# ---------- Helpers ----------
def norm(s): return str(s).strip().lower()

def infer_col(df, must_have=(), any_of=()):
    for c in df.columns:
        lc = norm(c)
        if all(m in lc for m in must_have) and (not any_of or any(a in lc for a in any_of)):
            return c
    return None

def build_long_from_input(df, visits):
    # Expect raw QOL/IPSS: SubjectID, Group, Baseline, Day 28, Day 56, Day 84 (names flexible)
    df = df.rename(columns=lambda x: str(x).strip())
    subj = infer_col(df, ("subject","id")) or "Subject ID"
    grp  = infer_col(df, ("group",)) or "Group"
    base = infer_col(df, ("baseline",), ("qol","score","ipss","total","symptom")) or infer_col(df, ("baseline",))
    if not base:
        return None, None, "Baseline column not found."

    visit_map = {}
    for v in visits:
        aliases = [v.lower(), v.lower().replace("day","day ")]  # "day28" or "day 28"
        col = None
        for c in df.columns:
            lc = norm(c)
            if any(a in lc for a in aliases) and any(w in lc for w in ("qol","score","ipss","total","symptom")):
                col = c; break
        if not col:
            # last resort: accept any column containing the day alias
            for c in df.columns:
                if any(a in norm(c) for a in aliases):
                    col = c; break
        if not col:
            return None, None, f"Could not find raw column for {v}."
        visit_map[v] = col

    use = df[[subj, grp, base] + list(visit_map.values())].copy()
    use.columns = ["Subject","Group","Baseline"] + visits
    use = use.dropna(subset=["Subject","Group"])
    use["Group"] = use["Group"].astype(str).str.strip()
    use = use[use["Group"].isin(["Active","Placebo"])]

    # compute change-from-baseline
    for v in visits:
        use[f"{v}_change"] = use[v] - use["Baseline"]

    # build long format for RM analysis
    rows = []
    for _, r in use.iterrows():
        for v in visits:
            val = r.get(f"{v}_change")
            if pd.notna(val):
                rows.append({"Subject": r["Subject"], "Group": r["Group"], "Time": v, "Change": float(val)})
    long_change = pd.DataFrame(rows)
    if long_change.empty:
        return None, None, "No change values computed (check raw columns)."
    long_change["Time"] = pd.Categorical(long_change["Time"], categories=visits, ordered=True)

    echo_df = use[["Subject","Group"] + [f"{v}_change" for v in visits]].copy()
    return long_change, echo_df, None

def build_long_from_changewide(df, visits):
    # Expect *_change columns (names flexible: contain day + change/delta/diff)
    df = df.rename(columns=lambda x: str(x).strip())
    subj = infer_col(df, ("subject","id")) or "Subject ID"
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
            # accept exact e.g., "Day28_change"
            exact = f"{v.lower()}_change"
            for c in df.columns:
                if norm(c) == exact:
                    col = c; break
        if not col:
            return None, "Missing change column for " + v

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
        return None, "No change values found."
    long_change["Time"] = pd.Categorical(long_change["Time"], categories=visits, ordered=True)
    return long_change, None

def compute_stats(long_change, visits):
    # rank-based RM (subject-blocked)
    long_change = long_change.copy()
    long_change["rank_change"] = long_change["Change"].rank(method="average")
    model = ols("rank_change ~ C(Subject) + C(Group) * C(Time)", data=long_change).fit()
    an = anova_lm(model, typ=3)
    group_p = float(an.loc["C(Group)", "PR(>F)"])
    time_p  = float(an.loc["C(Time)", "PR(>F)"])
    inter_p = float(an.loc["C(Group):C(Time)", "PR(>F)"])

    # per-visit MWU, BM
    rows = []
    for v in visits:
        a = long_change.loc[(long_change["Group"]=="Active") & (long_change["Time"]==v), "Change"]
        p = long_change.loc[(long_change["Group"]=="Placebo") & (long_change["Time"]==v), "Change"]

        # Mann–Whitney
        U, mw_p = (np.nan, np.nan)
        if len(a) > 0 and len(p) > 0:
            U, mw_p = mannwhitneyu(a, p, alternative="two-sided")

        # Brunner–Munzel (needs >3 per group typically)
        bm_stat, bm_p = (np.nan, np.nan)
        if len(a) > 3 and len(p) > 3:
            bm_stat, bm_p = brunnermunzel(a, p, alternative="two-sided")

        rows.append({
            "Visit": v,
            "N Active": int(len(a)),
            "N Placebo": int(len(p)),
            "Active median Δ": float(np.median(a)) if len(a)>0 else np.nan,
            "Placebo median Δ": float(np.median(p)) if len(p)>0 else np.nan,
            "Mann–Whitney U": float(U) if U==U else np.nan,
            "Mann–Whitney p": float(mw_p) if mw_p==mw_p else np.nan,
            "Brunner–Munzel stat": float(bm_stat) if bm_stat==bm_stat else np.nan,
            "Brunner–Munzel p": float(bm_p) if bm_p==bm_p else np.nan
        })

    pv_df = pd.DataFrame(rows)
    rm_df = pd.DataFrame({
        "Effect": ["Group (Active vs Placebo)", f"Time ({', '.join(visits)})", "Group × Time interaction"],
        "p-value": [group_p, time_p, inter_p]
    })
    return pv_df, rm_df

# ---------- UI ----------
st.subheader("1) Upload your Excel (.xlsx)")
uploaded = st.file_uploader("Upload a workbook that contains either an 'Input' sheet (raw) or a 'Change Wide' sheet (changes).", type=["xlsx"])

mode = st.radio("Layout detection priority", ["Auto (detect)", "Prefer Change Wide", "Prefer Input"], index=0)

if uploaded:
    try:
        xl = pd.ExcelFile(uploaded)
    except Exception as e:
        st.error(f"Could not read Excel: {e}")
        st.stop()

    long_change = None
    echo_df = None
    used_mode = None
    used_sheet = None

    # Try Change Wide first (or per user preference)
    sheet_order = xl.sheet_names
    if mode == "Prefer Input":
        sheet_order = sheet_order[::-1]  # just to alter try order

    # Detection
    # 1) Try Change Wide
    if mode in ["Auto (detect)", "Prefer Change Wide"]:
        for s in xl.sheet_names:
            try:
                df = pd.read_excel(uploaded, sheet_name=s)
                lc, err = build_long_from_changewide(df, VISITS)
                if lc is not None and err is None and not lc.empty:
                    long_change = lc; used_mode, used_sheet = "CHANGE", s
                    break
            except Exception:
                pass

    # 2) Try Input
    if long_change is None:
        for s in xl.sheet_names:
            try:
                df = pd.read_excel(uploaded, sheet_name=s)
                lc, echo, err = build_long_from_input(df, VISITS)
                if lc is not None and err is None and not lc.empty:
                    long_change = lc; echo_df = echo; used_mode, used_sheet = "RAW", s
                    break
            except Exception:
                pass

    if long_change is None or long_change.empty:
        st.error("Could not auto-detect a usable sheet.\n\n"
                 "Expected either:\n"
                 " • A Change Wide sheet with change columns (headers containing day 28/56/84 + change/delta/diff), OR\n"
                 " • An Input sheet with Baseline and Day 28/56/84 raw QOL/IPSS scores.")
        st.stop()

    st.success(f"Detected: **{used_mode}** on sheet **{used_sheet}**")

    # Preview
    st.subheader("2) Preview (first 10 rows of long-format change)")
    st.dataframe(long_change.head(10))

    # Compute
    st.subheader("3) Compute nonparametric tests")
    pv_df, rm_df = compute_stats(long_change, VISITS)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**P_VALUES** (per visit)")
        st.dataframe(pv_df)
    with c2:
        st.markdown("**RM_NONPARAM_SUMMARY**")
        st.dataframe(rm_df)

    # Build Excel in memory
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
        if echo_df is not None:
            echo_df.to_excel(writer, sheet_name="Change Wide (computed)", index=False)
        pv_df.to_excel(writer, sheet_name="P_VALUES", index=False)
        rm_df.to_excel(writer, sheet_name="RM_NONPARAM_SUMMARY", index=False)
    out.seek(0)

    st.subheader("4) Download results")
    base = uploaded.name.rsplit(".",1)[0]
    st.download_button(
        label="⬇️ Download Excel with results",
        data=out,
        file_name=f"{base}_RESULTS.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.caption("Notes: Brunner–Munzel needs ≥4 observations per group per visit. Group labels must be exactly 'Active' and 'Placebo'.")
