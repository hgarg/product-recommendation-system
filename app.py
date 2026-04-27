"""
Product Recommendation System — Streamlit GUI

"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib          # line 9
matplotlib.use('Agg')      # line 10  ← must come before pyplot
import matplotlib.pyplot as plt  # line 11
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import time
import warnings
warnings.filterwarnings('ignore')

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Product Recommendation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --primary:   #2E4057;
        --accent:    #048A81;
        --warm:      #E76F51;
        --light-bg:  #EAF0F6;
        --text-muted:#6b7280;
        --fs-base:   16px;
    }

    /* Streamlit top-right run/stop buttons — make them visible and branded */
    [data-testid="stToolbar"] { visibility: visible !important; opacity: 1 !important; }
    [data-testid="stStatusWidget"] { visibility: visible !important; }
    header[data-testid="stHeader"] { background-color: #2E4057 !important; }
    header[data-testid="stHeader"] button {
        background: #048A81 !important;
        color: white !important;
        border-radius: 6px !important;
        border: none !important;
        font-weight: 600 !important;
    }
    header[data-testid="stHeader"] button:hover {
        background: #E76F51 !important;
    }

    /* Base font size — driven by accessibility slider */
    body, .stMarkdown, .stText, label, p { font-size: var(--fs-base) !important; }

    /* High contrast overrides (applied via .hc class on body) */
    body.hc { background: #000 !important; color: #fff !important; }
    body.hc .metric-card { background:#111 !important; border-color:#fff !important; color:#fff !important; }
    body.hc .metric-card .val { color:#fff !important; }
    body.hc .pipeline-bar { background:#111 !important; border-color:#fff !important; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1e2d3d; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] small, [data-testid="stSidebar"] .stMarkdown p { color: #b0bec5 !important; }
    [data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; font-size: 0.95rem; }

    /* Pipeline status bar */
    .pipeline-bar {
        display: flex; align-items: center; gap: 6px;
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 10px 18px;
        margin-bottom: 8px; flex-wrap: wrap;
    }
    .phase-badge {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 4px 12px; border-radius: 20px;
        font-size: 0.78rem; font-weight: 600; letter-spacing: 0.03em;
    }
    .phase-done   { background:#d1fae5; color:#065f46; }
    .phase-active { background:#dbeafe; color:#1e40af; }
    .phase-wait   { background:#f1f5f9; color:#475569; }
    .arrow-sep    { color:#94a3b8; font-size:1rem; }

    /* Top-right hint bar */
    .topright-hint {
        font-size: 0.75rem; color: #4b5563; text-align: right;
        margin-bottom: 6px; font-style: italic;
    }

    /* Section header */
    .section-header {
        background: linear-gradient(90deg, #2E4057, #048A81);
        color: white; padding: 12px 20px; border-radius: 8px;
        margin-bottom: 16px; font-size: 1.1rem; font-weight: 700;
        letter-spacing: 0.04em;
    }

    /* Metric cards */
    .metric-row { display: flex; gap: 14px; margin-bottom: 16px; flex-wrap: wrap; }
    .metric-card {
        flex: 1; min-width: 130px; background: white;
        border: 1px solid #e2e8f0; border-top: 3px solid #2E4057;
        border-radius: 8px; padding: 14px 16px; text-align: center;
    }
    .metric-card .val { font-size: 1.6rem; font-weight: 700; color: #2E4057; line-height: 1.1; }
    .metric-card .lbl { font-size: 0.73rem; color: #64748b; margin-top: 4px;
                        text-transform: uppercase; letter-spacing: 0.05em; }

    /* Banners */
    .banner-ok   { background:#d1fae5; border-left:4px solid #10b981; padding:10px 14px;
                   border-radius:6px; color:#065f46; font-size:0.87rem; margin:8px 0; }
    .banner-err  { background:#fee2e2; border-left:4px solid #ef4444; padding:10px 14px;
                   border-radius:6px; color:#991b1b; font-size:0.87rem; margin:8px 0; }
    .banner-info { background:#dbeafe; border-left:4px solid #3b82f6; padding:10px 14px;
                   border-radius:6px; color:#1e3a8a; font-size:0.87rem; margin:8px 0; }

    /* Info box for model type */
    .model-info-box {
        background: #EAF0F6; border-left: 4px solid #048A81;
        padding: 10px 14px; border-radius: 6px; font-size: 0.85rem;
        color: #1e3a5f; margin: 8px 0 14px 0;
    }

    /* Prev / Next nav buttons */
    .nav-row {
        display: flex; justify-content: space-between;
        align-items: center; margin-top: 28px; padding-top: 16px;
        border-top: 1px solid #e2e8f0;
    }
    .step-label { font-size: 0.75rem; color:#64748b; text-align:center; flex:1; }

    /* Table */
    .dataframe thead th { background-color: #2E4057 !important; color: white !important; }

    /* Tooltip helper */
    .help-text { font-size: 0.78rem; color: #64748b; font-style: italic; margin-top: -6px; margin-bottom: 8px; }

    /* Focus outline for keyboard nav */
    button:focus, input:focus, select:focus, [role="radio"]:focus {
        outline: 3px solid #048A81 !important; outline-offset: 2px !important;
    }

    /* Footer */
    .footer { text-align: center; color: #94a3b8; font-size: 0.75rem;
              margin-top: 40px; padding-top: 16px; border-top: 1px solid #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ─── Session state initialisation ─────────────────────────────────────────────
def init_state():
    defaults = {
        'df_raw':          None,
        'df_meta':         None,
        'df_clean':        None,
        'df_meta_c':       None,
        'tfidf_matrix':    None,
        'pid_idx':         None,
        'pids':            None,
        'train_df':        None,
        'test_df':         None,
        'R_predicted':     None,
        'user_enc':        None,
        'item_enc':        None,
        'results':         None,
        'coverage':        None,
        'sparsity_raw':    None,
        'sparsity_clean':  None,
        # Pipeline status: 0=wait, 1=done
        'status_obtain':   0,
        'status_scrub':    0,
        'status_explore':  0,
        'status_model':    0,
        'status_interpret':0,
        'current_step':    0,
        'high_contrast':   False,
        'font_size':       16,
        'colorblind':      False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─── Apply accessibility settings dynamically ─────────────────────────────────
_fs  = st.session_state.get('font_size', 16)
_hc  = st.session_state.get('high_contrast', False)
_cb  = st.session_state.get('colorblind', False)
st.markdown(f"""<style>
    :root {{ --fs-base: {_fs}px; }}

    /* Font size — covers all Streamlit rendered elements */
    html, body, [class*="st-"], .stMarkdown, .stMarkdown p,
    .stMarkdown li, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    label, p, span, div, input, textarea, select, button,
    [data-testid="stText"], [data-testid="stMarkdown"],
    [data-testid="stWidgetLabel"], [data-testid="stRadio"] label,
    [data-testid="stCheckbox"] label, [data-testid="stExpander"] summary,
    [data-testid="stSlider"] label, [data-testid="stNumberInput"] label,
    [data-testid="metric-container"] label,
    [data-testid="stSidebar"] * {{ font-size: {_fs}px !important; }}

    /* High contrast */
    {'body { background:#000 !important; color:#fff !important; }' if _hc else ''}
    {'.metric-card { background:#111 !important; border-color:#fff !important; } .metric-card .val,.metric-card .lbl { color:#fff !important; }' if _hc else ''}
    {'[data-testid="stSidebar"] { background:#000 !important; }' if _hc else ''}
    {'[data-testid="stSidebar"] * { color:#fff !important; }' if _hc else ''}
</style>""", unsafe_allow_html=True)

# ─── Colorblind palette ────────────────────────────────────────────────────────
CB_PALETTE = ['#0072B2','#E69F00','#009E73','#F0E442','#CC79A7','#56B4E9','#D55E00']
STD_PALETTE = ['#2E4057','#048A81','#E76F51','#EAF0F6','#54C6EB']

# ─── Step navigation helper ───────────────────────────────────────────────────
STEPS = ["📂 Load Data", "🔍 Explore", "🧹 Clean Data", "🤖 Run Model", "📊 Results"]

def nav_buttons():
    """Render Previous / Next buttons at the bottom of each section."""
    step = st.session_state.current_step
    col_prev, col_label, col_next = st.columns([1, 2, 1])
    with col_prev:
        if step > 0:
            if st.button(f"← {STEPS[step-1].split(' ',1)[1]}", use_container_width=True):
                st.session_state.current_step = step - 1
                st.rerun()
    with col_label:
        st.markdown(f'<div class="step-label">Step {step+1} of {len(STEPS)}</div>',
                    unsafe_allow_html=True)
    with col_next:
        if step < len(STEPS) - 1:
            if st.button(f"{STEPS[step+1].split(' ',1)[1]} →", use_container_width=True):
                st.session_state.current_step = step + 1
                st.rerun()


# ─── Pipeline status bar ──────────────────────────────────────────────────────
def pipeline_bar():
    phases = [
        ('Load Data',  'status_obtain'),
        ('Explore',    'status_explore'),
        ('Clean Data', 'status_scrub'),
        ('Run Model',  'status_model'),
        ('Results',    'status_interpret'),
    ]
    html = '<div class="pipeline-bar">'
    for i, (label, key) in enumerate(phases):
        s = st.session_state[key]
        cls = 'phase-done' if s == 1 else ('phase-active' if i == current_phase_index() else 'phase-wait')
        icon = '✓' if s == 1 else ('●' if i == current_phase_index() else '○')
        html += f'<span class="phase-badge {cls}">{icon} {label}</span>'
        if i < len(phases) - 1:
            html += '<span class="arrow-sep">→</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def current_phase_index():
    statuses = ['status_obtain','status_explore','status_scrub','status_model','status_interpret']
    for i, k in enumerate(statuses):
        if st.session_state[k] == 0:
            return i
    return 4


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:

    # ── Help & Accessibility at the TOP — always visible, no scroll needed ────
    with st.expander("❓ Help & Guidance", expanded=False):
        st.markdown("""
**How to use this tool:**
1. **Load Data** — Upload your two CSV files
2. **Explore** — Review charts to understand your data
3. **Clean Data** — Filter and prepare the data based on what you found
4. **Run Model** — Generate personalized recommendations
5. **Results** — View performance metrics and download a report

**Navigating:** Use the **Previous / Next** buttons at the bottom of each screen, or click any step above to jump directly.

**Running & stopping:** The **▶ Run** and **■ Stop** buttons are in the **top-right corner** of the screen.

**Need help?** Contact your system administrator.
        """)

    with st.expander("♿ Accessibility", expanded=False):
        fs = st.slider("Text size", min_value=14, max_value=22, value=16, step=1,
                       help="Increase text size for easier reading.")
        st.session_state.font_size = fs
        hc = st.toggle("High contrast mode",
                       value=st.session_state.high_contrast,
                       help="Switches to a high-contrast display for users with visual impairments.")
        st.session_state.high_contrast = hc
        cb = st.toggle("Colorblind-friendly charts",
                       value=st.session_state.colorblind,
                       help="Replaces chart colors with a colorblind-accessible palette.")
        st.session_state.colorblind = cb
        st.caption("Use Tab to navigate, Enter/Space to activate. Charts are described in captions below each figure.")

    st.markdown("---")

    # ── Navigation ────────────────────────────────────────────────────────────
    st.markdown("### Navigation")
    chosen = st.radio(
        "Jump to step",
        STEPS,
        index=st.session_state.current_step,
        label_visibility="collapsed",
    )
    if chosen != STEPS[st.session_state.current_step]:
        st.session_state.current_step = STEPS.index(chosen)
        st.rerun()
    section = STEPS[st.session_state.current_step]

    st.markdown("---")

    # ── Pipeline Controls inside expander — keeps sidebar compact ─────────────
    with st.expander("⚙️ Pipeline Controls", expanded=False):
        alpha = st.slider(
            "Personalization strength",
            min_value=0.0, max_value=1.0, value=0.7, step=0.05,
            help="Controls how much the model relies on what similar users liked (higher) vs. product attributes (lower). Default of 0.7 works well for most cases."
        )
        n_factors = st.number_input(
            "Model complexity", min_value=10, max_value=200, value=50, step=10,
            help="Higher values may improve accuracy but increase processing time. The default of 50 is a reliable starting point."
        )
        adaptive_alpha = st.toggle(
            "Smart adjustment for new users",
            value=True,
            help="When enabled, the system automatically adjusts recommendations for users with limited history, relying more on product attributes rather than similar-user patterns."
        )
        topk = st.number_input("Number of recommendations", min_value=5, max_value=20, value=10,
            help="How many product recommendations to show per user.")

    st.markdown("---")
    if st.button("▶ Run Full Pipeline", type="primary", use_container_width=True):
        if st.session_state.df_raw is None:
            st.error("Load data first (📂 Load Data section).")
        else:
            st.session_state['run_full'] = True


# ─── Main area ────────────────────────────────────────────────────────────────
# Top-right run/stop hint
# Top-right hint + title in same row
st.markdown("""
<div style="display:flex; align-items:flex-start; justify-content:space-between; margin-bottom:4px;">
    <div style="text-align:center; flex:1; padding-top:2px;">
        <span style="font-size:1.8rem;" role="img" aria-label="Shopping bag">🛍️</span>
        <div style="font-size:1.4rem; font-weight:700; color:#2E4057; line-height:1.2; margin-top:2px;">
            Product Recommendation System
        </div>
        <div style="font-size:0.82rem; color:#4b5563; margin-top:2px;">
            Personalized product suggestions powered by your data
        </div>
    </div>
    <div style="flex-shrink:0; font-size:0.7rem; color:#4b5563; font-style:italic;
                text-align:right; padding-top:4px; max-width:160px; line-height:1.4;">
        💡 Use <strong>▶ Run</strong> / <strong>■ Stop</strong><br>in the top-right corner
    </div>
</div>
""", unsafe_allow_html=True)
pipeline_bar()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA
# ══════════════════════════════════════════════════════════════════════════════
if section == "📂 Load Data":
    st.markdown('<div class="section-header">📂 Load Data</div>', unsafe_allow_html=True)

    st.markdown("""
    **Welcome!** This tool helps you build and explore a product recommendation system.
    Follow the steps in order using the left sidebar navigation:
    **Load Data → Explore → Clean Data → Run Model → Results**

    Start by uploading your two data files below.
    """)

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        st.markdown("**Interaction file** (`interactions.csv`)")
        int_file = st.file_uploader("Upload interactions", type=["csv"],
                                    label_visibility="collapsed", key="int_upload")
    with col_up2:
        st.markdown("**Product metadata** (`products.csv`)")
        meta_file = st.file_uploader("Upload metadata", type=["csv"],
                                     label_visibility="collapsed", key="meta_upload")

    st.markdown('<p class="help-text">Accepted format: CSV · Max size: 50 MB · '
                'Files must share a productId key</p>', unsafe_allow_html=True)

    if int_file and meta_file:
        df_raw  = pd.read_csv(int_file,  dtype={'userId':str,'productId':str,'rating':int,'timestamp':int})
        df_meta = pd.read_csv(meta_file, dtype={'productId':str,'category':str,'price':float,'avgRating':float})
        df_meta = df_meta[df_meta['productId'].isin(df_raw['productId'].unique())].copy()

        # Integrity check
        missing = set(df_raw['productId'].unique()) - set(df_meta['productId'].unique())
        if missing:
            st.markdown(f'<div class="banner-err">⚠ Referential integrity failed: '
                        f'{len(missing)} product IDs in interactions have no metadata row. '
                        f'Check that both files share the same source.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="banner-ok">✓ Referential integrity confirmed — '
                        'both files are fully consistent.</div>', unsafe_allow_html=True)
            st.session_state.df_raw  = df_raw
            st.session_state.df_meta = df_meta
            sp = 1 - len(df_raw)/(df_raw['userId'].nunique()*df_raw['productId'].nunique())
            st.session_state.sparsity_raw = sp
            st.session_state.status_obtain = 1

        # Summary metric cards
        st.markdown('<div class="metric-row">'
            f'<div class="metric-card"><div class="val">{len(df_raw):,}</div>'
            f'<div class="lbl">Raw interactions</div></div>'
            f'<div class="metric-card"><div class="val">{df_raw["userId"].nunique():,}</div>'
            f'<div class="lbl">Unique users</div></div>'
            f'<div class="metric-card"><div class="val">{df_raw["productId"].nunique():,}</div>'
            f'<div class="lbl">Unique products</div></div>'
            f'<div class="metric-card"><div class="val">{df_meta["category"].nunique()}</div>'
            f'<div class="lbl">Categories</div></div>'
            f'<div class="metric-card"><div class="val">{sp:.2%}</div>'
            f'<div class="lbl">Data density</div></div>'
            '</div>', unsafe_allow_html=True)

        # Data preview with format toggle
        st.markdown("#### Data Preview")
        fmt = st.radio("View format", ["Table", "Summary Stats"],
                       horizontal=True, index=0)
        view_df = st.radio("Dataset", ["Interactions", "Products"], horizontal=True)
        disp = df_raw if view_df == "Interactions" else df_meta

        if fmt == "Table":
            st.dataframe(disp.head(10), use_container_width=True)
            # Column info
            info = pd.DataFrame({
                'Column':    disp.columns,
                'Dtype':     [str(t) for t in disp.dtypes],
                'Null count':[disp[c].isnull().sum() for c in disp.columns],
                'Sample':    [str(disp[c].iloc[0]) if len(disp)>0 else '' for c in disp.columns]
            })
            st.caption("Column information")
            st.dataframe(info, use_container_width=True, hide_index=True)

        elif fmt == "Summary Stats":
            st.dataframe(disp.describe().round(3), use_container_width=True)

    else:
        st.markdown('<div class="banner-info">ℹ Upload both CSV files above to begin.</div>',
                    unsafe_allow_html=True)
    nav_buttons()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: EXPLORE
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔍 Explore":
    st.markdown('<div class="section-header">🔍 Explore Data</div>', unsafe_allow_html=True)

    if st.session_state.df_raw is None:
        st.markdown('<div class="banner-err">⚠ Load data first (📂 Load Data section).</div>',
                    unsafe_allow_html=True)
    else:
        df   = st.session_state.df_clean if st.session_state.df_clean is not None else st.session_state.df_raw
        meta = st.session_state.df_meta_c if st.session_state.df_meta_c is not None else st.session_state.df_meta

        # Global filters
        with st.expander("🔧 Global filters", expanded=False):
            cats = sorted(meta['category'].unique().tolist())
            sel_cats = st.multiselect("Filter by category", cats, default=cats)
            if 'timestamp' in df.columns:
                df['ts'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                min_d = df['ts'].min().date()
                max_d = df['ts'].max().date()
                date_range = st.date_input("Date range", value=(min_d, max_d))
            min_rating = st.slider("Minimum rating", 1, 5, 1)

        # Apply filters
        df_f = df[df['rating'] >= min_rating].copy()
        if meta is not None and sel_cats:
            valid_prods = meta[meta['category'].isin(sel_cats)]['productId']
            df_f = df_f[df_f['productId'].isin(valid_prods)]

        # Chart selector
        chart_type = st.selectbox("Select chart type", [
            "Rating Distribution",
            "Item Popularity — Long Tail",
            "User Activity Distribution",
            "Category Breakdown",
            "Interaction Heatmap (Category × Rating)"
        ])

        STYLE = {'primary': CB_PALETTE[0] if _cb else '#2E4057',
                 'accent':  CB_PALETTE[2] if _cb else '#048A81',
                 'warm':    CB_PALETTE[6] if _cb else '#E76F51',
                 'light':   CB_PALETTE[3] if _cb else '#EAF0F6'}
        plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.top':False,
                             'axes.spines.right':False,'figure.dpi':130})

        if chart_type == "Rating Distribution":
            fig, ax = plt.subplots(figsize=(7, 4))
            rc = df_f['rating'].value_counts().sort_index()
            colors = [STYLE['light'],'#54C6EB',STYLE['accent'],STYLE['primary'],STYLE['warm']]
            bars = ax.bar(rc.index, rc.values, color=colors, edgecolor='white', linewidth=1.5, width=0.65)
            for bar, v in zip(bars, rc.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+rc.max()*0.015,
                        f'{v/len(df_f)*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
            pct = (df_f['rating']>=4).mean()
            ax.set_title(f'Rating Distribution  ({pct:.1%} ≥ 4★)', fontweight='bold', color=STYLE['primary'])
            ax.set_xlabel('Star Rating'); ax.set_ylabel('Count'); ax.set_xticks([1,2,3,4,5])
            st.pyplot(fig, use_container_width=True)
            st.caption(f"📌 What this means: {pct:.1%} of ratings are 4★ or 5★, indicating users tend to rate products they like. The recommendation model accounts for this to avoid over-weighting high ratings.")

        elif chart_type == "Item Popularity — Long Tail":
            with st.expander("Chart options"):
                pct_cutoff = st.slider("Top-N percentile marker", 1, 20, 5)
                log_scale  = st.toggle("Log scale Y-axis", value=True)
            ip = df_f['productId'].value_counts().reset_index()
            ip.columns = ['productId','count']
            ip = ip.sort_values('count', ascending=False).reset_index(drop=True)
            cut = int(len(ip)*pct_cutoff/100)
            share = ip.iloc[:cut]['count'].sum()/ip['count'].sum()
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(range(len(ip)), ip['count'], color=STYLE['primary'], linewidth=2)
            ax.fill_between(range(len(ip)), ip['count'], alpha=0.12, color=STYLE['accent'])
            if log_scale: ax.set_yscale('log')
            ax.axvline(cut, color=STYLE['warm'], linestyle='--', linewidth=2,
                       label=f'Top {pct_cutoff}% items = {share:.1%} of interactions')
            ax.set_title('Item Popularity — Long-Tail Distribution', fontweight='bold', color=STYLE['primary'])
            ax.set_xlabel('Item Rank'); ax.set_ylabel('Interactions'); ax.legend(fontsize=9)
            st.pyplot(fig, use_container_width=True)
            st.caption(f"📌 What this means: Just {pct_cutoff}% of products receive {share:.1%} of all interactions. The recommendation model is designed to surface products beyond these popular items — this is where personalization adds the most value.")

        elif chart_type == "User Activity Distribution":
            ua = df_f['userId'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(ua.values, bins=35, color=STYLE['accent'], edgecolor='white', linewidth=0.8, log=True)
            ax.axvline(ua.median(), color=STYLE['warm'], linestyle='--', linewidth=2,
                       label=f'Median = {ua.median():.0f} reviews/user')
            ax.set_title('User Activity Distribution', fontweight='bold', color=STYLE['primary'])
            ax.set_xlabel('Interactions per User'); ax.set_ylabel('User Count (log)')
            ax.legend(fontsize=9)
            st.pyplot(fig, use_container_width=True)
            cold = (ua < 10).mean()
            st.caption(f"Median interactions/user: {ua.median():.0f} | "
                       f"Users with < 10 reviews: {cold:.1%} (cold-start risk)")

        elif chart_type == "Category Breakdown":
            if meta is not None:
                cat_df = df_f.merge(meta[['productId','category']], on='productId', how='left')
                cr = (cat_df.groupby('category')['rating']
                      .agg(['mean','count']).reset_index()
                      .rename(columns={'mean':'Avg Rating','count':'Interactions'})
                      .sort_values('Interactions', ascending=True))
                fig, ax = plt.subplots(figsize=(9, 5))
                colors_h = [STYLE['primary'] if i%2==0 else STYLE['accent'] for i in range(len(cr))]
                bars = ax.barh(cr['category'], cr['Interactions'], color=colors_h, edgecolor='white')
                for bar, avg in zip(bars, cr['Avg Rating']):
                    ax.text(bar.get_width()+cr['Interactions'].max()*0.01,
                            bar.get_y()+bar.get_height()/2,
                            f'avg ★ {avg:.2f}', va='center', fontsize=9)
                ax.set_title('Interactions & Avg Rating by Category', fontweight='bold', color=STYLE['primary'])
                ax.set_xlabel('Number of Interactions')
                st.pyplot(fig, use_container_width=True)

        elif chart_type == "Interaction Heatmap (Category × Rating)":
            if meta is not None:
                cat_df = df_f.merge(meta[['productId','category']], on='productId', how='left')
                pivot = cat_df.groupby(['category','rating']).size().unstack(fill_value=0)
                fig, ax = plt.subplots(figsize=(9, 5))
                sns.heatmap(pivot, annot=True, fmt='d', cmap='Blues', ax=ax,
                            linewidths=0.5, cbar_kws={'label':'Interaction Count'})
                ax.set_title('Interaction Count: Category × Rating', fontweight='bold', color=STYLE['primary'])
                st.pyplot(fig, use_container_width=True)

        st.session_state.status_explore = 1
    nav_buttons()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: CLEAN
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🧹 Clean Data":
    st.markdown('<div class="section-header">🧹 Clean Data</div>', unsafe_allow_html=True)

    if st.session_state.df_raw is None:
        st.markdown('<div class="banner-err">⚠ Load data first (📂 Load Data section).</div>',
                    unsafe_allow_html=True)
    else:
        df_raw  = st.session_state.df_raw
        df_meta = st.session_state.df_meta

        col1, col2 = st.columns(2)
        with col1:
            min_user_int = st.number_input(
                "Min interactions per user", min_value=1, max_value=50, value=5,
                help="Users with fewer interactions than this are removed."
            )
        with col2:
            min_prod_rev = st.number_input(
                "Min reviews per product", min_value=1, max_value=100, value=10,
                help="Products with fewer reviews than this are removed."
            )

        # Live impact preview
        df_preview = df_raw.copy()
        for _ in range(3):
            uc = df_preview['userId'].value_counts()
            ic = df_preview['productId'].value_counts()
            df_preview = df_preview[
                df_preview['userId'].isin(uc[uc>=min_user_int].index) &
                df_preview['productId'].isin(ic[ic>=min_prod_rev].index)
            ]
        sp_after = 1 - len(df_preview)/(df_preview['userId'].nunique()*df_preview['productId'].nunique()) if len(df_preview)>0 else 1.0

        st.markdown("#### Live Impact Preview")
        st.markdown('<div class="metric-row">'
            f'<div class="metric-card"><div class="val">{len(df_preview):,}</div>'
            f'<div class="lbl">Remaining interactions</div></div>'
            f'<div class="metric-card"><div class="val">{df_preview["userId"].nunique():,}</div>'
            f'<div class="lbl">Active users</div></div>'
            f'<div class="metric-card"><div class="val">{df_preview["productId"].nunique():,}</div>'
            f'<div class="lbl">Active products</div></div>'
            f'<div class="metric-card"><div class="val">{len(df_raw)-len(df_preview):,}</div>'
            f'<div class="lbl">Rows removed</div></div>'
            f'<div class="metric-card"><div class="val">{sp_after:.2%}</div>'
            f'<div class="lbl">Post-filter sparsity</div></div>'
            '</div>', unsafe_allow_html=True)

        st.markdown("#### Preprocessing Steps")
        st.markdown('<p class="help-text">Uncheck a step to skip it and isolate its effect.</p>',
                    unsafe_allow_html=True)
        do_dedup  = st.checkbox("Remove duplicates", value=True)
        do_sparse = st.checkbox("Apply sparsity filter", value=True)
        do_weight = st.checkbox("Assign interaction weights", value=True)
        do_tfidf  = st.checkbox("Build TF-IDF content vectors", value=True)

        if st.button("✅ Apply Cleaning", type="primary"):
            with st.spinner("Cleaning data..."):
                df = df_raw.copy()

                if do_dedup:
                    df = df.drop_duplicates(subset=['userId','productId'], keep='last')

                if do_sparse:
                    for _ in range(3):
                        uc = df['userId'].value_counts()
                        ic = df['productId'].value_counts()
                        df = df[df['userId'].isin(uc[uc>=min_user_int].index) &
                                df['productId'].isin(ic[ic>=min_prod_rev].index)]

                if do_weight:
                    df['weight'] = df['rating'].map({1:0.2,2:0.4,3:0.6,4:0.8,5:1.0})

                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                df = df.sort_values('timestamp').reset_index(drop=True)

                df_meta_c = df_meta[df_meta['productId'].isin(df['productId'].unique())].copy()

                tfidf_matrix = None
                pids = None
                pid_idx = None

                if do_tfidf:
                    df_meta_c['price_bucket'] = pd.cut(
                        df_meta_c['price'], bins=5,
                        labels=['budget','low','mid','high','premium']
                    )
                    df_meta_c['content'] = (
                        df_meta_c['category'] + ' ' +
                        df_meta_c['price_bucket'].astype(str) + ' ' +
                        df_meta_c['avgRating'].astype(str)
                    )
                    tfidf        = TfidfVectorizer(max_features=500)
                    tfidf_matrix = tfidf.fit_transform(df_meta_c['content'])
                    pids         = df_meta_c['productId'].values
                    pid_idx      = {p:i for i,p in enumerate(pids)}

                sp = 1 - len(df)/(df['userId'].nunique()*df['productId'].nunique())
                st.session_state.df_clean       = df
                st.session_state.df_meta_c      = df_meta_c
                st.session_state.tfidf_matrix   = tfidf_matrix
                st.session_state.pids           = pids
                st.session_state.pid_idx        = pid_idx
                st.session_state.sparsity_clean = sp
                st.session_state.status_scrub   = 1

            st.markdown(
                f'<div class="banner-ok">✓ Cleaning complete. '
                f'{len(df):,} interactions retained across '
                f'{df["userId"].nunique():,} users and '
                f'{df["productId"].nunique():,} products. '
                f'Sparsity: {sp:.4%}. '
                f'{"TF-IDF matrix built: " + str(tfidf_matrix.shape) + "." if do_tfidf else ""}'
                f'</div>', unsafe_allow_html=True
            )

        if st.session_state.df_clean is not None:
            st.markdown("#### Cleaned Data Preview")
            st.dataframe(st.session_state.df_clean.head(8), use_container_width=True)
    nav_buttons()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🤖 Run Model":
    st.markdown('<div class="section-header">🤖 Run Model</div>', unsafe_allow_html=True)

    if st.session_state.df_clean is None:
        st.markdown('<div class="banner-err">⚠ Clean data first (🧹 Clean Data section).</div>',
                    unsafe_allow_html=True)
    else:
        df         = st.session_state.df_clean
        tfidf_mat  = st.session_state.tfidf_matrix
        pids       = st.session_state.pids
        pid_idx    = st.session_state.pid_idx
        df_meta_c  = st.session_state.df_meta_c

        # Model config
        st.markdown("#### Model Configuration")
        model_type = st.radio(
            "Recommendation approach",
            ["Hybrid (Recommended)", "Behavior-Based Only", "Product Similarity Only"],
            horizontal=True,
            help="Hybrid combines user behavior patterns with product similarity — recommended for most cases.")

        # Dynamic info box per selection
        _model_desc = {
            "Hybrid (Recommended)":
                "✅ <strong>Hybrid (Recommended):</strong> Combines behavior patterns from similar users with product attribute similarity. Best overall accuracy — recommended for most use cases.",
            "Behavior-Based Only":
                "👥 <strong>Behavior-Based Only:</strong> Recommends based purely on what similar users have liked. Works best when users have a rich interaction history.",
            "Product Similarity Only":
                "🏷️ <strong>Product Similarity Only:</strong> Recommends products similar in attributes (category, price, rating) to what the user has engaged with. Useful for new users with limited history.",
        }
        st.markdown(f'<div class="model-info-box" role="status" aria-live="polite">'
                    f'{_model_desc.get(model_type,"")}</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Personalization strength", f"{alpha:.2f}", help="How strongly the model weights similar-user behavior vs. product attributes.")
        with col_b:
            st.metric("Model complexity level", n_factors, help="Higher values may improve accuracy but take longer to train.")

        if st.button("▶ Run Model", type="primary"):
            with st.spinner("Training SVD — step 1 of 4..."):
                # Temporal split
                split_idx = int(len(df)*0.80)
                train_df  = df.iloc[:split_idx].copy()
                test_df   = df.iloc[split_idx:].copy()

                user_enc = LabelEncoder().fit(train_df['userId'])
                item_enc = LabelEncoder().fit(train_df['productId'])
                t = train_df.copy()
                t['u_idx'] = user_enc.transform(t['userId'])
                t['i_idx'] = item_enc.transform(t['productId'])

                R = csr_matrix(
                    (t['weight'].values, (t['u_idx'].values, t['i_idx'].values)),
                    shape=(t['u_idx'].max()+1, t['i_idx'].max()+1)
                )

            with st.spinner("Decomposing matrix — step 2 of 4..."):
                t0 = time.time()
                k  = min(n_factors, min(R.shape)-1)
                U, sigma, Vt = svds(R, k=k)
                U, sigma, Vt = U[:,::-1], sigma[::-1], Vt[::-1,:]
                R_pred = np.clip(U @ np.diag(sigma) @ Vt * 5, 1.0, 5.0)
                train_time = time.time()-t0

            with st.spinner("Evaluating model — step 3 of 4..."):
                def svd_scores_for_user(uid, prods):
                    if uid not in user_enc.classes_: return {p:2.5 for p in prods}
                    u = user_enc.transform([uid])[0]
                    known   = [p for p in prods if p in item_enc.classes_]
                    unknown = [p for p in prods if p not in item_enc.classes_]
                    idxs    = item_enc.transform(known)
                    sc      = dict(zip(known, R_pred[u, idxs].tolist()))
                    sc.update({p:2.5 for p in unknown})
                    return sc

                cat_map      = df_meta_c.set_index('productId')['category'].to_dict()
                all_products = df['productId'].unique()

                def hybrid_rec(uid, n=10):
                    rated  = set(train_df[train_df['userId']==uid]['productId'])
                    cands  = [p for p in all_products if p not in rated]
                    cf_sc  = svd_scores_for_user(uid, cands)
                    top5   = train_df[train_df['userId']==uid].nlargest(5,'rating')['productId'].tolist()
                    ua     = alpha if len(rated)>=10 and adaptive_alpha else 0.3
                    scores = []
                    for p in cands:
                        cf = cf_sc[p]
                        if top5 and pid_idx and p in pid_idx:
                            ri = [pid_idx[r] for r in top5 if r in pid_idx]
                            cb = float(cosine_similarity(tfidf_mat[pid_idx[p]], tfidf_mat[ri]).mean())*5 if ri else 2.5
                        else: cb = 2.5
                        scores.append((p, ua*cf+(1-ua)*cb))
                    scores.sort(key=lambda x:x[1], reverse=True)
                    final, cc = [], {}
                    for pid, sc in scores:
                        cat = cat_map.get(pid,'Unknown')
                        if cc.get(cat,0)<3: final.append(pid); cc[cat]=cc.get(cat,0)+1
                        if len(final)==n: break
                    return final

                def svd_rec(uid, n=10):
                    rated = set(train_df[train_df['userId']==uid]['productId'])
                    cands = [p for p in all_products if p not in rated]
                    sc    = svd_scores_for_user(uid, cands)
                    return sorted(cands, key=lambda p:sc[p], reverse=True)[:n]

                def prec(recs, rel, k=10): return len(set(recs[:k])&rel)/k
                def rec(recs, rel, k=10):  return len(set(recs[:k])&rel)/len(rel) if rel else 0
                def ndcg(recs, rel, k=10):
                    d=i=0.0
                    for j,it in enumerate(recs[:k]):
                        if it in rel: d+=1/np.log2(j+2)
                    for j in range(min(len(rel),k)): i+=1/np.log2(j+2)
                    return d/i if i else 0

                tui = test_df.groupby('userId')['productId'].apply(set).to_dict()
                eu  = [u for u,its in tui.items() if len(its)>=3 and u in train_df['userId'].values][:80]
                pop = train_df['productId'].value_counts().head(topk).index.tolist()

                mets = {m:{'p':[],'r':[],'n':[]} for m in ['hybrid','svd','pop']}
                t0 = time.time()
                for uid in eu:
                    rel   = tui[uid]
                    rated = set(train_df[train_df['userId']==uid]['productId'])
                    hr = hybrid_rec(uid, topk); sr = svd_rec(uid, topk)
                    pr = [x for x in pop if x not in rated][:topk]
                    for key,recs in [('hybrid',hr),('svd',sr),('pop',pr)]:
                        mets[key]['p'].append(prec(recs,rel,topk))
                        mets[key]['r'].append(rec(recs,rel,topk))
                        mets[key]['n'].append(ndcg(recs,rel,topk))
                lat = (time.time()-t0)/len(eu)*1000

            with st.spinner("Computing coverage — step 4 of 4..."):
                all_h = set()
                for u in eu[:50]: all_h.update(hybrid_rec(u,topk))
                cov_h = len(all_h)/df['productId'].nunique()
                cov_p = len(set(pop))/df['productId'].nunique()

            # Store results
            results = {m:{k:np.mean(v) for k,v in mv.items()} for m,mv in mets.items()}
            st.session_state.results       = results
            st.session_state.coverage      = (cov_h, cov_p)
            st.session_state.train_df      = train_df
            st.session_state.R_predicted   = R_pred
            st.session_state.user_enc      = user_enc
            st.session_state.item_enc      = item_enc
            st.session_state.status_model      = 1
            st.session_state.status_interpret  = 1
            st.session_state['lat_ms']         = lat
            st.session_state['train_time']      = train_time
            st.session_state['R_shape']         = R.shape
            st.session_state['sigma_top5']      = sigma[:5]
            st.session_state['var_explained']   = (sigma**2).sum()/(R.data**2).sum()
            st.session_state['hybrid_rec_fn']   = hybrid_rec
            st.session_state['pop_items']       = pop

            st.markdown(
                f'<div class="banner-ok">✓ Model trained in {train_time:.2f}s. '
                f'Latency: {lat:.0f}ms/user. '
                f'Coverage: {cov_h:.1%} (Hybrid) vs {cov_p:.1%} (Popularity).'
                f'</div>', unsafe_allow_html=True
            )

        # Show results if available
        if st.session_state.results is not None:
            results = st.session_state.results
            cov_h, cov_p = st.session_state.coverage

            st.markdown("#### Model Performance")
            viz = st.selectbox("Select visualization", [
                "Performance Comparison",
                "Catalog Coverage",
                "Sample Recommendations",
                "Model Summary"
            ])

            STYLE = {'primary': CB_PALETTE[0] if _cb else '#2E4057',
                 'accent':  CB_PALETTE[2] if _cb else '#048A81',
                 'warm':    CB_PALETTE[6] if _cb else '#E76F51',
                 'light':   CB_PALETTE[3] if _cb else '#EAF0F6'}

            if viz == "Performance Comparison":
                fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
                fig.suptitle(f'Recommendation Quality — Top {topk} Results', fontweight='bold', color=STYLE['primary'])
                labs = ['Most Popular', 'Behavior-Based', 'Hybrid\n(Recommended)']
                cols = [STYLE['light'], STYLE['accent'], STYLE['primary']]
                for ax, (key, label) in zip(axes, [('p','Recommendation Accuracy'),
                                                     ('r','Recommendation Coverage'),
                                                     ('n','Ranking Quality')]):
                    vals = [results['pop'][key], results['svd'][key], results['hybrid'][key]]
                    bars = ax.bar(labs, vals, color=cols, edgecolor=STYLE['primary'], linewidth=1.2, width=0.55)
                    ax.set_title(label, fontweight='bold', color=STYLE['primary'], fontsize=10)
                    ax.set_ylim(0, max(vals)*1.38)
                    for bar, v in zip(bars, vals):
                        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.025,
                                f'{v:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

            elif viz == "Catalog Coverage":
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(['Popularity', 'Hybrid'], [cov_p*100, cov_h*100],
                       color=[STYLE['light'], STYLE['primary']],
                       edgecolor=STYLE['primary'], linewidth=1.5, width=0.45)
                for i, v in enumerate([cov_p*100, cov_h*100]):
                    ax.text(i, v+0.4, f'{v:.1f}%', ha='center', va='bottom',
                            fontweight='bold', fontsize=12)
                ax.set_title('Catalog Coverage: Hybrid vs. Popularity Baseline',
                             fontweight='bold', color=STYLE['primary'])
                ax.set_ylabel('% of Catalog Recommended')
                ax.set_ylim(0, max(cov_p,cov_h)*100*1.35)
                st.pyplot(fig, use_container_width=True)
                st.caption(f"📌 What this means: The personalized model recommends {cov_h/cov_p:.1f}× more unique products than simply showing the most popular items — meaning users see genuinely tailored suggestions rather than the same products everyone else sees.")

            elif viz == "Sample Recommendations":
                uid_input = st.text_input("Enter a User ID (e.g. U00042)",
                                          placeholder="U00042")
                if uid_input and st.session_state.get('hybrid_rec_fn'):
                    fn = st.session_state['hybrid_rec_fn']
                    recs = fn(uid_input, n=topk)
                    if recs:
                        recs_df = (st.session_state.df_meta_c[
                            st.session_state.df_meta_c['productId'].isin(recs)]
                            [['productId','category','price','avgRating']]
                            .set_index('productId').reindex(recs))
                        col_r, col_h = st.columns(2)
                        with col_r:
                            st.markdown(f"**Top-{topk} recommendations for {uid_input}**")
                            st.dataframe(recs_df, use_container_width=True)
                        with col_h:
                            hist = st.session_state.train_df[
                                st.session_state.train_df['userId']==uid_input
                            ][['productId','rating']].tail(10)
                            st.markdown(f"**{uid_input}'s recent interactions**")
                            if len(hist):
                                st.dataframe(hist.reset_index(drop=True), use_container_width=True)
                            else:
                                st.info("User not found in training set.")
                    else:
                        st.warning("No recommendations generated. User may not be in training set.")

            elif viz == "Model Summary":
                st.markdown("**SVD Training Summary**")
                s5 = st.session_state.get('sigma_top5', [])
                st.markdown('<div class="metric-row">'
                    f'<div class="metric-card"><div class="val">'
                    f'{st.session_state.get("train_time",0):.2f}s</div>'
                    f'<div class="lbl">Training time</div></div>'
                    f'<div class="metric-card"><div class="val">'
                    f'{str(st.session_state.get("R_shape","—"))}</div>'
                    f'<div class="lbl">Matrix shape</div></div>'
                    f'<div class="metric-card"><div class="val">'
                    f'{st.session_state.get("var_explained",0):.1%}</div>'
                    f'<div class="lbl">Variance explained</div></div>'
                    f'<div class="metric-card"><div class="val">'
                    f'{st.session_state.get("lat_ms",0):.0f}ms</div>'
                    f'<div class="lbl">Response time/user</div></div>'
                    '</div>', unsafe_allow_html=True)
                if len(s5):
                    st.caption(f"Top-5 singular values: {', '.join(f'{v:.3f}' for v in s5)}")


    if st.session_state.results is None and st.session_state.df_clean is not None:
        nav_buttons()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📊 Results":
    st.markdown('<div class="section-header">📊 Results & Report</div>', unsafe_allow_html=True)

    st.markdown("#### Report Configuration")
    report_title = st.text_input("Report title",
                                  value="Product Recommendation System — Results Report")
    notes = st.text_area("Notes / Observations",
                          placeholder="Add any observations about the results...",
                          height=100)

    st.markdown("#### Sections to include")
    inc_dataset  = st.checkbox("Dataset Summary", value=True)
    inc_clean    = st.checkbox("Cleaning Report", value=True)
    inc_explore  = st.checkbox("Exploration Findings", value=True)
    inc_model    = st.checkbox("Model Performance Metrics", value=True)
    inc_recs     = st.checkbox("Sample Recommendations", value=True)

    if st.button("📄 Generate Report", type="primary"):
        if st.session_state.df_raw is None:
            st.markdown('<div class="banner-err">⚠ No data loaded. Complete the Load Data section first.</div>',
                        unsafe_allow_html=True)
        else:
            report_lines = [f"# {report_title}\n"]

            if inc_dataset and st.session_state.df_raw is not None:
                df = st.session_state.df_raw
                sp = st.session_state.sparsity_raw or 0
                report_lines += [
                    "## Dataset Summary",
                    f"- Raw interactions: {len(df):,}",
                    f"- Unique users: {df['userId'].nunique():,}",
                    f"- Unique products: {df['productId'].nunique():,}",
                    f"- Raw matrix sparsity: {sp:.4%}",
                    f"- Rating range: {df['rating'].min()}–{df['rating'].max()}",
                    ""
                ]

            if inc_clean and st.session_state.df_clean is not None:
                df_c = st.session_state.df_clean
                sp_c = st.session_state.sparsity_clean or 0
                report_lines += [
                    "## Cleaning Report",
                    f"- Interactions after cleaning: {len(df_c):,}",
                    f"- Active users: {df_c['userId'].nunique():,}",
                    f"- Active products: {df_c['productId'].nunique():,}",
                    f"- Post-filter sparsity: {sp_c:.4%}",
                    f"- Positivity bias: {(df_c['rating']>=4).mean():.1%} of ratings ≥ 4★",
                    ""
                ]

            if inc_explore:
                report_lines += [
                    "## Exploration Findings",
                    "- Rating distribution exhibits strong positivity bias (see Explore section charts).",
                    "- Item popularity follows a long-tail distribution.",
                    "- Significant cold-start risk identified for low-activity users.",
                    ""
                ]

            if inc_model and st.session_state.results is not None:
                r = st.session_state.results
                cov_h, cov_p = st.session_state.coverage
                lat = st.session_state.get('lat_ms', 0)
                report_lines += [
                    "## Model Performance Metrics",
                    f"| Metric | Hybrid (Recommended) | Behavior-Based | Most Popular |",
                    f"|--------|--------|----------|------------|",
                    f"| Precision@{topk} | {r['hybrid']['p']:.4f} | {r['svd']['p']:.4f} | {r['pop']['p']:.4f} |",
                    f"| Recall@{topk} | {r['hybrid']['r']:.4f} | {r['svd']['r']:.4f} | {r['pop']['r']:.4f} |",
                    f"| NDCG@{topk} | {r['hybrid']['n']:.4f} | {r['svd']['n']:.4f} | {r['pop']['n']:.4f} |",
                    "",
                    f"- Catalog coverage (Hybrid): {cov_h:.1%}",
                    f"- Catalog coverage (Popularity): {cov_p:.1%}",
                    f"- Coverage ratio: {cov_h/cov_p:.1f}× improvement",
                    f"- Avg latency per user: {lat:.0f}ms",
                    ""
                ]

            if notes.strip():
                report_lines += ["## Notes / Observations", notes, ""]

            report_text = "\n".join(report_lines)
            st.markdown("#### Report Preview")
            st.markdown(report_text)
            st.download_button(
                "⬇ Download Report (.md)",
                data=report_text,
                file_name="recommendation_system_report.md",
                mime="text/markdown"
            )

    # Pipeline status summary
    st.markdown("---")
    st.markdown("#### Pipeline Completion Status")
    phases = [
        ("Load Data",  'status_obtain'),
        ("Clean Data", 'status_scrub'),
        ("Explore",    'status_explore'),
        ("Run Model",  'status_model'),
        ("Results",    'status_interpret'),
    ]
    for label, key in phases:
        done = st.session_state[key] == 1
        st.markdown(f"{'✅' if done else '⬜'} **{label}** — {'Complete' if done else 'Not yet run'}")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Product Recommendation System · Built with Streamlit</div>',
    unsafe_allow_html=True
)
