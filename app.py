"""Mozi Wash Review Intelligence — Streamlit dashboard."""
import streamlit as st
import pandas as pd
import plotly.express as px
import json
import re
from anthropic import Anthropic
from datetime import datetime, timedelta

st.set_page_config(page_title="Mozi Wash Review Intelligence", layout="wide")

SCENT_NAMES = {
    'AW': 'Alpine Woods', 'CC': 'Central Coast', 'CZ': 'Signature Cozy',
    'DP': 'Desert Poppy', 'FC': 'Free & Clear', 'GH': 'Golden Hour',
    'HR': 'Hollywood Rouge', 'MM': 'Malibu Mornings', 'SD': 'Sugar Dew', 'VM': 'Vanilla Moon',
}

SCENT_COLORS = {
    'AW': '#4F7942', 'CC': '#4F86C6', 'CZ': '#C8A96E', 'DP': '#E8734A',
    'FC': '#88BBAA', 'GH': '#F2C94C', 'HR': '#9B2335', 'MM': '#6BAED6',
    'SD': '#F4A460', 'VM': '#9B59B6',
}
SCENT_COLOR_MAP = {SCENT_NAMES[k]: v for k, v in SCENT_COLORS.items()}


def safe_json_list(val):
    if not val or (isinstance(val, float) and pd.isna(val)):
        return []
    try:
        result = json.loads(val)
        return [str(x).strip() for x in result if x] if isinstance(result, list) else []
    except Exception:
        return []


def safe_json_dict(val):
    if not val or (isinstance(val, float) and pd.isna(val)):
        return {}
    try:
        result = json.loads(val)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}


@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv('reviews.csv')
    df['date_created'] = pd.to_datetime(df['date_created'], utc=True).dt.tz_convert(None)
    df['scent'] = df['sku'].map(SCENT_NAMES).fillna(df['sku'])
    for col in ('themes', 'standout_phrases', 'emotional_triggers', 'comparison_phrases', 'competitor_mentions', 'use_cases'):
        if col in df.columns:
            df[f'{col}_list'] = df[col].apply(safe_json_list)
        else:
            df[f'{col}_list'] = [[] for _ in range(len(df))]
    for col in ('switching_language', 'skeptic_converted'):
        if col in df.columns:
            df[f'{col}_parsed'] = df[col].apply(safe_json_dict)
        else:
            df[f'{col}_parsed'] = [{} for _ in range(len(df))]
    return df


@st.cache_data(ttl=3600)
def load_weekly_analysis():
    try:
        with open('weekly_analysis.json', encoding='utf-8') as f:
            d = json.load(f)
        return d.get('week_date'), d.get('new_review_count'), d.get('analysis', {})
    except Exception:
        return None, None, {}


@st.cache_data(ttl=3600)
def build_aggregate_context(df):
    from collections import Counter
    lines = []
    n = len(df)
    date_min = str(df['date_created'].min())[:10]
    date_max = str(df['date_created'].max())[:10]
    lines.append(f"ALL-TIME AGGREGATE ({n:,} reviews · {date_min} to {date_max})")
    lines.append("")

    # Per-scent stats
    lines.append("By scent:")
    scent_stats = (
        df.groupby('scent')['rating']
        .agg(count='count', avg=('mean'))
        .round(2)
        .sort_values('count', ascending=False)
    )
    for scent, row in scent_stats.iterrows():
        sdf = df[df['scent'] == scent]
        pct5 = int((sdf['rating'] == 5).mean() * 100)
        theme_ctr = Counter(t for tl in sdf['themes_list'] for t in tl)
        top_themes = ', '.join(f"{t}({c})" for t, c in theme_ctr.most_common(4)) or 'none tagged'
        lines.append(f"  {scent}: {int(row['count'])} reviews | avg {row['avg']:.1f}★ | {pct5}% 5-star | top tags: {top_themes}")

    # Theme frequency across all reviews
    theme_ctr = Counter(t for tl in df['themes_list'] for t in tl)
    if theme_ctr:
        lines.append("")
        lines.append(f"Tag frequency (all {n:,} reviews):")
        for tag, cnt in theme_ctr.most_common():
            lines.append(f"  {tag}: {cnt:,} ({cnt/n*100:.1f}%)")

    # Top standout phrases
    phrase_ctr = Counter(p for pl in df['standout_phrases_list'] for p in pl if p)
    if phrase_ctr:
        lines.append("")
        lines.append("Most-cited customer phrases:")
        for phrase, cnt in phrase_ctr.most_common(12):
            lines.append(f"  \"{phrase}\" — {cnt} reviews")

    # Competitor mentions
    comp_ctr = Counter(c for cl in df['competitor_mentions_list'] for c in cl if c)
    if comp_ctr:
        lines.append("")
        lines.append("Competitor mentions: " + ', '.join(f"{c}({n})" for c, n in comp_ctr.most_common()))

    # Use cases
    use_ctr = Counter(u for ul in df['use_cases_list'] for u in ul if u)
    if use_ctr:
        lines.append("")
        lines.append("Use cases: " + ', '.join(f"{u}({n})" for u, n in use_ctr.most_common(8)))

    return '\n'.join(lines)


try:
    df_all = load_data()
except FileNotFoundError:
    st.error("reviews.csv not found. Run okendo_dashboard.py first to generate the data file.")
    st.stop()

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Filters")

    if 'filter_preset' not in st.session_state:
        st.session_state.filter_preset = None
    pc1, pc2, pc3 = st.columns(3)
    if pc1.button("90 Days", use_container_width=True, key='pre_90d'):
        st.session_state.filter_preset = '90d'
        st.rerun()
    if pc2.button("5★ Only", use_container_width=True, key='pre_5star'):
        st.session_state.filter_preset = '5star'
        st.rerun()
    if pc3.button("Reset", use_container_width=True, key='pre_reset'):
        st.session_state.filter_preset = None
        st.rerun()

    _today    = df_all['date_created'].max().date()
    _min_date = df_all['date_created'].min().date()
    _preset   = st.session_state.filter_preset
    _def_dates  = [_today - timedelta(days=90), _today] if _preset == '90d' else [_min_date, _today]
    _def_rating = 5 if _preset == '5star' else 1

    date_range = st.date_input(
        "Date range",
        value=_def_dates,
        min_value=_min_date,
        max_value=_today,
        key=f'date_range_{_preset}',
    )
    all_skus = sorted(df_all['sku'].dropna().unique())
    sel_scents = st.multiselect(
        "Scents", options=all_skus,
        format_func=lambda x: SCENT_NAMES.get(x, x),
        default=all_skus,
    )
    min_rating = st.slider("Min rating", 1, 5, _def_rating, key=f'min_rating_{_preset}')

# ── Apply filters ──────────────────────────────────────────────────────────────
df = df_all.copy()
if len(date_range) == 2:
    df = df[
        (df['date_created'].dt.date >= date_range[0]) &
        (df['date_created'].dt.date <= date_range[1])
    ]
if sel_scents:
    df = df[df['sku'].isin(sel_scents)]
df = df[df['rating'].fillna(0) >= min_rating]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("Mozi Wash Review Intelligence")
st.caption(
    f"Last updated: {df_all['date_created'].max().strftime('%B %d, %Y')} "
    f"· {len(df_all):,} total reviews in database"
)

# ── Stats row ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
week_ago = datetime.now() - timedelta(days=7)
new_this_week = int((df_all['date_created'] >= week_ago).sum())
avg_rating_str = f"{df['rating'].mean():.2f} ★" if len(df) else "—"
pct_rec_str = f"{df['is_recommended'].mean() * 100:.0f}%" if len(df) else "—"

c1.metric("Reviews (filtered)", f"{len(df):,}")
c2.metric("Avg Rating", avg_rating_str)
c3.metric("New This Week", new_this_week)
c4.metric("% Recommended", pct_rec_str)

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_themes, tab_hooks, tab_generators, tab_competitors, tab_usecases, tab_trends, tab_attrs, tab_reviews, tab_chat = st.tabs([
    "Themes by Scent", "Marketing Hooks", "Generators", "Competitor Mentions", "Use Cases", "Trends", "Profile Attributes", "Reviews", "Ask Claude"
])

# ── Claude theme analysis (cached at module level) ─────────────────────────────
@st.cache_data(show_spinner="Analyzing themes with Claude — this takes about 30 seconds...")
def get_themes(review_count, max_date):
    client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    scent_blocks = []
    for sku in sorted(df_all['sku'].dropna().unique()):
        name = SCENT_NAMES.get(sku, sku)
        reviews = (
            df_all[(df_all['sku'] == sku) & df_all['body'].notna()]
            .sort_values('date_created', ascending=False)
            .head(40)
        )
        if reviews.empty:
            continue
        text = '\n'.join(
            f"[{r['rating']}★] {r['title']}: {str(r['body'])[:250]}"
            for _, r in reviews.iterrows()
        )
        scent_blocks.append(f"=== {sku} — {name} ===\n{text}")

    prompt = f"""You are a senior marketing strategist for Mozi Wash, a premium laundry detergent in beautiful metal tins.
Core customer: women who love premium home goods and care deeply about scent.

Analyze these customer reviews grouped by scent. For each scent return exactly 4 recurring themes that are useful for marketing.

Return ONLY a raw JSON object — no markdown, no code fences. Schema:
{{
  "AW": {{
    "themes": [
      {{"theme": "short theme name", "description": "one sentence", "example": "exact short quote from a review"}}
    ]
  }},
  ... (same structure for every scent present)
}}

Reviews:
{''.join(scent_blocks)}"""

    raw = client.messages.create(
        model='claude-sonnet-4-6',
        max_tokens=4096,
        messages=[{'role': 'user', 'content': prompt}],
    ).content[0].text.strip()
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[1].rsplit('```', 1)[0]
    return json.loads(raw)


# ── Themes by Scent tab ────────────────────────────────────────────────────────
with tab_themes:
    tagged_df = df[df['themes_list'].apply(len) > 0]

    if not tagged_df.empty:
        exploded = (
            tagged_df[['scent', 'themes_list']]
            .explode('themes_list')
            .rename(columns={'themes_list': 'theme'})
        )
        exploded = exploded[exploded['theme'].notna() & (exploded['theme'] != '')]

        top_themes = exploded['theme'].value_counts().head(15).index.tolist()
        hm_data = (
            exploded[exploded['theme'].isin(top_themes)]
            .groupby(['scent', 'theme'])
            .size()
            .reset_index(name='count')
        )
        scent_totals = df.groupby('scent').size().rename('total').reset_index()
        hm_data = hm_data.merge(scent_totals, on='scent')
        hm_data['pct'] = (hm_data['count'] / hm_data['total'] * 100).round(1)

        pivot = hm_data.pivot(index='theme', columns='scent', values='pct').fillna(0)
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

        fig_hm = px.imshow(
            pivot,
            labels=dict(color="% of Reviews"),
            color_continuous_scale='Blues',
            aspect='auto',
            text_auto='.0f',
        )
        fig_hm.update_layout(
            title="Theme Frequency by Scent (% of each scent's reviews tagged with that theme)",
            height=560,
            margin=dict(t=50, l=200, r=20, b=60),
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    else:
        st.info("No tagged reviews in current filter.")

    st.divider()
    all_themes_available = sorted(set(t for tl in df['themes_list'] for t in tl if t))

    if all_themes_available:
        st.subheader("Theme Detail")
        selected_theme = st.selectbox("Select a theme", all_themes_available, key='theme_select')

        theme_df = df[df['themes_list'].apply(lambda x: selected_theme in x)]

        now = df_all['date_created'].max()
        recent_cutoff = now - pd.Timedelta(days=90)
        prior_cutoff = now - pd.Timedelta(days=180)
        all_tagged_base = df_all[df_all['themes_list'].apply(len) > 0]
        recent_base = all_tagged_base[all_tagged_base['date_created'] >= recent_cutoff]
        prior_base = all_tagged_base[
            (all_tagged_base['date_created'] >= prior_cutoff) &
            (all_tagged_base['date_created'] < recent_cutoff)
        ]
        r_pct = recent_base['themes_list'].apply(lambda x: selected_theme in x).mean() if len(recent_base) else 0
        p_pct = prior_base['themes_list'].apply(lambda x: selected_theme in x).mean() if len(prior_base) else 0

        if p_pct == 0:
            trend_str = "— New theme"
        elif r_pct >= p_pct * 1.15:
            trend_str = "↑ Trending up"
        elif r_pct <= p_pct * 0.85:
            trend_str = "↓ Trending down"
        else:
            trend_str = "→ Stable"

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Reviews tagged", len(theme_df))
        mc2.metric("% of filtered reviews", f"{len(theme_df) / max(len(df), 1) * 100:.1f}%")
        mc3.metric("Trend (90d vs prior 90d)", trend_str)

        phrases = [p for tl in theme_df['standout_phrases_list'] for p in tl if p]
        if phrases:
            st.markdown("**Customer quotes:**")
            qc1, qc2 = st.columns(2)
            for i, phrase in enumerate(phrases[:10]):
                with (qc1 if i % 2 == 0 else qc2):
                    st.markdown(f"> *\"{phrase}\"*")

        scent_bk = (
            theme_df.groupby('scent').size()
            .reset_index(name='count')
            .sort_values('count', ascending=False)
        )
        fig_bk = px.bar(
            scent_bk, x='scent', y='count',
            title=f'"{selected_theme}" — mentions by scent',
            color='scent', color_discrete_map=SCENT_COLOR_MAP,
        )
        fig_bk.update_layout(height=280, margin=dict(t=40), showlegend=False)
        st.plotly_chart(fig_bk, use_container_width=True)

    st.divider()
    with st.expander("AI Deep Dive — Claude-generated theme analysis per scent", expanded=False):
        st.caption("Claude reads the latest 40 reviews per scent and surfaces 4 marketing themes. Cached weekly.")
        try:
            themes_data = get_themes(len(df_all), str(df_all['date_created'].max().date()))
        except Exception:
            st.warning("Claude is temporarily busy — refresh the page in a minute to load themes.")
            themes_data = None

        if themes_data:
            skus_present = [s for s in sorted(SCENT_NAMES.keys()) if s in themes_data]
            ai_cols = st.columns(2)
            for i, sku in enumerate(skus_present):
                name = SCENT_NAMES.get(sku, sku)
                scent_themes = themes_data.get(sku, {}).get('themes', [])
                with ai_cols[i % 2]:
                    st.markdown(f"#### {name}")
                    for t in scent_themes:
                        with st.expander(f"**{t['theme']}** — {t['description']}"):
                            st.markdown(f"*\"{t['example']}\"*")
                    st.divider()


# ── Marketing Hooks tab ────────────────────────────────────────────────────────
with tab_hooks:
    hdr_col, refresh_col = st.columns([6, 1])
    with hdr_col:
        st.caption("Customer language ready for ad copy — pulled directly from tagged reviews")
    with refresh_col:
        if st.button("Refresh", key='hooks_refresh'):
            load_data.clear()
            st.rerun()

    # ── Ad-Ready Phrases ──────────────────────────────────────────────────────
    st.subheader("Ad-Ready Phrases")
    st.caption("Vivid, specific customer quotes under 15 words — tagged as standout by Claude")

    all_phrases = [
        {'phrase': p, 'scent': row['scent'], 'rating': row['rating']}
        for _, row in df.iterrows()
        for p in row['standout_phrases_list'] if p
    ]

    if all_phrases:
        phrases_df = pd.DataFrame(all_phrases).drop_duplicates(subset='phrase')
        p_cols = st.columns(3)
        for i, (_, row) in enumerate(phrases_df.head(30).iterrows()):
            with p_cols[i % 3]:
                st.code(row['phrase'], language=None)
                st.caption(f"{row['scent']} · {int(row['rating'])}★")
    else:
        st.info("No standout phrases in current filter.")

    # ── Switching Stories ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("Switching Stories")
    st.caption("Customers who explicitly mention switching to Mozi from another brand")

    switch_rows = df[df['switching_language_parsed'].apply(
        lambda x: x.get('detected') is True and bool((x.get('quote') or '').strip())
    )]

    if not switch_rows.empty:
        st.caption(f"{len(switch_rows)} reviews with switching language")
        sw_cols = st.columns(2)
        for i, (_, row) in enumerate(switch_rows.head(20).iterrows()):
            quote = row['switching_language_parsed'].get('quote', '')
            with sw_cols[i % 2]:
                st.code(quote, language=None)
                st.caption(f"{row['scent']} · {int(row['rating'])}★")
    else:
        st.info("No switching language detected in current filter.")

    # ── Skeptic Converts ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Skeptic Converts")
    st.caption("Former skeptics who became believers — powerful social proof")

    skeptic_rows = df[df['skeptic_converted_parsed'].apply(
        lambda x: x.get('detected') is True and bool((x.get('quote') or '').strip())
    )]

    if not skeptic_rows.empty:
        st.caption(f"{len(skeptic_rows)} skeptic-converted reviews")
        sk_cols = st.columns(2)
        for i, (_, row) in enumerate(skeptic_rows.head(20).iterrows()):
            quote = row['skeptic_converted_parsed'].get('quote', '')
            with sk_cols[i % 2]:
                st.code(quote, language=None)
                st.caption(f"{row['scent']} · {int(row['rating'])}★")
    else:
        st.info("No skeptic-converted language detected in current filter.")

    # ── Comparison Phrases ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Comparison Phrases")
    st.caption("Customers comparing Mozi to other brands, scents, or experiences")

    all_comp = [
        {'phrase': p, 'scent': row['scent'], 'rating': row['rating']}
        for _, row in df.iterrows()
        for p in row['comparison_phrases_list'] if p
    ]

    if all_comp:
        comp_df = pd.DataFrame(all_comp).drop_duplicates(subset='phrase')
        st.caption(f"{len(comp_df)} comparison phrases")
        cp_cols = st.columns(2)
        for i, (_, row) in enumerate(comp_df.head(30).iterrows()):
            with cp_cols[i % 2]:
                st.code(row['phrase'], language=None)
                st.caption(f"{row['scent']} · {int(row['rating'])}★")
    else:
        st.info("No comparison phrases found in current filter.")

    # ── Emotional Triggers ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("Emotional Triggers")

    all_triggers = [t for tl in df['emotional_triggers_list'] for t in tl if t]
    if all_triggers:
        trigger_counts = pd.Series(all_triggers).value_counts().head(12).reset_index()
        trigger_counts.columns = ['trigger', 'count']
        fig_trig = px.bar(
            trigger_counts, x='count', y='trigger', orientation='h',
            title='Most Common Emotional Triggers across Reviews',
            color_discrete_sequence=['#9B59B6'],
        )
        fig_trig.update_layout(
            height=400, margin=dict(t=40, l=160),
            yaxis={'categoryorder': 'total ascending'},
        )
        st.plotly_chart(fig_trig, use_container_width=True)
    else:
        st.info("No emotional triggers found in current filter.")

    # ── Gift Signal Tracker ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Gift Signal Tracker")
    st.caption("Reviews where customers mention giving or receiving Mozi as a gift")

    gift_df = df[df['gifting_signal'].fillna(0).astype(int) == 1] if 'gifting_signal' in df.columns else pd.DataFrame()
    if not gift_df.empty:
        gsc1, gsc2 = st.columns(2)
        gsc1.metric("Reviews with gift signal", len(gift_df))
        gsc2.metric("% of filtered reviews", f"{len(gift_df) / max(len(df), 1) * 100:.1f}%")

        gift_scents = (
            gift_df.groupby('scent').size()
            .reset_index(name='count')
            .sort_values('count', ascending=False)
        )
        fig_gift = px.bar(
            gift_scents, x='scent', y='count', title='Gift Signal by Scent',
            color='scent', color_discrete_map=SCENT_COLOR_MAP,
        )
        fig_gift.update_layout(height=260, margin=dict(t=40), showlegend=False)
        st.plotly_chart(fig_gift, use_container_width=True)

        gift_phrases = [p for tl in gift_df['standout_phrases_list'] for p in tl if p]
        if gift_phrases:
            st.markdown("**Best quotes from gift reviews:**")
            gpc1, gpc2 = st.columns(2)
            for i, phrase in enumerate(gift_phrases[:8]):
                with (gpc1 if i % 2 == 0 else gpc2):
                    st.code(phrase, language=None)
    else:
        st.info("No gift signals in current filter.")


# ── Competitor Mentions tab ───────────────────────────────────────────────────
with tab_competitors:
    st.caption("Brands mentioned in reviews — all data is 4-5 star reviews so mentions are almost always in a switching or comparison context")

    comp_rows = df[df['competitor_mentions_list'].apply(len) > 0]

    if comp_rows.empty:
        st.info("No competitor mentions in current filter.")
    else:
        # Overall frequency chart
        all_comp_mentions = [c for cl in comp_rows['competitor_mentions_list'] for c in cl if c]
        comp_freq = pd.Series(all_comp_mentions).value_counts().reset_index()
        comp_freq.columns = ['competitor', 'mentions']

        fig_comp = px.bar(
            comp_freq, x='mentions', y='competitor', orientation='h',
            title='Competitor Mentions across Reviews',
            color_discrete_sequence=['#E8734A'],
        )
        fig_comp.update_layout(
            height=max(300, len(comp_freq) * 28),
            margin=dict(t=40, l=160),
            yaxis={'categoryorder': 'total ascending'},
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        # Competitor detail
        st.divider()
        st.subheader("Competitor Detail")
        competitor_options = comp_freq['competitor'].tolist()
        selected_comp = st.selectbox("Select a competitor", competitor_options, key='comp_select')

        comp_detail = comp_rows[comp_rows['competitor_mentions_list'].apply(lambda x: selected_comp in x)]

        cc1, cc2 = st.columns(2)
        cc1.metric("Reviews mentioning", len(comp_detail))
        cc2.metric("% of filtered reviews", f"{len(comp_detail) / max(len(df), 1) * 100:.1f}%")

        # Scent breakdown
        scent_comp = (
            comp_detail.groupby('scent').size()
            .reset_index(name='count')
            .sort_values('count', ascending=False)
        )
        fig_sc = px.bar(
            scent_comp, x='scent', y='count',
            title=f'{selected_comp} mentions by scent',
            color='scent', color_discrete_map=SCENT_COLOR_MAP,
        )
        fig_sc.update_layout(height=280, margin=dict(t=40), showlegend=False)
        st.plotly_chart(fig_sc, use_container_width=True)

        # Sample reviews
        st.markdown("**Reviews mentioning this competitor:**")
        for _, row in comp_detail.head(10).iterrows():
            with st.expander(f"{row['scent']} · {int(row['rating'])}★ · {str(row['date_created'])[:10]} — {row['title'] or ''}"):
                st.write(row['body'])


# ── Use Cases tab ──────────────────────────────────────────────────────────────
with tab_usecases:
    st.caption("What customers are washing — useful for audience targeting and ad copy")

    uc_rows = df[df['use_cases_list'].apply(len) > 0]

    if uc_rows.empty:
        st.info("No use cases tagged in current filter.")
    else:
        # Overall frequency chart
        all_use_cases = [u for ul in uc_rows['use_cases_list'] for u in ul if u]
        uc_freq = pd.Series(all_use_cases).value_counts().head(20).reset_index()
        uc_freq.columns = ['use_case', 'count']

        fig_uc = px.bar(
            uc_freq, x='count', y='use_case', orientation='h',
            title='Most Common Use Cases',
            color_discrete_sequence=['#4F7942'],
        )
        fig_uc.update_layout(
            height=max(300, len(uc_freq) * 28),
            margin=dict(t=40, l=180),
            yaxis={'categoryorder': 'total ascending'},
        )
        st.plotly_chart(fig_uc, use_container_width=True)

        # Gifting signal summary
        if 'gifting_signal' in df.columns:
            gift_count = int(df['gifting_signal'].fillna(0).sum())
            gift_pct = gift_count / max(len(df), 1) * 100
            st.info(f"Gift signal detected in **{gift_count} reviews** ({gift_pct:.1f}% of filtered) — customers mention giving or receiving Mozi as a gift")

        # Use case detail
        st.divider()
        st.subheader("Use Case Detail")
        uc_options = uc_freq['use_case'].tolist()
        selected_uc = st.selectbox("Select a use case", uc_options, key='uc_select')

        uc_detail = uc_rows[uc_rows['use_cases_list'].apply(lambda x: selected_uc in x)]

        uc1, uc2 = st.columns(2)
        uc1.metric("Reviews mentioning", len(uc_detail))
        uc2.metric("% of filtered reviews", f"{len(uc_detail) / max(len(df), 1) * 100:.1f}%")

        # Scent breakdown for this use case
        scent_uc = (
            uc_detail.groupby('scent').size()
            .reset_index(name='count')
            .sort_values('count', ascending=False)
        )
        fig_su = px.bar(
            scent_uc, x='scent', y='count',
            title=f'"{selected_uc}" — which scents customers buy for this',
            color='scent', color_discrete_map=SCENT_COLOR_MAP,
        )
        fig_su.update_layout(height=280, margin=dict(t=40), showlegend=False)
        st.plotly_chart(fig_su, use_container_width=True)

        # Standout phrases for this use case
        uc_phrases = [p for tl in uc_detail['standout_phrases_list'] for p in tl if p]
        if uc_phrases:
            st.markdown("**Customer quotes:**")
            uqc1, uqc2 = st.columns(2)
            for i, phrase in enumerate(uc_phrases[:8]):
                with (uqc1 if i % 2 == 0 else uqc2):
                    st.markdown(f"> *\"{phrase}\"*")


# ── Trends Over Time tab ──────────────────────────────────────────────────────
with tab_trends:
    st.caption("Full history shown here — date range filter ignored, scent filter applies")

    trends_df = df_all.copy()
    if sel_scents:
        trends_df = trends_df[trends_df['sku'].isin(sel_scents)]

    gran = st.radio("Granularity", ["Monthly", "Weekly"], horizontal=True, key='trends_gran')
    trends_df['period'] = trends_df['date_created'].dt.to_period(
        'M' if gran == "Monthly" else 'W'
    ).dt.to_timestamp()

    # ── Event annotations ──
    if 'events' not in st.session_state:
        st.session_state.events = []

    with st.expander("Event annotations", expanded=False):
        ec1, ec2, ec3 = st.columns([2, 4, 1])
        with ec1:
            evt_date = st.date_input("Date", key='evt_date')
        with ec2:
            evt_label = st.text_input("Label (e.g. 'Launched Alpine Woods')", key='evt_label')
        with ec3:
            st.write("")
            st.write("")
            if st.button("Add", key='evt_add') and evt_label:
                st.session_state.events.append({'date': str(evt_date), 'label': evt_label})
                st.rerun()
        if st.session_state.events:
            for e in st.session_state.events:
                st.caption(f"· {e['date']}: {e['label']}")
            if st.button("Clear all", key='evt_clear'):
                st.session_state.events = []
                st.rerun()

    def add_events_to_fig(fig):
        for evt in st.session_state.get('events', []):
            ts = pd.Timestamp(evt['date'])
            fig.add_vline(x=ts, line_dash='dash', line_color='rgba(220,80,80,0.5)', line_width=1.5)
            fig.add_annotation(
                x=ts, y=1.02, yref='paper',
                text=evt['label'], showarrow=False,
                textangle=-30, font=dict(size=9, color='rgba(220,80,80,0.9)'),
                xanchor='left',
            )
        return fig

    # ── Review Volume ──
    vol = trends_df.groupby('period').size().reset_index(name='reviews')
    fig_vol = px.bar(vol, x='period', y='reviews', title='Review Volume Over Time',
                     color_discrete_sequence=['#4F86C6'])
    fig_vol.update_layout(height=300, margin=dict(t=40))
    fig_vol = add_events_to_fig(fig_vol)
    st.plotly_chart(fig_vol, use_container_width=True)

    # ── Avg Rating Trend ──
    rating_trend = (
        trends_df.groupby('period')['rating']
        .mean()
        .reset_index()
        .rename(columns={'rating': 'avg_rating'})
        .round(2)
    )
    fig_rating = px.line(rating_trend, x='period', y='avg_rating',
                         title='Average Rating Over Time', markers=True,
                         color_discrete_sequence=['#F2C94C'])
    fig_rating.update_layout(height=280, margin=dict(t=40), yaxis=dict(range=[3.5, 5.1]))
    fig_rating = add_events_to_fig(fig_rating)
    st.plotly_chart(fig_rating, use_container_width=True)

    # ── Theme Trends ──
    tagged_trends = trends_df[trends_df['themes_list'].apply(len) > 0]
    if not tagged_trends.empty:
        all_t = [t for tl in tagged_trends['themes_list'] for t in tl if t]
        top8 = pd.Series(all_t).value_counts().head(8).index.tolist()

        monthly_totals = tagged_trends.groupby('period').size().rename('total')
        trend_rows = []
        for theme in top8:
            with_theme = (
                tagged_trends[tagged_trends['themes_list'].apply(lambda x: theme in x)]
                .groupby('period').size().rename('with_theme')
            )
            merged = monthly_totals.to_frame().join(with_theme, how='left').fillna(0)
            merged['pct'] = (merged['with_theme'] / merged['total'] * 100).round(1)
            merged['theme'] = theme
            trend_rows.append(merged.reset_index())

        theme_trend_df = pd.concat(trend_rows)
        fig_tt = px.line(
            theme_trend_df, x='period', y='pct', color='theme',
            title='Theme Mention Rate Over Time (% of tagged reviews)',
        )
        fig_tt.update_layout(
            height=400, margin=dict(t=40),
            yaxis_title='% of Reviews',
            legend=dict(orientation='h', yanchor='bottom', y=-0.45),
        )
        fig_tt = add_events_to_fig(fig_tt)
        st.plotly_chart(fig_tt, use_container_width=True)
    else:
        st.info("No tagged reviews available for theme trends.")


# ── Generators tab ────────────────────────────────────────────────────────────
with tab_generators:
    st.caption("AI copy tools trained on your real customer language — every output draws from actual review tags")

    scent_options_gen = ['All scents'] + [SCENT_NAMES[s] for s in sorted(SCENT_NAMES.keys())]

    def get_scent_df(scent_label):
        if scent_label == 'All scents':
            return df
        sku = next((k for k, v in SCENT_NAMES.items() if v == scent_label), None)
        return df[df['sku'] == sku] if sku else df

    def get_gen_context(sdf):
        phrases = list(set(p for tl in sdf['standout_phrases_list'] for p in tl if p))[:25]
        triggers = [t for tl in sdf['emotional_triggers_list'] for t in tl if t]
        top_triggers = pd.Series(triggers).value_counts().head(8).to_dict() if triggers else {}
        switch_quotes = [
            r['switching_language_parsed'].get('quote', '')
            for _, r in sdf.iterrows()
            if r['switching_language_parsed'].get('detected') and r['switching_language_parsed'].get('quote')
        ][:8]
        skeptic_quotes = [
            r['skeptic_converted_parsed'].get('quote', '')
            for _, r in sdf.iterrows()
            if r['skeptic_converted_parsed'].get('detected') and r['skeptic_converted_parsed'].get('quote')
        ][:8]
        comp_phrases = list(set(p for tl in sdf['comparison_phrases_list'] for p in tl if p))[:10]
        top_themes = pd.Series(
            [t for tl in sdf['themes_list'] for t in tl if t]
        ).value_counts().head(5).to_dict()
        return dict(phrases=phrases, triggers=top_triggers, switch_quotes=switch_quotes,
                    skeptic_quotes=skeptic_quotes, comp_phrases=comp_phrases, themes=top_themes)

    def show_lines(result_key):
        if st.session_state.get(result_key):
            st.divider()
            for line in [l.strip() for l in st.session_state[result_key].strip().split('\n') if l.strip()]:
                st.code(line, language=None)

    gen_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

    g1, g2, g3, g4, g5, g6 = st.tabs([
        "Hook Generator", "Testimonial Picker", "Subject Lines",
        "Landing Page", "TikTok Script", "Scent Personality",
    ])

    # ── Hook Generator ──
    with g1:
        st.subheader("Hook Generator")
        st.caption("5 ad hooks written in your customers' own voice")
        hg_scent = st.selectbox("Scent", scent_options_gen, key='hg_scent')
        hg_style = st.selectbox("Hook style", [
            "Sensory / scent-first", "Social proof / compliments",
            "Switching story", "Skeptic converted", "Emotional / lifestyle",
        ], key='hg_style')
        if st.button("Generate hooks", key='hg_btn'):
            ctx = get_gen_context(get_scent_df(hg_scent))
            scent_str = '' if hg_scent == 'All scents' else f' — {hg_scent}'
            prompt = f"""Write 5 short ad hooks for Mozi Wash{scent_str}. Style: {hg_style}

Real customer language (use their words, not generic marketing):
Standout quotes: {ctx['phrases'][:20]}
Top emotional triggers: {ctx['triggers']}
Switching stories: {ctx['switch_quotes']}
Comparison phrases: {ctx['comp_phrases']}

Rules: under 15 words each · vivid and specific · sound like a real person · each hook takes a different angle · no hashtags, emojis, or "Introducing"

Return exactly 5 hooks, one per line, no numbering or bullets."""
            with st.spinner("Generating..."):
                try:
                    st.session_state['hg_result'] = gen_client.messages.create(
                        model='claude-sonnet-4-6', max_tokens=400,
                        messages=[{'role': 'user', 'content': prompt}],
                    ).content[0].text
                except Exception:
                    st.error("Generation failed — try again.")
        show_lines('hg_result')

    # ── Testimonial Picker ──
    with g2:
        st.subheader("Testimonial Picker")
        st.caption("Finds the most persuasive reviews for a specific audience or use case")
        tp_scent = st.selectbox("Scent", scent_options_gen, key='tp_scent')
        tp_audience = st.text_input("Target audience or use case",
            placeholder="e.g. new moms, fitness enthusiasts, gift buyers, sensitive skin", key='tp_audience')
        if st.button("Find testimonials", key='tp_btn') and tp_audience:
            sdf = get_scent_df(tp_scent)
            sample = sdf[sdf['body'].notna()].sort_values('rating', ascending=False).head(80)
            review_text = '\n'.join(
                f"[{i}] {r['scent']} {int(r['rating'])}★: {str(r['body'])[:300]}"
                for i, (_, r) in enumerate(sample.iterrows(), 1)
            )
            prompt = f"""Pick the 5 best testimonials for this target audience: "{tp_audience}"

Select reviews that will resonate most — prioritize specificity, relatability, vivid language.

Reviews:
{review_text}

For each: paste the exact review text (unedited), then one sentence on why it works for this audience. Number them 1–5."""
            with st.spinner("Finding best testimonials..."):
                try:
                    st.session_state['tp_result'] = gen_client.messages.create(
                        model='claude-sonnet-4-6', max_tokens=1024,
                        messages=[{'role': 'user', 'content': prompt}],
                    ).content[0].text
                except Exception:
                    st.error("Generation failed — try again.")
        if st.session_state.get('tp_result'):
            st.divider()
            st.markdown(st.session_state['tp_result'])

    # ── Subject Lines ──
    with g3:
        st.subheader("Subject Line Generator")
        st.caption("Email subject lines built from customer language")
        sl_scent = st.selectbox("Scent focus", scent_options_gen, key='sl_scent')
        sl_angle = st.selectbox("Campaign angle", [
            "New scent / product launch", "Gifting season",
            "Re-engagement / win-back", "Seasonal refresh", "Social proof / bestseller",
        ], key='sl_angle')
        if st.button("Generate subject lines", key='sl_btn'):
            ctx = get_gen_context(get_scent_df(sl_scent))
            scent_str = '' if sl_scent == 'All scents' else f' — {sl_scent}'
            prompt = f"""Write 5 email subject lines for Mozi Wash{scent_str}. Campaign: {sl_angle}

Customer language:
Standout phrases: {ctx['phrases'][:15]}
Top emotions: {ctx['triggers']}
Top themes: {ctx['themes']}

Rules: under 50 chars ideally · specific and intriguing · mix curiosity, social proof, direct benefit · no spam words (free, %, !!!)

Return 5 subject lines, one per line, no numbering."""
            with st.spinner("Generating..."):
                try:
                    st.session_state['sl_result'] = gen_client.messages.create(
                        model='claude-sonnet-4-6', max_tokens=300,
                        messages=[{'role': 'user', 'content': prompt}],
                    ).content[0].text
                except Exception:
                    st.error("Generation failed — try again.")
        show_lines('sl_result')

    # ── Landing Page Block ──
    with g4:
        st.subheader("Landing Page Block Builder")
        st.caption("Copy blocks for product pages — grounded in real customer language")
        lp_scent = st.selectbox("Scent", list(SCENT_NAMES.values()), key='lp_scent')
        lp_block = st.selectbox("Block type", [
            "Hero headline + subhead",
            "Social proof section (3 quotes + intro)",
            "Product description paragraph",
            "FAQ — common objections answered",
        ], key='lp_block')
        if st.button("Build block", key='lp_btn'):
            sku = next((k for k, v in SCENT_NAMES.items() if v == lp_scent), None)
            sdf = df[df['sku'] == sku] if sku else df
            ctx = get_gen_context(sdf)
            prompt = f"""Write a {lp_block} for the Mozi Wash {lp_scent} product page.

About Mozi Wash: Premium laundry detergent in a beautiful metal tin. Clean ingredients, incredible scents, for people who care about their home environment.

Real customer language for {lp_scent}:
Standout quotes: {ctx['phrases'][:15]}
Top emotional triggers: {ctx['triggers']}
Top themes: {ctx['themes']}
Switching quotes: {ctx['switch_quotes'][:3]}

Write copy that feels premium, specific to this scent, and grounded in real customer language. No fluff."""
            with st.spinner("Building..."):
                try:
                    st.session_state['lp_result'] = gen_client.messages.create(
                        model='claude-sonnet-4-6', max_tokens=600,
                        messages=[{'role': 'user', 'content': prompt}],
                    ).content[0].text
                except Exception:
                    st.error("Generation failed — try again.")
        if st.session_state.get('lp_result'):
            st.divider()
            st.markdown(st.session_state['lp_result'])

    # ── TikTok Script Seed ──
    with g5:
        st.subheader("TikTok Script Seed")
        st.caption("Short-form video outlines based on real customer stories")
        tt_scent = st.selectbox("Scent", scent_options_gen, key='tt_scent')
        tt_angle = st.selectbox("Video angle", [
            "Skeptic story (I was doubtful until...)",
            "Scent reveal / reaction",
            "Switching story (I used to use X...)",
            "Day in the life / routine",
            "Gift reveal",
        ], key='tt_angle')
        if st.button("Generate script seed", key='tt_btn'):
            ctx = get_gen_context(get_scent_df(tt_scent))
            scent_str = '' if tt_scent == 'All scents' else f' — {tt_scent}'
            prompt = f"""Write a TikTok script seed for Mozi Wash{scent_str}. Angle: {tt_angle}

Real customer stories:
Standout phrases: {ctx['phrases'][:15]}
Skeptic quotes: {ctx['skeptic_quotes']}
Switching quotes: {ctx['switch_quotes']}
Emotional triggers: {ctx['triggers']}

Format:
HOOK (0-3s): [opening line that stops the scroll]
SETUP (3-10s): [context / who this is for]
PAYOFF (10-25s): [the reveal / emotional moment]
CTA (25-30s): [what to do next]

Conversational, specific, grounded in real customer language. Should feel like someone telling a friend, not an ad."""
            with st.spinner("Generating..."):
                try:
                    st.session_state['tt_result'] = gen_client.messages.create(
                        model='claude-sonnet-4-6', max_tokens=500,
                        messages=[{'role': 'user', 'content': prompt}],
                    ).content[0].text
                except Exception:
                    st.error("Generation failed — try again.")
        if st.session_state.get('tt_result'):
            st.divider()
            st.markdown(st.session_state['tt_result'])

    # ── Scent Personality ──
    with g6:
        st.subheader("Scent Personality Generator")
        st.caption("Who actually buys this scent and how to talk to them")
        sp_scent = st.selectbox("Scent", list(SCENT_NAMES.values()), key='sp_scent')
        if st.button("Generate personality profile", key='sp_btn'):
            sku = next((k for k, v in SCENT_NAMES.items() if v == sp_scent), None)
            sdf = df[df['sku'] == sku] if sku else df
            ctx = get_gen_context(sdf)
            age_data = sdf['reviewer_age'].dropna().value_counts().head(3).to_dict() if 'reviewer_age' in sdf.columns else {}
            prompt = f"""Create a scent personality profile for Mozi Wash {sp_scent} based on real customer data.

Customer data:
Standout phrases: {ctx['phrases'][:20]}
Top emotional triggers: {ctx['triggers']}
Top themes: {ctx['themes']}
Age breakdown: {age_data}
Review count: {len(sdf)}

Write a profile covering:
1. **Who she is** — 2-3 sentence archetype
2. **What she values** — 3-4 bullets
3. **How she talks about it** — the specific language and phrases she uses
4. **Best ad angles** — 3 specific angles that resonate with her
5. **Scent in one sentence** — a single vivid positioning line

Base everything on the real data. Be specific, not generic."""
            with st.spinner("Building profile..."):
                try:
                    st.session_state['sp_result'] = gen_client.messages.create(
                        model='claude-sonnet-4-6', max_tokens=800,
                        messages=[{'role': 'user', 'content': prompt}],
                    ).content[0].text
                except Exception:
                    st.error("Generation failed — try again.")
        if st.session_state.get('sp_result'):
            st.divider()
            st.markdown(st.session_state['sp_result'])


# ── Profile Attributes ─────────────────────────────────────────────────────────
with tab_attrs:
    st.caption("Reviewer demographics from the current filtered view")

    if df.empty:
        st.info("No data for current filters.")
    else:
        # ── Fragrance Comparison ──────────────────────────────────────────────
        st.subheader("Fragrance Comparison")
        fcc1, fcc2 = st.columns(2)
        scent_name_options = [SCENT_NAMES[s] for s in sorted(SCENT_NAMES.keys())]
        with fcc1:
            cmp_a = st.selectbox("Scent A", scent_name_options, index=0, key='cmp_a')
        with fcc2:
            cmp_b = st.selectbox("Scent B", scent_name_options, index=1, key='cmp_b')

        def scent_stats(name):
            sku = next((k for k, v in SCENT_NAMES.items() if v == name), None)
            s = df[df['sku'] == sku] if sku else pd.DataFrame()
            themes = pd.Series([t for tl in s['themes_list'] for t in tl if t]).value_counts().head(3).index.tolist() if len(s) else []
            phrase = next((p for tl in s.sort_values('rating', ascending=False)['standout_phrases_list'] for p in tl if p), None) if len(s) else None
            return {'n': len(s), 'avg': s['rating'].mean() if len(s) else 0, 'themes': themes, 'phrase': phrase}

        sa, sb = scent_stats(cmp_a), scent_stats(cmp_b)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"{cmp_a} reviews", sa['n'])
        m2.metric(f"{cmp_a} avg rating", f"{sa['avg']:.2f}★" if sa['n'] else "—")
        m3.metric(f"{cmp_b} reviews", sb['n'])
        m4.metric(f"{cmp_b} avg rating", f"{sb['avg']:.2f}★" if sb['n'] else "—")

        tc1, tc2 = st.columns(2)
        with tc1:
            st.markdown(f"**{cmp_a} — Top Themes**")
            for t in sa['themes']:
                st.markdown(f"- {t}")
            if sa['phrase']:
                st.markdown(f"> *\"{sa['phrase']}\"*")
        with tc2:
            st.markdown(f"**{cmp_b} — Top Themes**")
            for t in sb['themes']:
                st.markdown(f"- {t}")
            if sb['phrase']:
                st.markdown(f"> *\"{sb['phrase']}\"*")

        st.divider()
        col1, col2 = st.columns(2)

        AGE_ORDER = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        WASH_ORDER = ['1', '2-4', '4-6', '7+']
        HOUSEHOLD_ORDER = ['1', '2 - 4', '4 - 6', '7 +']
        HOUSEHOLD_LABELS = {'1': '1', '2 - 4': '2-4', '4 - 6': '4-6', '7 +': '7+'}

        def clean_washes(val):
            if not val or str(val).strip() in ('', 'nan'):
                return None
            val = str(val).strip().strip("[]'\"").split(',')[0].strip().strip("'\"")
            return {'1': '1', '2': '2-4', '3-4': '2-4', '5-6': '4-6', '7+': '7+'}.get(val)

        with col1:
            if 'reviewer_name' in df.columns:
                repeat = (
                    df[df['reviewer_name'].notna() & (df['reviewer_name'] != '')]
                    .groupby('reviewer_name')
                    .filter(lambda x: len(x) > 1)
                )
                repeat_by_scent = (
                    repeat.groupby('scent').size()
                    .reset_index(name='reviews_from_repeat_customers')
                    .sort_values('reviews_from_repeat_customers', ascending=False)
                )
                fig_repeat = px.bar(
                    repeat_by_scent, x='scent', y='reviews_from_repeat_customers',
                    title='Reviews from Repeat Customers by Scent',
                    color_discrete_sequence=['#C8A96E'],
                )
                fig_repeat.update_layout(height=320, margin=dict(t=40), xaxis_tickangle=-35)
                st.plotly_chart(fig_repeat, use_container_width=True)
            else:
                st.info("Reviewer name data not yet available — run okendo_dashboard.py to update.")

            wash_counts = (
                df['reviewer_washes']
                .apply(clean_washes)
                .dropna()
                .value_counts()
                .reindex(WASH_ORDER)
                .dropna()
                .reset_index()
            )
            wash_counts.columns = ['washes', 'count']
            fig_wash = px.bar(
                wash_counts, x='washes', y='count',
                title='Washes per Week',
                color_discrete_sequence=['#4F86C6'],
                category_orders={'washes': WASH_ORDER},
            )
            fig_wash.update_layout(height=320, margin=dict(t=40))
            st.plotly_chart(fig_wash, use_container_width=True)

        with col2:
            hh_counts = (
                df['reviewer_household'].dropna()
                .astype(str).str.strip()
                .value_counts()
                .reindex(HOUSEHOLD_ORDER)
                .dropna()
                .reset_index()
            )
            hh_counts.columns = ['household', 'count']
            hh_counts['household'] = hh_counts['household'].map(HOUSEHOLD_LABELS)
            fig_hh = px.bar(
                hh_counts, x='household', y='count',
                title='Household Size',
                color_discrete_sequence=['#5BAD6F'],
                category_orders={'household': ['1', '2-4', '4-6', '7+']},
            )
            fig_hh.update_layout(height=320, margin=dict(t=40))
            st.plotly_chart(fig_hh, use_container_width=True)

            age_scent = (
                df[df['reviewer_age'].notna() & df['scent'].notna()]
                .groupby(['scent', 'reviewer_age'])
                .size()
                .reset_index(name='count')
            )
            age_scent['reviewer_age'] = pd.Categorical(age_scent['reviewer_age'], categories=AGE_ORDER, ordered=True)
            age_scent = age_scent.sort_values('reviewer_age')
            fig_as = px.bar(
                age_scent, x='scent', y='count', color='reviewer_age',
                title='Age Range by Scent',
                category_orders={'reviewer_age': AGE_ORDER},
            )
            fig_as.update_layout(height=320, margin=dict(t=40), xaxis_tickangle=-35)
            st.plotly_chart(fig_as, use_container_width=True)

        # ── Repeat Purchase Reasons ───────────────────────────────────────────
        st.divider()
        st.subheader("Repeat Purchase Reasons")
        st.caption("What customers who have reviewed multiple times say about coming back")

        if 'reviewer_name' in df.columns:
            repeat_all = (
                df[df['reviewer_name'].notna() & (df['reviewer_name'] != '')]
                .groupby('reviewer_name')
                .filter(lambda x: len(x) > 1)
            )
            if not repeat_all.empty:
                rp1, rp2, rp3 = st.columns(3)
                rp1.metric("Repeat reviewers", repeat_all['reviewer_name'].nunique())
                rp2.metric("Reviews from repeats", len(repeat_all))
                rp3.metric("Avg rating (repeats)", f"{repeat_all['rating'].mean():.2f}★")

                repeat_themes = pd.Series(
                    [t for tl in repeat_all['themes_list'] for t in tl if t]
                ).value_counts().head(6).reset_index()
                repeat_themes.columns = ['theme', 'count']
                if not repeat_themes.empty:
                    fig_rpt = px.bar(
                        repeat_themes, x='count', y='theme', orientation='h',
                        title='Top Themes — Repeat Customers',
                        color_discrete_sequence=['#C8A96E'],
                    )
                    fig_rpt.update_layout(height=280, margin=dict(t=40, l=160),
                                          yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig_rpt, use_container_width=True)

                repeat_phrases = [p for tl in repeat_all['standout_phrases_list'] for p in tl if p]
                if repeat_phrases:
                    st.markdown("**What keeps them coming back:**")
                    rpqc1, rpqc2 = st.columns(2)
                    for i, phrase in enumerate(repeat_phrases[:8]):
                        with (rpqc1 if i % 2 == 0 else rpqc2):
                            st.markdown(f"> *\"{phrase}\"*")
            else:
                st.info("No repeat reviewers in current filter.")


# ── Reviews ────────────────────────────────────────────────────────────────────
with tab_reviews:
    rc1, rc2 = st.columns([3, 1])
    with rc1:
        search = st.text_input("Search review text")
    with rc2:
        sort_by = st.selectbox("Sort by", ["Most Recent", "Most Persuasive", "Highest Rated"])

    view = df.copy()
    view['has_standout'] = view['standout_phrases_list'].apply(len) > 0

    if search:
        mask = (
            view['body'].fillna('').str.contains(search, case=False) |
            view['title'].fillna('').str.contains(search, case=False)
        )
        view = view[mask]

    if sort_by == "Most Recent":
        view = view.sort_values('date_created', ascending=False)
    elif sort_by == "Most Persuasive":
        view = view.sort_values(['has_standout', 'rating'], ascending=[False, False])
    elif sort_by == "Highest Rated":
        view = view.sort_values('rating', ascending=False)

    display = view[['date_created', 'scent', 'rating', 'title', 'body', 'is_recommended']].copy()
    display['date_created'] = display['date_created'].dt.strftime('%Y-%m-%d')
    st.dataframe(display, use_container_width=True, height=520)

    st.download_button(
        "Export as CSV",
        display.to_csv(index=False).encode('utf-8'),
        f"mozi_reviews_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
        key='export_csv',
    )


# ── Ask Claude ─────────────────────────────────────────────────────────────────
with tab_chat:
    st.caption("Ask anything about the reviews in the current filtered view.")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'inject_prompt' not in st.session_state:
        st.session_state.inject_prompt = None

    SUGGESTIONS = [
        "What are the biggest risks to retention right now?",
        "Which scent has the strongest word-of-mouth signal?",
        "What should we say in ads this week?",
        "Where are customers most likely to churn and why?",
        "What product or scent changes are customers asking for?",
    ]
    st.markdown("**Quick questions:**")
    s_cols = st.columns(len(SUGGESTIONS))
    for i, s in enumerate(SUGGESTIONS):
        if s_cols[i].button(s, use_container_width=True, key=f'suggest_{i}'):
            st.session_state.inject_prompt = s
            st.rerun()

    st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    user_input = st.chat_input("e.g. What are customers saying about Vanilla Moon?")
    prompt = st.session_state.inject_prompt or user_input
    if st.session_state.inject_prompt:
        st.session_state.inject_prompt = None

    if prompt:
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.write(prompt)

        n = len(df)
        scent_summary = (
            df.groupby('scent')['rating']
            .agg(['mean', 'count'])
            .round(2)
            .to_string()
        ) if n else "No data"

        stop_words = {'the','a','an','is','are','was','were','what','which','who','how',
                      'can','you','me','i','we','our','about','from','with','for','of',
                      'and','or','in','on','to','do','it','this','that','be','at','by'}
        keywords = [w.lower() for w in prompt.replace('?','').replace(',','').split()
                    if len(w) > 2 and w.lower() not in stop_words]

        with_body = df_all[df_all['body'].notna()].copy()

        # ── Date pre-filter ───────────────────────────────────────────────────
        prompt_lower = prompt.lower()
        today = datetime.now()
        date_filter_desc = ''

        quarter_m = re.search(r'\bq([1-4])\s*(202\d)\b', prompt_lower)
        year_m    = re.search(r'\b(202\d)\b', prompt_lower)

        if quarter_m:
            q, yr = int(quarter_m.group(1)), int(quarter_m.group(2))
            sm = (q - 1) * 3 + 1
            em = sm + 2
            d_from = f"{yr}-{sm:02d}-01"
            d_to   = f"{yr}-{em:02d}-30"
            with_body = with_body[
                (with_body['date_created'].fillna('') >= d_from) &
                (with_body['date_created'].fillna('') <= d_to + 'Z')
            ]
            date_filter_desc = f"Q{q} {yr}"
        elif 'last year' in prompt_lower:
            yr = today.year - 1
            with_body = with_body[
                (with_body['date_created'].fillna('') >= f"{yr}-01-01") &
                (with_body['date_created'].fillna('') <= f"{yr}-12-31Z")
            ]
            date_filter_desc = str(yr)
        elif 'this year' in prompt_lower:
            with_body = with_body[with_body['date_created'].fillna('') >= f"{today.year}-01-01"]
            date_filter_desc = str(today.year)
        elif 'last month' in prompt_lower or 'past month' in prompt_lower:
            cutoff = (today - timedelta(days=30)).strftime('%Y-%m-%d')
            with_body = with_body[with_body['date_created'].fillna('') >= cutoff]
            date_filter_desc = 'last 30 days'
        elif 'last week' in prompt_lower or 'past week' in prompt_lower:
            cutoff = (today - timedelta(days=7)).strftime('%Y-%m-%d')
            with_body = with_body[with_body['date_created'].fillna('') >= cutoff]
            date_filter_desc = 'last 7 days'
        elif year_m:
            yr = int(year_m.group(1))
            with_body = with_body[
                (with_body['date_created'].fillna('') >= f"{yr}-01-01") &
                (with_body['date_created'].fillna('') <= f"{yr}-12-31Z")
            ]
            date_filter_desc = str(yr)

        # ── Scent pre-filter ──────────────────────────────────────────────────
        SCENT_ALIASES = {
            'AW': ['alpine woods', 'alpine'],
            'CC': ['central coast'],
            'CZ': ['signature cozy', 'cozy cashmere', 'cozy'],
            'DP': ['desert poppy'],
            'FC': ['free and clear', 'free & clear', 'free+clear', 'fragrance free', 'fragrance-free'],
            'GH': ['golden hour'],
            'HR': ['hollywood rouge', 'hollywood'],
            'MM': ['malibu mornings', 'malibu'],
            'SD': ['sugar dew', 'sugar dew'],
            'VM': ['vanilla moon', 'vanilla'],
        }
        matched_skus = [
            sku for sku, aliases in SCENT_ALIASES.items()
            if any(a in prompt_lower for a in aliases)
            or re.search(rf'\b{sku.lower()}\b', prompt_lower)
        ]
        scent_filter_desc = ''
        if matched_skus and 'sku' in with_body.columns:
            with_body = with_body[with_body['sku'].isin(matched_skus)]
            scent_filter_desc = ', '.join(matched_skus)

        # ── Keyword matching — no hard cap, 1500 safety ceiling ───────────────
        MAX_REVIEWS = 1500

        SYNONYMS = {
            'skin': ['skin', 'sensitiv', 'irritat', 'allerg', 'rash', 'eczema', 'reaction'],
            'hormone': ['hormone', 'endocrin', 'disrupt', 'chemical', 'toxic', 'clean formula', 'natural'],
            'safety': ['safe', 'harmful', 'chemical', 'toxic', 'natural', 'clean'],
            'scent': ['scent', 'smell', 'fragrance', 'aroma'],
            'clean': ['clean', 'wash', 'stain', 'dirt', 'soil'],
        }
        expanded = list(keywords)
        for kw in keywords:
            for base, syns in SYNONYMS.items():
                if kw in base or base in kw:
                    expanded.extend(syns)
        expanded = list(set(expanded))

        if expanded:
            pattern = '|'.join(expanded)
            mask = (
                with_body['body'].str.contains(pattern, case=False, na=False) |
                with_body['title'].fillna('').str.contains(pattern, case=False, na=False)
            )
            relevant = with_body[mask].sort_values('date_created', ascending=False)
            if len(relevant) < 15:
                recent = with_body[~with_body.index.isin(relevant.index)].sort_values('date_created', ascending=False).head(20)
                relevant = pd.concat([relevant, recent])
        else:
            relevant = with_body.sort_values('date_created', ascending=False)

        if len(relevant) > MAX_REVIEWS:
            relevant = relevant.head(MAX_REVIEWS)

        filter_parts = [p for p in [scent_filter_desc, date_filter_desc] if p]
        filter_label = f" — filtered to: {', '.join(filter_parts)}" if filter_parts else ''

        sample_text = '\n'.join(
            f"[{r['scent']} | {r['rating']}★ | {str(r['date_created'])[:10]}] {r['title']}: {str(r['body'])[:400]}"
            for _, r in relevant.iterrows()
        )

        avg_str = f"{df['rating'].mean():.2f}" if n else 'N/A'

        aggregate_context = build_aggregate_context(df_all)
        wa_week, wa_count, wa = load_weekly_analysis()
        weekly_intel = ""
        if wa:
            themes_str = '\n'.join(
                f"  - {t['theme']} ({t['count']} mentions): {t['description'][:180]}"
                for t in wa.get('themes', [])
            )
            phrases_str = '\n'.join(
                f"  - \"{p['phrase']}\" (x{p['count']}) — {p['angle']}"
                for p in wa.get('phrases', [])[:6]
            )
            emerging_str = '\n'.join(
                f"  - {e['observation']} | Why it matters: {e['why_notable'][:150]}"
                for e in wa.get('emerging', [])
            )
            by_scent_lines = []
            for sku, info in wa.get('by_scent', {}).items():
                scent_name = SCENT_NAMES.get(sku, sku)
                sentiment = info.get('sentiment', '')
                notable = info.get('notable', '')
                top_phrase = info.get('top_phrase', '')
                if notable and notable != 'No reviews identified for this SKU in this window':
                    line = f"  - {sku} ({scent_name}): {sentiment}"
                    if top_phrase:
                        line += f" | Top phrase: \"{top_phrase}\""
                    line += f" | {notable[:200]}"
                    by_scent_lines.append(line)
            by_scent_str = '\n'.join(by_scent_lines)

            weekly_intel = f"""
── LATEST WEEKLY INTELLIGENCE (week of {wa_week}, {wa_count} reviews analyzed) ──

Top themes this period:
{themes_str}

Standout customer phrases:
{phrases_str}

By scent:
{by_scent_str}

Emerging signals:
{emerging_str}
"""

        system = f"""You are a senior brand strategist and marketing analyst for Mozi Wash. Answer strategically — connect insights to business implications, not just describe what customers said.

── BRAND BRIEF ──
Mozi Wash is a premium laundry detergent sold in elegant metal tins, marketed as a sensory upgrade to everyday laundry. DTC brand, sold via subscription and one-time purchase. Price point is premium — customers are making a considered purchase and expect a luxury experience.

Scent lineup (10 SKUs):
  AW – Alpine Woods: woodsy, outdoorsy, masculine-leaning. Strong male household approval signal.
  CC – Central Coast: bright, clean, coastal. High volume of stranger-compliment mentions; some "too masculine" or "too strong" feedback.
  CZ – Signature Cozy: warm, cozy, soft. Known scent-longevity issue on the March 2026 batch (6 Degrees vendor) — customers report scent fading post-dryer.
  DP – Desert Poppy: floral-adjacent, warm. Occasional reports of color/consistency changes between batches.
  FC – Free & Clear: fragrance-free, sensitivity-focused. Lower review volume.
  GH – Golden Hour: warm, golden, designer-adjacent (customers compare to Le Labo Santal 33).
  HR – Hollywood Rouge: bold, glamorous, feminine. Recent scent-longevity complaints emerging.
  MM – Malibu Mornings: fresh, beachy, light.
  SD – Sugar Dew: sweet, soft, feminine.
  VM – Vanilla Moon: warm, vanilla-forward. Generates "mistaken for perfume" moments; some fade-after-drying feedback.

Business model: subscription-first DTC. Subscription complaints are a real churn signal. Samples are a key conversion funnel — many reviews explicitly mention starting with a sample pack then subscribing.

Core customer: women who treat laundry as a sensory ritual, love premium home goods, and will pay more for scent that lasts. They share and compare with household members; male household approval ("my husband loves it") is a recurring loyalty signal.

Competitors: Frey (clean/functional positioning), Dirty Labs (performance/science), Tyler Candle (luxury scent adjacent), Laundry Sauce (mentioned by switchers, negative customer service rep).

Operational context: Products come from two vendors — 6 Degrees (March Order batch, started arriving April 2026) and Product Society (January Order batch). The March Order - 6 Degrees batch has generated a pattern of scent-longevity complaints across multiple SKUs, most concentrated in CZ and HR. This is a known quality signal worth flagging when relevant.

── ALL-TIME DATA ──
{aggregate_context}

── CURRENT FILTERED VIEW ({n:,} reviews · avg {avg_str}★) ──
Scent breakdown (filtered):
{scent_summary}
{weekly_intel}
── SAMPLE REVIEWS FOR THIS QUERY ({len(relevant)} reviews{filter_label}) ──
Use the aggregate stats above for counts and trends. Use these reviews for direct quotes and specific examples.

{sample_text}

Be direct and strategic. Lead with the insight, not the methodology."""

        client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

        with st.chat_message('assistant'):
            with st.spinner():
                try:
                    resp = client.messages.create(
                        model='claude-sonnet-4-6',
                        max_tokens=2048,
                        system=system,
                        messages=[
                            {'role': m['role'], 'content': m['content']}
                            for m in st.session_state.messages
                        ],
                    )
                    answer = resp.content[0].text
                except Exception as e:
                    if 'rate_limit' in str(e).lower() or 'overloaded' in str(e).lower():
                        answer = "Claude is busy right now — wait 30 seconds and try again."
                    else:
                        answer = "Something went wrong. Please try again."
                st.write(answer)

        st.session_state.messages.append({'role': 'assistant', 'content': answer})
