"""Mozi Wash Review Intelligence — Streamlit dashboard."""
import streamlit as st
import pandas as pd
import plotly.express as px
import json
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
    for col in ('themes', 'standout_phrases', 'emotional_triggers', 'comparison_phrases'):
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


try:
    df_all = load_data()
except FileNotFoundError:
    st.error("reviews.csv not found. Run okendo_dashboard.py first to generate the data file.")
    st.stop()

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Filters")
    date_range = st.date_input(
        "Date range",
        value=[df_all['date_created'].min().date(), df_all['date_created'].max().date()],
        min_value=df_all['date_created'].min().date(),
        max_value=df_all['date_created'].max().date(),
    )
    all_skus = sorted(df_all['sku'].dropna().unique())
    sel_scents = st.multiselect(
        "Scents", options=all_skus,
        format_func=lambda x: SCENT_NAMES.get(x, x),
        default=all_skus,
    )
    min_rating = st.slider("Min rating", 1, 5, 1)

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
tab_themes, tab_hooks, tab_attrs, tab_reviews, tab_chat = st.tabs([
    "Themes by Scent", "Marketing Hooks", "Profile Attributes", "Reviews", "Ask Claude"
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


# ── Profile Attributes ─────────────────────────────────────────────────────────
with tab_attrs:
    st.caption("Reviewer demographics from the current filtered view")

    if df.empty:
        st.info("No data for current filters.")
    else:
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


# ── Ask Claude ─────────────────────────────────────────────────────────────────
with tab_chat:
    st.caption("Ask anything about the reviews in the current filtered view.")

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'inject_prompt' not in st.session_state:
        st.session_state.inject_prompt = None

    SUGGESTIONS = [
        "What themes drive the most loyalty?",
        "Best quotes for ad copy?",
        "Which scent gets the most compliments?",
        "What switching language are customers using?",
        "What are skeptic-converted customers saying?",
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
            relevant = (
                with_body[mask]
                .sort_values('rating', ascending=False)
                .head(150)
            )
            if len(relevant) < 15:
                recent = with_body[~with_body.index.isin(relevant.index)].sort_values('date_created', ascending=False).head(20)
                relevant = pd.concat([relevant, recent])
        else:
            relevant = with_body.sort_values('date_created', ascending=False).head(60)

        sample_text = '\n'.join(
            f"[{r['scent']} | {r['rating']}★] {r['title']}: {str(r['body'])[:200]}"
            for _, r in relevant.iterrows()
        )

        avg_str = f"{df['rating'].mean():.2f}" if n else 'N/A'
        system = f"""You are a marketing analyst for Mozi Wash, a premium laundry detergent brand sold in beautiful metal tins.
Competitors: Frey, Dirty Labs, Tyler Candle. Core customer: women who love premium home goods and care deeply about scent.

Total database: {len(df_all):,} reviews (all time) · Filtered view: {n:,} reviews · Avg rating: {avg_str}
Scent breakdown (mean rating, count):
{scent_summary}

IMPORTANT: The {len(relevant)} reviews below are the COMPLETE set of matched reviews from the full database for this query — not a sample. You have everything. When asked for more examples, pull directly from the list below. Do not say you need to query the database or that you only have a sample.

Reviews ({len(relevant)} total matched):
{sample_text}

Answer concisely. Focus on actionable marketing insights when relevant."""

        client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

        with st.chat_message('assistant'):
            with st.spinner():
                try:
                    resp = client.messages.create(
                        model='claude-sonnet-4-6',
                        max_tokens=1024,
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
