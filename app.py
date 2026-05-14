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


@st.cache_data(ttl=3600)
def load_data():
    df = pd.read_csv('reviews.csv')
    df['date_created'] = pd.to_datetime(df['date_created'], utc=True).dt.tz_convert(None)
    df['scent'] = df['sku'].map(SCENT_NAMES).fillna(df['sku'])
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
tab_themes, tab_attrs, tab_reviews, tab_chat = st.tabs([
    "Themes by Scent", "Profile Attributes", "Reviews", "Ask Claude"
])

# ── Themes by Scent (cached at module level so failures don't get cached) ──────
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


with tab_themes:
    st.caption("Claude-generated marketing themes per scent · Based on all reviews · Updates weekly with new data")

    try:
        themes_data = get_themes(len(df_all), str(df_all['date_created'].max().date()))
    except Exception:
        st.warning("Claude is temporarily busy — refresh the page in a minute to load themes.")
        themes_data = None

    if themes_data:
        skus_present = [s for s in sorted(SCENT_NAMES.keys()) if s in themes_data]
        cols = st.columns(2)
        for i, sku in enumerate(skus_present):
            name = SCENT_NAMES.get(sku, sku)
            color = SCENT_COLORS.get(sku, '#888')
            scent_themes = themes_data.get(sku, {}).get('themes', [])
            with cols[i % 2]:
                st.markdown(f"#### {name}")
                for t in scent_themes:
                    with st.expander(f"**{t['theme']}** — {t['description']}"):
                        st.markdown(f"*\"{t['example']}\"*")
                st.divider()

# ── Profile Attributes ─────────────────────────────────────────────────────────
with tab_attrs:
    st.caption("Reviewer demographics from the current filtered view")

    if df.empty:
        st.info("No data for current filters.")
    else:
        col1, col2 = st.columns(2)

        AGE_ORDER = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
        WASH_ORDER = ['1', '2', '3-4', '5+']
        HOUSEHOLD_ORDER = ['1', '2 - 4', '5+']

        with col1:
            age_counts = (
                df['reviewer_age'].dropna()
                .value_counts()
                .reindex(AGE_ORDER)
                .dropna()
                .reset_index()
            )
            age_counts.columns = ['age', 'count']
            fig_age = px.bar(
                age_counts, x='age', y='count',
                title='Reviewers by Age Group',
                color_discrete_sequence=['#C8A96E'],
            )
            fig_age.update_layout(height=320, margin=dict(t=40))
            st.plotly_chart(fig_age, use_container_width=True)

            wash_counts = (
                df['reviewer_washes'].dropna()
                .astype(str).str.strip()
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
            fig_hh = px.bar(
                hh_counts, x='household', y='count',
                title='Household Size',
                color_discrete_sequence=['#5BAD6F'],
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
    search = st.text_input("Search review text")
    view = df[['date_created', 'scent', 'rating', 'title', 'body', 'is_recommended']].copy()
    view['date_created'] = view['date_created'].dt.strftime('%Y-%m-%d')
    if search:
        mask = (
            view['body'].fillna('').str.contains(search, case=False) |
            view['title'].fillna('').str.contains(search, case=False)
        )
        view = view[mask]
    st.dataframe(view.sort_values('date_created', ascending=False), use_container_width=True, height=520)

# ── Ask Claude ─────────────────────────────────────────────────────────────────
with tab_chat:
    st.caption("Ask anything about the reviews shown in the current filtered view.")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.write(msg['content'])

    if prompt := st.chat_input("e.g. What are customers saying about Vanilla Moon?"):
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

        # Search all reviews (full dataset, ignoring sidebar filters)
        stop_words = {'the','a','an','is','are','was','were','what','which','who','how',
                      'can','you','me','i','we','our','about','from','with','for','of',
                      'and','or','in','on','to','do','it','this','that','be','at','by'}
        keywords = [w.lower() for w in prompt.replace('?','').replace(',','').split()
                    if len(w) > 2 and w.lower() not in stop_words]

        with_body = df_all[df_all['body'].notna()].copy()

        # Expand keywords with synonyms for common topics
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
                st.write(answer)

        st.session_state.messages.append({'role': 'assistant', 'content': answer})
