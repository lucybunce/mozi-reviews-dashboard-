"""Mozi Wash Review Intelligence — Streamlit dashboard."""
import streamlit as st
import pandas as pd
import plotly.express as px
from anthropic import Anthropic
from datetime import datetime, timedelta

st.set_page_config(page_title="Mozi Wash Review Intelligence", layout="wide")

SCENT_NAMES = {
    'AW': 'Alpine Woods', 'CC': 'Central Coast', 'CZ': 'Signature Cozy',
    'DP': 'Desert Poppy', 'FC': 'Free & Clear', 'GH': 'Golden Hour',
    'HR': 'Hollywood Rouge', 'MM': 'Malibu Mornings', 'SD': 'Sugar Dew', 'VM': 'Vanilla Moon',
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
tab_trends, tab_scent, tab_reviews, tab_chat = st.tabs(["Trends", "By Scent", "Reviews", "Ask Claude"])

with tab_trends:
    if df.empty:
        st.info("No data for current filters.")
    else:
        weekly = (
            df.set_index('date_created')
            .resample('W')['review_id']
            .count()
            .reset_index(name='reviews')
        )
        fig = px.line(weekly, x='date_created', y='reviews', title='Reviews per week', markers=True)
        fig.update_layout(height=280, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

        avg_wk = (
            df.set_index('date_created')
            .resample('W')['rating']
            .mean()
            .reset_index(name='avg_rating')
        )
        overall_avg = df['rating'].mean()
        fig2 = px.line(avg_wk, x='date_created', y='avg_rating', title='Avg rating per week')
        fig2.update_layout(height=280, margin=dict(t=40), yaxis_range=[1, 5])
        fig2.add_hline(
            y=overall_avg, line_dash='dot', line_color='gray',
            annotation_text=f"Overall avg: {overall_avg:.2f}",
        )
        st.plotly_chart(fig2, use_container_width=True)

with tab_scent:
    if df.empty:
        st.info("No data for current filters.")
    else:
        col_l, col_r = st.columns(2)
        with col_l:
            scent_stats = (
                df.groupby('scent')
                .agg(avg_rating=('rating', 'mean'), reviews=('review_id', 'count'))
                .reset_index()
                .sort_values('avg_rating')
            )
            fig3 = px.bar(
                scent_stats, y='scent', x='avg_rating', orientation='h',
                title='Avg Rating by Scent', text='avg_rating',
                color='avg_rating', color_continuous_scale='RdYlGn', range_color=[3.5, 5],
            )
            fig3.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig3.update_layout(height=380, showlegend=False, margin=dict(t=40))
            st.plotly_chart(fig3, use_container_width=True)
        with col_r:
            vol = df.groupby('scent').size().reset_index(name='count')
            fig4 = px.pie(vol, values='count', names='scent', title='Review Volume by Scent')
            fig4.update_layout(height=380, margin=dict(t=40))
            st.plotly_chart(fig4, use_container_width=True)

        attrs = {
            'scent_appeal': 'Scent Appeal',
            'cleaning_power': 'Cleaning Power',
            'cap_pouring': 'Cap / Pouring',
        }
        attr_df = (
            df.groupby('scent')[list(attrs.keys())]
            .mean()
            .reset_index()
            .melt(id_vars='scent', var_name='attr', value_name='score')
        )
        attr_df['attr'] = attr_df['attr'].map(attrs)
        fig5 = px.bar(
            attr_df, x='scent', y='score', color='attr', barmode='group',
            title='Attribute Scores by Scent (1–5)', range_y=[0, 5],
        )
        fig5.update_layout(height=380, margin=dict(t=40))
        st.plotly_chart(fig5, use_container_width=True)

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

        sample = (
            df[df['body'].notna()]
            .sort_values('date_created', ascending=False)
            .head(30)
        )
        sample_text = '\n'.join(
            f"[{r['scent']} | {r['rating']}★] {r['title']}: {str(r['body'])[:200]}"
            for _, r in sample.iterrows()
        )

        avg_str = f"{df['rating'].mean():.2f}" if n else 'N/A'
        system = f"""You are a marketing analyst for Mozi Wash, a premium laundry detergent brand sold in beautiful metal tins.
Competitors: Frey, Dirty Labs, Tyler Candle. Core customer: women who love premium home goods and care deeply about scent.

Current filtered dataset: {n:,} reviews · Avg rating: {avg_str}
Scent breakdown (mean rating, count):
{scent_summary}

Recent review sample (up to 30):
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
