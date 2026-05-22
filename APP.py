# app.py
# ============================================================
# DASHBOARD ANALISIS SENTIMEN SAHAM BUMI
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================

st.set_page_config(
    page_title="Dashboard Sentimen Saham BUMI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

.main {
    background-color: #f7f4ef;
}

[data-testid="metric-container"] {
    background: linear-gradient(135deg, #ffffff 0%, #f5efe6 100%);
    border: 1px solid #d6c7b2;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1f3b2d 0%, #4b3621 100%);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

h1, h2, h3 {
    color: #1f3b2d;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #e9dfd0;
    padding: 10px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #1f3b2d !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("df_clean.csv")

    return df


df = load_data()

# ============================================================
# PREPROCESSING
# ============================================================

# Label sentimen
if 'label' in df.columns:

    df['sentimen'] = df['label'].map({
        1: 'Positif',
        0: 'Negatif'
    })

# Panjang teks
if 'stem_text' in df.columns:

    df['panjang_teks'] = df['stem_text'].astype(str).apply(len)

# ============================================================
# HEADER
# ============================================================

st.title("📈 Dashboard Analisis Sentimen Saham BUMI")

st.markdown("""
Dashboard ini menampilkan hasil analisis sentimen investor terhadap saham BUMI
berdasarkan data komentar hasil scraping Stockbit.
""")

st.markdown("---")

# ============================================================
# SIDEBAR FILTER
# ============================================================

st.sidebar.header("🔍 Filter Dashboard")

# FILTER SENTIMEN
if 'sentimen' in df.columns:

    selected_sentiment = st.sidebar.multiselect(
        "Pilih Sentimen",
        options=df['sentimen'].unique(),
        default=df['sentimen'].unique()
    )

    df = df[df['sentimen'].isin(selected_sentiment)]

# FILTER LIKE
if 'likes' in df.columns:

    min_like = int(df['likes'].min())
    max_like = int(df['likes'].max())

    selected_like = st.sidebar.slider(
        "Range Likes",
        min_value=min_like,
        max_value=max_like,
        value=(min_like, max_like)
    )

    df = df[
        (df['likes'] >= selected_like[0]) &
        (df['likes'] <= selected_like[1])
    ]

# ============================================================
# METRICS
# ============================================================

st.subheader("📌 Ringkasan Data")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Komentar",
        f"{len(df):,}"
    )

with col2:
    if 'sentimen' in df.columns:
        st.metric(
            "Sentimen Positif",
            int((df['sentimen'] == 'Positif').sum())
        )

with col3:
    if 'sentimen' in df.columns:
        st.metric(
            "Sentimen Negatif",
            int((df['sentimen'] == 'Negatif').sum())
        )

with col4:
    if 'likes' in df.columns:
        st.metric(
            "Total Likes",
            int(df['likes'].sum())
        )

st.markdown("---")

# ============================================================
# TABS DASHBOARD
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Distribusi Sentimen",
    "☁️ Analisis Kata",
    "📈 Engagement",
    "📌 Insight"
])

# ============================================================
# TAB 1 - DISTRIBUSI SENTIMEN
# ============================================================

with tab1:

    colA, colB = st.columns(2)

    # PIE CHART
    with colA:

        st.subheader("Distribusi Sentimen")

        sentiment_count = (
            df['sentimen']
            .value_counts()
            .reset_index()
        )

        sentiment_count.columns = ['Sentimen', 'Jumlah']

        fig_pie = px.pie(
            sentiment_count,
            names='Sentimen',
            values='Jumlah',
            color='Sentimen',
            color_discrete_map={
                'Positif': '#1f7a4d',
                'Negatif': '#8b5e3c'
            },
            hole=0.5
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    # HISTOGRAM PANJANG TEKS
    with colB:

        st.subheader("Distribusi Panjang Teks")

        fig_length = px.histogram(
            df,
            x='panjang_teks',
            nbins=30,
            color_discrete_sequence=['#4b3621'],
            template='plotly_white'
        )

        fig_length.update_layout(
            xaxis_title='Panjang Karakter',
            yaxis_title='Jumlah Komentar'
        )

        st.plotly_chart(fig_length, use_container_width=True)

    # BAR SENTIMEN
    st.subheader("Perbandingan Sentimen")

    fig_bar = px.bar(
        sentiment_count,
        x='Sentimen',
        y='Jumlah',
        color='Sentimen',
        color_discrete_map={
            'Positif': '#1f7a4d',
            'Negatif': '#8b5e3c'
        },
        text='Jumlah',
        template='plotly_white'
    )

    st.plotly_chart(fig_bar, use_container_width=True)

# ============================================================
# TAB 2 - ANALISIS KATA
# ============================================================

with tab2:

    st.subheader("☁️ Wordcloud")

    text = " ".join(df['stem_text'].dropna().astype(str))

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color='white',
        colormap='Greens'
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    st.pyplot(fig)

    # TOP WORDS
    st.subheader("📌 Top 10 Kata Paling Sering Muncul")

    words = text.split()
    word_freq = Counter(words)

    top_words = pd.DataFrame(
        word_freq.most_common(10),
        columns=['Kata', 'Frekuensi']
    )

    fig_words = px.bar(
        top_words,
        x='Kata',
        y='Frekuensi',
        color='Frekuensi',
        color_continuous_scale=[
            '#d9c2a3',
            '#8b5e3c',
            '#1f3b2d'
        ],
        text='Frekuensi',
        template='plotly_white'
    )

    st.plotly_chart(fig_words, use_container_width=True)

# ============================================================
# TAB 3 - ENGAGEMENT
# ============================================================

with tab3:

    st.subheader("📈 Analisis Engagement")

    # SCATTER
    fig_scatter = px.scatter(
        df,
        x='likes',
        y='panjang_teks',
        color='sentimen',
        color_discrete_map={
            'Positif': '#1f7a4d',
            'Negatif': '#8b5e3c'
        },
        hover_data=['username'],
        template='plotly_white'
    )

    fig_scatter.update_layout(
        xaxis_title='Jumlah Likes',
        yaxis_title='Panjang Teks'
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    # TOP USER
    st.subheader("👤 Top User Berdasarkan Likes")

    top_user = (
        df.groupby('username')['likes']
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig_user = px.bar(
        top_user,
        x='username',
        y='likes',
        color='likes',
        color_continuous_scale='Greens',
        text='likes',
        template='plotly_white'
    )

    st.plotly_chart(fig_user, use_container_width=True)

# ============================================================
# TAB 4 - INSIGHT
# ============================================================

with tab4:

    st.subheader("📌 Insight Utama")

    total_positive = int((df['sentimen'] == 'Positif').sum())
    total_negative = int((df['sentimen'] == 'Negatif').sum())

    dominant_sentiment = (
        'Positif'
        if total_positive > total_negative
        else 'Negatif'
    )

    top_word = top_words.iloc[0]['Kata']

    st.info(f"""
    📈 Sentimen dominan terhadap saham BUMI adalah sentimen {dominant_sentiment}.

    💬 Total komentar positif sebanyak {total_positive:,} komentar.

    ⚠️ Total komentar negatif sebanyak {total_negative:,} komentar.

    🔥 Kata yang paling sering muncul adalah '{top_word}'.

    📊 Dashboard menunjukkan pola engagement dan persepsi investor terhadap saham BUMI.
    """)

    # DATA PREVIEW
    st.subheader("📄 Preview Dataset")

    st.dataframe(df.head(10))

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Dashboard dibuat menggunakan Streamlit • Plotly • NLP 🚀"
)
