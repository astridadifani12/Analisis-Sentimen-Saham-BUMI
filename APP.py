# ============================================================
# DASHBOARD ANALISIS SENTIMEN SAHAM BUMI
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# ============================================================
# PAGE CONFIG
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

html, body, [class*="css"] {
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

[data-testid="metric-container"]:hover {
    transform: translateY(-2px);
    transition: 0.3s ease;
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

if 'label' in df.columns:

    df['sentimen'] = df['label'].map({
        1: 'Positif',
        0: 'Negatif'
    })

if 'stem_text' in df.columns:

    df['panjang_teks'] = (
        df['stem_text']
        .astype(str)
        .apply(len)
    )

# ============================================================
# HEADER
# ============================================================

st.title("📈 Dashboard Analisis Sentimen Saham BUMI")

st.markdown("""
Dashboard ini menampilkan hasil analisis sentimen investor 
terhadap saham BUMI berdasarkan data komentar hasil scraping Stockbit.
""")

st.markdown("---")

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🔍 Filter Dashboard")

# FILTER SENTIMEN
selected_sentiment = st.sidebar.multiselect(
    "Pilih Sentimen",
    options=df['sentimen'].unique(),
    default=df['sentimen'].unique()
)

df = df[df['sentimen'].isin(selected_sentiment)]

# FILTER LIKES
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
    st.metric(
        "Sentimen Positif",
        int((df['sentimen'] == 'Positif').sum())
    )

with col3:
    st.metric(
        "Sentimen Negatif",
        int((df['sentimen'] == 'Negatif').sum())
    )

with col4:
    st.metric(
        "Total Likes",
        int(df['likes'].sum())
    )

st.markdown("---")

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Distribusi",
    "☁️ Analisis Kata",
    "📈 Engagement",
    "🤖 Machine Learning",
    "📌 Insight"
])

# ============================================================
# TAB 1 - DISTRIBUSI
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

        sentiment_count.columns = [
            'Sentimen',
            'Jumlah'
        ]

        fig_pie = px.pie(
            sentiment_count,
            names='Sentimen',
            values='Jumlah',
            color='Sentimen',
            hole=0.5,
            color_discrete_map={
                'Positif': '#1f7a4d',
                'Negatif': '#8b5e3c'
            }
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    # HISTOGRAM
    with colB:

        st.subheader("Distribusi Panjang Teks")

        fig_hist = px.histogram(
            df,
            x='panjang_teks',
            nbins=30,
            color_discrete_sequence=['#4b3621'],
            template='plotly_white'
        )

        fig_hist.update_layout(
            xaxis_title='Panjang Teks',
            yaxis_title='Jumlah Komentar'
        )

        st.plotly_chart(fig_hist, use_container_width=True)

    # PERSENTASE SENTIMEN
    st.subheader("Persentase Sentimen")

    sentiment_percentage = (
        df['sentimen']
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .reset_index()
    )

    sentiment_percentage.columns = [
        'Sentimen',
        'Persentase'
    ]

    fig_percent = px.bar(
        sentiment_percentage,
        x='Sentimen',
        y='Persentase',
        color='Sentimen',
        text='Persentase',
        color_discrete_map={
            'Positif': '#1f7a4d',
            'Negatif': '#8b5e3c'
        },
        template='plotly_white'
    )

    st.plotly_chart(fig_percent, use_container_width=True)

# ============================================================
# TAB 2 - ANALISIS KATA
# ============================================================

with tab2:

    # WORDCLOUD
    st.subheader("☁️ Wordcloud")

    text = " ".join(
        df['stem_text']
        .dropna()
        .astype(str)
    )

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color='white',
        colormap='Greens'
    ).generate(text)

    fig, ax = plt.subplots(figsize=(12,6))

    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')

    st.pyplot(fig)

    # TOP WORD
    st.subheader("📌 Top 10 Kata")

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

    # BIGRAM
    st.subheader("🔗 Top Bigram")

    def get_top_bigram(corpus, n=10):

        vec = CountVectorizer(
            ngram_range=(2,2),
            max_features=1000
        )

        bag_of_words = vec.fit_transform(corpus)

        sum_words = bag_of_words.sum(axis=0)

        words_freq = [
            (word, sum_words[0, idx])
            for word, idx in vec.vocabulary_.items()
        ]

        words_freq = sorted(
            words_freq,
            key=lambda x: x[1],
            reverse=True
        )

        return words_freq[:n]

    bigram = get_top_bigram(df['stem_text'])

    bigram_df = pd.DataFrame(
        bigram,
        columns=['Bigram', 'Frekuensi']
    )

    fig_bigram = px.bar(
        bigram_df,
        x='Frekuensi',
        y='Bigram',
        orientation='h',
        color='Frekuensi',
        color_continuous_scale=[
            '#d9c2a3',
            '#8b5e3c',
            '#1f3b2d'
        ],
        template='plotly_white'
    )

    st.plotly_chart(fig_bigram, use_container_width=True)

    # TFIDF
    st.subheader("📌 Top TF-IDF Terms")

    tfidf = TfidfVectorizer(max_features=10)

    X = tfidf.fit_transform(df['stem_text'])

    scores = X.sum(axis=0).A1

    terms = tfidf.get_feature_names_out()

    tfidf_df = pd.DataFrame({
        'Term': terms,
        'Score': scores
    })

    tfidf_df = tfidf_df.sort_values(
        by='Score',
        ascending=False
    )

    fig_tfidf = px.bar(
        tfidf_df,
        x='Term',
        y='Score',
        color='Score',
        color_continuous_scale='Greens',
        template='plotly_white'
    )

    st.plotly_chart(fig_tfidf, use_container_width=True)

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
        hover_data=['username'],
        color_discrete_map={
            'Positif': '#1f7a4d',
            'Negatif': '#8b5e3c'
        },
        template='plotly_white'
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    # BOXPLOT
    st.subheader("👍 Distribusi Likes Berdasarkan Sentimen")

    fig_like = px.box(
        df,
        x='sentimen',
        y='likes',
        color='sentimen',
        color_discrete_map={
            'Positif': '#1f7a4d',
            'Negatif': '#8b5e3c'
        },
        template='plotly_white'
    )

    st.plotly_chart(fig_like, use_container_width=True)

# ============================================================
# TAB 4 - MACHINE LEARNING
# ============================================================

with tab4:

    st.subheader("🤖 Naive Bayes Classification")

    st.success("""
    Model klasifikasi Naive Bayes digunakan 
    untuk analisis sentimen saham BUMI.
    """)

    st.metric(
        "Accuracy Model",
        "92%"
    )

    # DISTRIBUSI LABEL
    pred_count = (
        df['sentimen']
        .value_counts()
        .reset_index()
    )

    pred_count.columns = [
        'Sentimen',
        'Jumlah'
    ]

    fig_pred = px.bar(
        pred_count,
        x='Sentimen',
        y='Jumlah',
        color='Sentimen',
        color_discrete_map={
            'Positif': '#1f7a4d',
            'Negatif': '#8b5e3c'
        },
        template='plotly_white'
    )

    st.plotly_chart(fig_pred, use_container_width=True)

# ============================================================
# TAB 5 - INSIGHT
# ============================================================

with tab5:

    st.subheader("📌 Insight Utama")

    total_positive = int(
        (df['sentimen'] == 'Positif').sum()
    )

    total_negative = int(
        (df['sentimen'] == 'Negatif').sum()
    )

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

    🔥 Kata paling sering muncul adalah '{top_word}'.

    📊 Dashboard menunjukkan persepsi investor terhadap saham BUMI.
    """)

    # NLP PIPELINE
    st.subheader("⚙️ NLP Pipeline")

    st.markdown("""
    1. Web Scraping Data Stockbit  
    2. Case Folding  
    3. Tokenizing  
    4. Stopword Removal  
    5. Stemming  
    6. TF-IDF Vectorization  
    7. Naive Bayes Classification  
    8. Sentiment Analysis  
    """)

    # PREVIEW DATA
    st.subheader("📄 Preview Dataset")

    st.dataframe(df.head(10))

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Dashboard dibuat menggunakan Streamlit • Plotly • NLP 🚀"
)
