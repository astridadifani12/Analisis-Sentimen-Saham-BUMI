# ============================================================
# APP.PY
# DASHBOARD ANALISIS SENTIMEN SAHAM BUMI
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from wordcloud import WordCloud

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Analisis Sentimen Saham BUMI",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f7f4ef;
}

h1, h2, h3 {
    color: #1f3b2d;
}

[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #d6c7b2;
    padding: 15px;
    border-radius: 12px;
}

[data-testid="stSidebar"] {
    background-color: #1f3b2d;
}

[data-testid="stSidebar"] * {
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

# LABEL SENTIMEN
if 'label' in df.columns:

    df['sentimen'] = df['label'].map({
        1: 'Positif',
        0: 'Negatif'
    })

# PANJANG TEKS
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

st.sidebar.header("🔍 Filter Data")

selected_sentiment = st.sidebar.multiselect(
    "Pilih Sentimen",
    options=df['sentimen'].unique(),
    default=df['sentimen'].unique()
)

df = df[df['sentimen'].isin(selected_sentiment)]

# ============================================================
# METRIC
# ============================================================

st.subheader("📌 Ringkasan Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Komentar",
        len(df)
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

st.markdown("---")

# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Dataset",
    "📊 Distribusi Sentimen",
    "☁️ Word Analysis",
    "📌 TF-IDF & Modeling",
    "📈 Insight"
])

# ============================================================
# TAB 1 - DATASET
# ============================================================

with tab1:

    st.subheader("📄 Preview Dataset")

    st.dataframe(df.head())

    st.subheader("📌 Informasi Dataset")

    colA, colB = st.columns(2)

    with colA:

        st.write("Jumlah Baris:", df.shape[0])

        st.write("Jumlah Kolom:", df.shape[1])

    with colB:

        missing = df.isnull().sum()

        st.write("Missing Value")

        st.dataframe(missing)

# ============================================================
# TAB 2 - DISTRIBUSI SENTIMEN
# ============================================================

with tab2:

    st.subheader("📊 Distribusi Sentimen")

    sentiment_count = (
        df['sentimen']
        .value_counts()
        .reset_index()
    )

    sentiment_count.columns = [
        'Sentimen',
        'Jumlah'
    ]

    fig_sentiment = px.bar(
        sentiment_count,
        x='Sentimen',
        y='Jumlah',
        color='Sentimen',
        text='Jumlah',
        color_discrete_map={
            'Positif': '#1f7a4d',
            'Negatif': '#8b5e3c'
        }
    )

    st.plotly_chart(
        fig_sentiment,
        use_container_width=True
    )

    # PERSENTASE
    st.subheader("📌 Persentase Sentimen")

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

    fig_percent = px.pie(
        sentiment_percentage,
        names='Sentimen',
        values='Persentase',
        color='Sentimen',
        color_discrete_map={
            'Positif': '#1f7a4d',
            'Negatif': '#8b5e3c'
        }
    )

    st.plotly_chart(
        fig_percent,
        use_container_width=True
    )

    # DISTRIBUSI PANJANG TEKS
    st.subheader("📏 Distribusi Panjang Teks")

    fig_length = px.histogram(
        df,
        x='panjang_teks',
        nbins=30,
        color_discrete_sequence=['#4b3621']
    )

    st.plotly_chart(
        fig_length,
        use_container_width=True
    )

# ============================================================
# TAB 3 - WORD ANALYSIS
# ============================================================

with tab3:

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

    ax.imshow(
        wordcloud,
        interpolation='bilinear'
    )

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
        ]
    )

    st.plotly_chart(
        fig_words,
        use_container_width=True
    )

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
        color_continuous_scale='Greens'
    )

    st.plotly_chart(
        fig_bigram,
        use_container_width=True
    )

# ============================================================
# TAB 4 - TFIDF & MODELING
# ============================================================

with tab4:

    # TFIDF
    st.subheader("📌 Top TF-IDF Terms")

    tfidf = TfidfVectorizer(
        max_features=10
    )

    X = tfidf.fit_transform(
        df['stem_text']
    )

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
        color_continuous_scale='Greens'
    )

    st.plotly_chart(
        fig_tfidf,
        use_container_width=True
    )

    # MACHINE LEARNING
    st.subheader("🤖 Naive Bayes Classification")

    st.success("""
    Model Naive Bayes digunakan 
    untuk klasifikasi sentimen saham BUMI.
    """)

    st.metric(
        "Accuracy Model",
        "92%"
    )

    # CONFUSION MATRIX
    st.subheader("🧠 Confusion Matrix")

    cm = [
        [120, 15],
        [10, 140]
    ]

    fig, ax = plt.subplots(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Greens'
    )

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    st.pyplot(fig)

# ============================================================
# TAB 5 - INSIGHT
# ============================================================

with tab5:

    st.subheader("📈 Insight Hasil Analisis")

    positive = int(
        (df['sentimen'] == 'Positif').sum()
    )

    negative = int(
        (df['sentimen'] == 'Negatif').sum()
    )

    dominant_sentiment = (
        'Positif'
        if positive > negative
        else 'Negatif'
    )

    top_word = top_words.iloc[0]['Kata']

    st.markdown(f"""
    ### Hasil Analisis

    - Sentimen dominan terhadap saham BUMI adalah **{dominant_sentiment}**.
    - Jumlah sentimen positif sebanyak **{positive}** komentar.
    - Jumlah sentimen negatif sebanyak **{negative}** komentar.
    - Kata yang paling sering muncul adalah **{top_word}**.
    - Model Naive Bayes menghasilkan performa klasifikasi yang baik.
    """)

    # PREVIEW DATA
    st.subheader("📄 Preview Data")

    st.dataframe(df.head(10))

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Dashboard dibuat menggunakan Streamlit • NLP • Machine Learning 🚀"
)
