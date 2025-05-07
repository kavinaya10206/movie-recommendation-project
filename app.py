import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

@st.cache_data
def load_data():
    return pd.read_csv("movies.csv")

data = load_data()

def build_model(data):
    tfidf = TfidfVectorizer(stop_words='english')
    data['description'] = data['description'].fillna('')
    tfidf_matrix = tfidf.fit_transform(data['description'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(data.index, index=data['title']).drop_duplicates()
    return cosine_sim, indices

cosine_sim, indices = build_model(data)

def get_recommendations(title, cosine_sim, indices, data, top_n=5):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return data.iloc[movie_indices][['title', 'genres', 'rating', 'description']]

st.set_page_config(page_title="CineMatch 🎬", layout="centered")
st.title("Welcome to CineMatch 🎬")
st.subheader("Your AI-powered personal movie recommender!")

genre = st.multiselect("Choose Genres:", options=list(set(", ".join(data['genres']).split(", "))))
rating = st.slider("Minimum Rating:", min_value=0.0, max_value=10.0, step=0.1, value=7.0)
mood_keyword = st.text_input("What mood are you in? (e.g., feel-good, thrilling, romantic)")

filtered_data = data[data['rating'] >= rating]
if genre:
    filtered_data = filtered_data[filtered_data['genres'].apply(lambda x: any(g.strip() in x for g in genre))]
if mood_keyword:
    filtered_data = filtered_data[filtered_data['description'].str.contains(mood_keyword, case=False, na=False)]

if not filtered_data.empty:
    st.write("### Top Movie Matches for You")
    top_titles = filtered_data.head(10)['title'].tolist()
    for title in top_titles:
        recs = get_recommendations(title, cosine_sim, indices, data, top_n=1)
        for _, row in recs.iterrows():
            st.markdown(f"**{row['title']}** ({row['rating']})")
            st.write(f"*Genres:* {row['genres']}")
            st.write(f"{row['description']}")
            st.markdown("---")
else:
    st.warning("No recommendations found. Try adjusting your filters.")
