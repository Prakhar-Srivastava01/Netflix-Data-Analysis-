import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "29d72dec"
def search_movie_online(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data["Response"] == "True":
        return data
    return None

st.set_page_config(page_title="Netflix Data Analysis", layout="wide")

# FETCH MOVIE POSTER
# ---------------------------
def fetch_poster(title):
    api_key = "29d72dec"  
    url = f"https://www.omdbapi.com/?t={title}&apikey={api_key}"
    data = requests.get(url).json()
    
    if data["Response"] == "True":
        return data["Poster"]
    else:
        return "https://via.placeholder.com/300x450?text=No+Image"
plt.close('all')
plt.ioff()
st.title("🎬 Netflix Movies & TV Shows Data Analysis")

# GLOBAL SEARCH
# ---------------------------
search_query = st.sidebar.text_input("🔎 Search Movie / TV Show")
if search_query:
    movie = search_movie_online(search_query)

    if movie:
        st.image(movie["Poster"], width=300)
        st.subheader(movie["Title"])
        st.write("Year:", movie["Year"])
        st.write("Genre:", movie["Genre"])
        st.write("IMDB Rating:", movie["imdbRating"])
        st.write(movie["Plot"])
    else:
        st.warning("Movie not found")

def search_movie_online(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={API_KEY}"
    data = requests.get(url).json()
    if data["Response"] == "True":
        return data
    return None

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\ASUS\Desktop\Netflix-Data-Analysis\netflix_titles.csv")
    df.fillna("Not Available", inplace=True)
    df.replace("None", "Not Available", inplace=True)
    return df

df = load_data()
df.fillna("Not Available", inplace=True)
df.replace("None", "Not Available", inplace=True)

# Genre Processing
df["listed_in"] = df["listed_in"].str.split(",")
genre_df = df.explode("listed_in")
genre_df["listed_in"] = genre_df["listed_in"].str.strip()

# Recommendation System Setup
# ---------------------------

# Combine important columns
df["combined_features"] = (
    df["listed_in"].astype(str) + " " +
    df["description"].astype(str)
)

# TF-IDF Vectorization
@st.cache_resource
def create_similarity(df):
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined_features"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

cosine_sim = create_similarity(df)

# Title index mapping
indices = pd.Series(df.index, index=df["title"]).drop_duplicates()
def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return []

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:6]  

    movie_indices = [i[0] for i in sim_scores]

    return df["title"].iloc[movie_indices]

# Director Processing
# ---------------------------
df["director"] = df["director"].str.split(",")

director_df = df.explode("director")
director_df["director"] = director_df["director"].str.strip()

# Load dataset

st.subheader("Dataset Preview")
st.write(df.head())
st.sidebar.title("Netflix Filters")
type_option = st.sidebar.selectbox(
    "Select Type",
    df["type"].unique()
)
country_option = st.sidebar.selectbox(
    "Select Country",
    ["All"] + sorted(df["country"].unique().tolist())
)
genre_option = st.sidebar.selectbox(
    "Select Genre",
    ["All"] + sorted(genre_df["listed_in"].unique().tolist())
)

filtered_df = df[df["type"] == type_option]
if country_option != "All":
    filtered_df = filtered_df[filtered_df["country"] == country_option]

year_range = st.sidebar.slider(
    "Release Year",
    int(df["release_year"].min()),
    int(df["release_year"].max()),
    (2000, 2021)
)

# GLOBAL MOVIE SEARCH
search_query = st.sidebar.text_input(
    "🔎 Search Movie / TV Show",
    key="global_search"
)
# Apply search filter to dataset
if search_query:
    filtered_df = filtered_df[
        filtered_df["title"].str.contains(search_query, case=False, na=False)
    ]
st.sidebar.markdown("🎯 Get Recommendations")
selected_title = st.sidebar.selectbox(
    "Choose a Title",
    sorted(df["title"].unique())
)
# Search Title
search_title = st.sidebar.text_input("Search Title")
filtered_df = filtered_df[
    (filtered_df["release_year"] >= year_range[0]) &
    (filtered_df["release_year"] <= year_range[1])
]
if search_title:
    filtered_df = filtered_df[
        filtered_df["title"].str.contains(search_title, case=False)
    ]

# ---------------------------
# KPI CARDS
# ---------------------------
st.markdown("## 📊 Netflix Overview")

col1, col2, col3, col4 = st.columns(4)

total_titles = len(filtered_df)
total_movies = len(filtered_df[filtered_df["type"] == "Movie"])
total_tv = len(filtered_df[filtered_df["type"] == "TV Show"])
latest_year = filtered_df["release_year"].max()
col1.metric("Total Titles", total_titles)
col2.metric("Movies", total_movies)
col3.metric("TV Shows", total_tv)
col4.metric("Latest Release Year", latest_year)


# Apply Genre Filter
if genre_option != "All":
    genre_filtered_ids = genre_df[
        genre_df["listed_in"] == genre_option
    ]["show_id"]

    filtered_df = filtered_df[
        filtered_df["show_id"].isin(genre_filtered_ids)
    ]

if filtered_df.empty:
    st.warning("⚠️ No data found for selected filters. Try changing Type or Year.")

# Movies vs TV Shows Chart 

st.subheader("📊 Movies vs TV Shows")

type_count =df["type"].value_counts().reset_index()
type_count.columns = ["Type", "Count"]

fig = px.bar(
    type_count,
    x="Type",
    y="Count",
    text="Count",
    title="Movies vs TV Shows Distribution"
)
st.plotly_chart(fig, use_container_width=True)

# Rating Distribution 
st.subheader("⭐ Rating Distribution")
rating_count = (
    filtered_df["rating"]
    .value_counts()
    .head(10)
    .reset_index()
)
rating_count.columns = ["Rating", "Count"]

fig = px.bar(
    rating_count,
    x="Rating",
    y="Count",
    text="Count",
    title="Top Ratings on Netflix"
)

st.plotly_chart(fig, use_container_width=True)

# Top Directors
# ---------------------------
st.subheader("🎬 Top 10 Directors on Netflix")
top_directors = (
    director_df["director"]
    .value_counts()
    .head(10)
    .reset_index()
)

top_directors.columns = ["Director", "Count"]

fig = px.bar(
    top_directors,
    x="Director",
    y="Count",
    text="Count",
    title="Most Frequent Directors"
)
st.plotly_chart(fig, use_container_width=True)

# Content Type Distribution
st.subheader("📺 Content Type Distribution")
content_type = filtered_df["type"].value_counts().reset_index()
content_type.columns = ["Type", "Count"]
fig = px.pie(
    content_type,
    names="Type",
    values="Count",
    title="Movies vs TV Shows Share",
    hole=0.4   # donut chart look
)
st.plotly_chart(fig, use_container_width=True)
# Top 10 Countries
st.subheader("🌍 Top 10 Countries")

country_count = (
    df["country"]
    .value_counts()
    .head(10)
    .reset_index()
)
country_count.columns = ["Country", "Count"]
fig = px.bar(
    country_count,
    x="Country",
    y="Count",
    text="Count",
    title="Top 10 Countries Producing Netflix Content"
)

st.plotly_chart(fig, use_container_width=True)

# Top Genres

st.subheader("🎭 Top 10 Netflix Genres")
top_genres = (
    genre_df["listed_in"]
    .value_counts()
    .head(10)
    .reset_index()
)

top_genres.columns = ["Genre", "Count"]
fig = px.bar(
    top_genres,
    x="Genre",
    y="Count",
    text="Count",
    title="Most Popular Genres on Netflix"
)
st.plotly_chart(fig, use_container_width=True)

# Content Release Year 
st.subheader("📅 Content Release Trend")
year_data = (
    filtered_df["release_year"]
    .value_counts()
    .sort_index()
    .reset_index()
)
year_data.columns = ["Year", "Count"]
fig = px.line(
    year_data,
    x="Year",
    y="Count",
    markers=True,
    title="Netflix Content Growth Over Time"
)

st.plotly_chart(fig, use_container_width=True)
st.subheader("⭐ Overall Ratings Distribution")
overall_rating = (
    df["rating"]
    .value_counts()
    .head(10)
    .reset_index()
)
overall_rating.columns = ["Rating", "Count"]
fig = px.bar(
    overall_rating,
    x="Rating",
    y="Count",
    text="Count",
    title="Overall Netflix Ratings"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔎 Filtered Results")

st.dataframe(
    filtered_df[["title", "type", "country", "release_year", "rating"]],
    use_container_width=True
)
# ---------------------------
# Recommendations Output
# ---------------------------

st.subheader("🎬 Recommended For You")
st.info(f"Because you watched: **{selected_title}**")

with st.spinner("Finding best recommendations..."):
    recommendations = get_recommendations(selected_title)
recommendations = get_recommendations(selected_title)

if len(recommendations) > 0:
    cols = st.columns(5)

for i, movie in enumerate(recommendations):
    poster_url = fetch_poster(movie)
    with cols[i % 5]:
        st.image(poster_url, use_container_width=True)
        st.caption(movie)

else:
    st.write("No recommendations found.")
    st.write("Selected:", selected_title)
st.success("Analysis Completed Successfully ✅")

