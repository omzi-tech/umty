# -----------------------------------------------------------
# umty — Full Netflix-Style Movie App (Final Version)
# -----------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import zipfile
import sqlite3
import hashlib
import difflib
import re
from urllib.request import urlretrieve
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------

TMDB_API_KEY = "f2ad154015d6abcfc9e32b86726dca5a"
ML_DATA_PATH = "ml-latest-small"
MOVIELENS_ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DB_PATH = "umty_users.db"

st.set_page_config(page_title="umty — Movies", layout="wide")

# -----------------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------------

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.stApp {background: #0f0f0f;}

.section-title {
    color: white;
    font-size: 28px;
    font-weight: 700;
    margin: 20px 0 10px 0;
}

.movie-poster {
    width: 150px;
    height: 225px;
    object-fit: cover;
    border-radius: 8px;
    transition: 0.25s;
}
.movie-poster:hover {transform: scale(1.07);}

.movie-title {
    color: #ddd;
    font-size: 13px;
    margin-top: 6px;
    text-align: center;
    max-width: 150px;
    overflow: hidden;
    text-overflow: ellipsis;
}

.no-poster {
    width:150px;
    height:225px;
    background:#333;
    border-radius:8px;
    display:flex;
    justify-content:center;
    align-items:center;
    color:#999;
}

.top-nav {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:10px 0;
    border-bottom:1px solid #333;
    margin-bottom:10px;
}

.nav-title {
    font-size:30px;
    color:#e50914;
    font-weight:900;
}

.category-select {
    width:350px;
    color:white !important;
}

.user-btn {
    color:white;
}

.poster-row {
    display:flex;
    gap:20px;
    overflow-x:auto;
    padding:10px 0;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------
# DATABASE
# -----------------------------------------------------------

def db_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def hash_pw(p):
    return hashlib.sha256(p.encode()).hexdigest()

def setup_db():
    conn = db_conn()
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        username TEXT UNIQUE,
        password_hash TEXT,
        created_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS favorites(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        title TEXT,
        added_at TEXT)""")
    conn.commit()

setup_db()

def create_user(name,email,username,password):
    try:
        conn = db_conn()
        c = conn.cursor()
        c.execute("INSERT INTO users(name,email,username,password_hash,created_at) VALUES (?,?,?,?,?)",
                  (name,email,username,hash_pw(password),datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        return True,""
    except:
        return False, "Username already exists"

def login(username,password):
    conn = db_conn()
    c = conn.cursor()
    c.execute("SELECT id,name,email,password_hash FROM users WHERE username=?",(username,))
    row = c.fetchone()
    conn.close()
    if not row: return False,"No such user"
    uid,name,email,pwh = row
    if pwh != hash_pw(password): return False,"Wrong password"
    return True, {"id":uid,"name":name,"email":email,"username":username}

def add_favorite(uid,title):
    conn = db_conn()
    c = conn.cursor()
    c.execute("INSERT INTO favorites(user_id,title,added_at) VALUES (?,?,?)",
              (uid,title,datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def get_favorites(uid):
    conn = db_conn()
    c = conn.cursor()
    c.execute("SELECT title FROM favorites WHERE user_id=?",(uid,))
    rows = c.fetchall(); conn.close()
    return [r[0] for r in rows]

def remove_favorite(uid,title):
    conn = db_conn()
    c = conn.cursor()
    c.execute("DELETE FROM favorites WHERE user_id=? AND title=?",(uid,title))
    conn.commit(); conn.close()


# -----------------------------------------------------------
# DATASET LOADING
# -----------------------------------------------------------

def ensure_dataset():
    os.makedirs(ML_DATA_PATH, exist_ok=True)
    if not os.path.exists(f"{ML_DATA_PATH}/movies.csv"):
        # create small sample set
        sample_movies = """movieId,title,genres,overview
1,The Matrix,Action|Sci-Fi,Reality is a simulation.
2,Inception,Action|Sci-Fi,Thief enters dreams.
3,Pulp Fiction,Crime|Drama,Stories interconnect.
4,Avatar,Action|Fantasy,Marine on Pandora.
5,Fight Club,Drama,Secret fight club.
"""
        with open(f"{ML_DATA_PATH}/movies.csv","w") as f: f.write(sample_movies)

    if not os.path.exists(f"{ML_DATA_PATH}/ratings.csv"):
        sample_ratings = """userId,movieId,rating,timestamp
1,1,5,100
1,2,4,100
2,1,4,100
2,3,5,100
3,4,5,100
"""
        with open(f"{ML_DATA_PATH}/ratings.csv","w") as f: f.write(sample_ratings)

ensure_dataset()

@st.cache_data(show_spinner=False)
def load_data():
    movies = pd.read_csv(f"{ML_DATA_PATH}/movies.csv")
    ratings = pd.read_csv(f"{ML_DATA_PATH}/ratings.csv")
    return movies,ratings

movies_df, ratings_df = load_data()

# -----------------------------------------------------------
# TMDB POSTER
# -----------------------------------------------------------

def clean_title(t):
    t = re.sub(r"\(\d{4}\)$","",t).strip()
    return t

@st.cache_data(show_spinner=False)
def fetch_poster(title):
    try:
        query = clean_title(title)
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={query}"
        r = requests.get(url).json()
        if r["results"]:
            p = r["results"][0]["poster_path"]
            if p:
                return f"https://image.tmdb.org/t/p/w342{p}"
    except:
        return None
    return None

# -----------------------------------------------------------
# RECOMMENDER (Simple Content Similarity)
# -----------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_content_matrix(movies):
    movies["overview"] = movies["overview"].fillna("")
    movies["genres"] = movies["genres"].fillna("")
    movies["content"] = movies["title"]+" "+movies["genres"]+" "+movies["overview"]
    tfidf = TfidfVectorizer(stop_words="english")
    mat = tfidf.fit_transform(movies["content"])
    sim = cosine_similarity(mat)
    return sim, movies["title"].tolist()

content_sim, title_list = build_content_matrix(movies_df)

def similar(title):
    if title not in title_list: return []
    idx = title_list.index(title)
    scores = list(enumerate(content_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:8]
    return [title_list[i] for i,_ in scores]

# -----------------------------------------------------------
# UI HELPERS
# -----------------------------------------------------------

def render_row(titles):
    html = '<div class="poster-row">'
    for t in titles:
        poster = fetch_poster(t)
        if poster:
            html += f"""
            <div>
                <img src="{poster}" class="movie-poster">
                <div class="movie-title">{t}</div>
            </div>
            """
        else:
            html += f"""
            <div>
                <div class="no-poster">No Image</div>
                <div class="movie-title">{t}</div>
            </div>
            """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# -----------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------

if "user" not in st.session_state:
    st.session_state.user = None

if "modal_title" not in st.session_state:
    st.session_state.modal_title = None

# -----------------------------------------------------------
# TOP NAV BAR
# -----------------------------------------------------------

st.markdown('<div class="top-nav">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,2,1])

with col1:
    st.markdown('<div class="nav-title">umty</div>', unsafe_allow_html=True)

with col2:
    categories = ["Trending","Action","Comedy","Crime","Documentary","Drama","Fantasy",
                  "Horror","Mystery","Romance"]
    category = st.selectbox(
        "Select Category",
        categories,
        label_visibility="collapsed",
        key="cat"
    )

with col3:
    if st.session_state.user:
        if st.button("Logout"):
            st.session_state.user = None
            st.rerun()
    else:
        st.caption("Not logged in")

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------------------
# HOME / CATEGORY DISPLAY
# -----------------------------------------------------------

st.markdown(f'<div class="section-title">{category}</div>', unsafe_allow_html=True)

if category == "Trending":
    # show popular movies = top rated count
    pop = ratings_df.groupby("movieId")["rating"].count().sort_values(ascending=False)
    ids = pop.head(20).index.tolist()
    titles = movies_df[movies_df["movieId"].isin(ids)]["title"].tolist()
    render_row(titles)
else:
    mask = movies_df["genres"].str.contains(category, case=False, na=False)
    titles = movies_df[mask]["title"].tolist()
    if titles:
        render_row(titles[:25])
    else:
        st.info("No movies found in this category.")

# -----------------------------------------------------------
# FAVORITES BUTTON BELOW MOVIES
# -----------------------------------------------------------

st.markdown('<div class="section-title">Add to Favorites</div>', unsafe_allow_html=True)

cols = st.columns(6)
if titles:
    for i,t in enumerate(titles[:12]):
        with cols[i % 6]:
            if st.button(f"❤ {t}", key=f"favbtn_{t}"):
                if not st.session_state.user:
                    st.session_state.modal_title = t
                    st.rerun()
                else:
                    add_favorite(st.session_state.user["id"], t)
                    st.success(f"Added {t}")

# -----------------------------------------------------------
# AI RECOMMENDATIONS
# -----------------------------------------------------------

if st.session_state.user:
    st.markdown('<div class="section-title">🤖 AI Recommendations</div>', unsafe_allow_html=True)

    favs = get_favorites(st.session_state.user["id"])
    if favs:
        rec = []
        for f in favs:
            rec += similar(f)
        # remove duplicates
        rec = list(dict.fromkeys(rec))
        rec = [r for r in rec if r not in favs]
        render_row(rec[:20])
    else:
        st.info("Add favorites to receive AI recommendations.")

# -----------------------------------------------------------
# FAVORITES PAGE BUTTON
# -----------------------------------------------------------

st.markdown('<div class="section-title">❤ Your Favorites</div>', unsafe_allow_html=True)

if st.session_state.user:
    favs = get_favorites(st.session_state.user["id"])
    if favs:
        render_row(favs)
    else:
        st.info("No favorites yet.")
else:
    st.info("Login to see favorites.")

# -----------------------------------------------------------
# LOGIN MODAL (ONLY WHEN FAVORITING)
# -----------------------------------------------------------

if st.session_state.modal_title:

    st.markdown("""
    <div style="
        position:fixed;
        top:0; left:0;
        width:100%; height:100%;
        background:rgba(0,0,0,0.85);
        z-index:999999;
        display:flex;
        justify-content:center;
        align-items:center;
    ">
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("""
        <div style="background:#1e1e1e;padding:30px;border-radius:10px;width:380px;">
        """, unsafe_allow_html=True)

        st.markdown(f"### Sign in to save *{st.session_state.modal_title}*")

        tab1, tab2 = st.tabs(["Sign In", "Sign Up"])

        with tab1:
            u = st.text_input("Username", key="login_u")
            p = st.text_input("Password", type="password", key="login_p")
            if st.button("Sign In"):
                ok,res = login(u,p)
                if ok:
                    st.session_state.user = res
                    add_favorite(res["id"], st.session_state.modal_title)
                    st.session_state.modal_title = None
                    st.rerun()
                else:
                    st.error(res)
            if st.button("Cancel"):
                st.session_state.modal_title = None
                st.rerun()

        with tab2:
            n = st.text_input("Full Name", key="reg_n")
            e = st.text_input("Email", key="reg_e")
            uu = st.text_input("Username", key="reg_u")
            pp = st.text_input("Password", type="password", key="reg_pw")
            pp2 = st.text_input("Confirm Password", type="password", key="reg_pw2")

            if st.button("Create Account"):
                if not all([n,e,uu,pp,pp2]):
                    st.error("Fill all fields")
                elif pp != pp2:
                    st.error("Passwords don't match")
                else:
                    ok,msg = create_user(n,e,uu,pp)
                    if ok:
                        st.success("Account created!")
                    else:
                        st.error(msg)

            if st.button("Cancel", key="reg_cancel"):
                st.session_state.modal_title = None
                st.rerun()

        st.markdown("</div></div>", unsafe_allow_html=True)