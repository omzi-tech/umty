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
from pathlib import Path
from urllib.request import urlretrieve
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------

TMDB_API_KEY = "f2ad154015d6abcfc9e32b86726dca5a"
BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "ml-latest-small",
    BASE_DIR.parent / "ml-latest-small",
    Path.cwd() / "ml-latest-small",
]
ML_DATA_PATH = next((p for p in DATA_CANDIDATES if p.exists()), BASE_DIR / "ml-latest-small")
MOVIELENS_ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DB_PATH = BASE_DIR / "umty_users.db"

st.set_page_config(page_title="umty — Movies", page_icon=":film_frames:", layout="wide")

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
        created_at TEXT,
        role TEXT DEFAULT 'user')""")

    c.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in c.fetchall()]
    if "role" not in columns:
        c.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")

    c.execute("""CREATE TABLE IF NOT EXISTS favorites(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        title TEXT,
        added_at TEXT)""")

    admin_username = "Admin99"
    admin_password_hash = hash_pw("Admin99")
    c.execute(
        """INSERT INTO users(name,email,username,password_hash,created_at,role)
           VALUES (?,?,?,?,?, 'admin')
           ON CONFLICT(username) DO UPDATE SET
               name=excluded.name,
               email=excluded.email,
               password_hash=excluded.password_hash,
               role='admin'""",
        ("Administrator", "admin@example.com", admin_username, admin_password_hash, datetime.utcnow().isoformat()),
    )
    c.execute(
        "UPDATE users SET name='Administrator', email='admin@example.com', password_hash=?, role='admin' WHERE username=?",
        (admin_password_hash, admin_username),
    )
    conn.commit()
    conn.close()

setup_db()

def create_user(name,email,username,password):
    try:
        conn = db_conn()
        c = conn.cursor()
        c.execute("INSERT INTO users(name,email,username,password_hash,created_at,role) VALUES (?,?,?,?,?,?)",
                  (name,email,username,hash_pw(password),datetime.utcnow().isoformat(),"user"))
        conn.commit()
        conn.close()
        return True, ""
    except:
        return False, "Username already exists"

def login(username,password):
    conn = db_conn()
    c = conn.cursor()
    c.execute("SELECT id,name,email,password_hash,COALESCE(role,'user') FROM users WHERE username=?",(username,))
    row = c.fetchone()
    conn.close()
    if not row: return False,"No such user"
    uid,name,email,pwh,role = row
    if pwh != hash_pw(password): return False,"Wrong password"
    return True, {"id":uid,"name":name,"email":email,"username":username,"role":role}

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

def get_all_users():
    conn = db_conn()
    rows = pd.read_sql_query(
        """
        SELECT u.username, u.role, COUNT(f.id) AS favorites
        FROM users u
        LEFT JOIN favorites f ON f.user_id = u.id
        GROUP BY u.id, u.username, u.role
        ORDER BY u.username
        """,
        conn,
    )
    conn.close()
    return rows

def remove_favorite(uid,title):
    conn = db_conn()
    c = conn.cursor()
    c.execute("DELETE FROM favorites WHERE user_id=? AND title=?",(uid,title))
    conn.commit(); conn.close()


# -----------------------------------------------------------
# DATASET LOADING
# -----------------------------------------------------------

def ensure_dataset():
    ML_DATA_PATH.mkdir(parents=True, exist_ok=True)
    movies_file = ML_DATA_PATH / "movies.csv"
    ratings_file = ML_DATA_PATH / "ratings.csv"

    if not movies_file.exists():
        sample_movies = """movieId,title,genres,overview
1,The Matrix,Action|Sci-Fi,Reality is a simulation.
2,Inception,Action|Sci-Fi,Thief enters dreams.
3,Pulp Fiction,Crime|Drama,Stories interconnect.
4,Avatar,Action|Fantasy,Marine on Pandora.
5,Fight Club,Drama,Secret fight club.
"""
        movies_file.write_text(sample_movies)

    if not ratings_file.exists():
        sample_ratings = """userId,movieId,rating,timestamp
1,1,5,100
1,2,4,100
2,1,4,100
2,3,5,100
3,4,5,100
"""
        ratings_file.write_text(sample_ratings)

ensure_dataset()

@st.cache_data(show_spinner=False)
def load_data():
    movies = pd.read_csv(ML_DATA_PATH / "movies.csv")
    ratings = pd.read_csv(ML_DATA_PATH / "ratings.csv")
    return movies, ratings

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
    movie_frame = movies.copy()
    if "overview" not in movie_frame.columns:
        movie_frame["overview"] = ""
    if "genres" not in movie_frame.columns:
        movie_frame["genres"] = ""
    if "title" not in movie_frame.columns:
        raise ValueError("Movie dataset must include a title column.")

    movie_frame["overview"] = movie_frame["overview"].fillna("")
    movie_frame["genres"] = movie_frame["genres"].fillna("")
    movie_frame["content"] = (
        movie_frame["title"].fillna("").astype(str)
        + " "
        + movie_frame["genres"].fillna("").astype(str)
        + " "
        + movie_frame["overview"].fillna("").astype(str)
    )
    tfidf = TfidfVectorizer(stop_words="english")
    mat = tfidf.fit_transform(movie_frame["content"])
    sim = cosine_similarity(mat)
    return sim, movie_frame["title"].tolist()

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

def slugify(value):
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()[:40] or "movie"


def render_row(titles, section="main"):
    if not titles:
        st.caption("No movies to show right now.")
        return

    cols = st.columns(min(5, max(1, len(titles))))
    for i, title in enumerate(titles):
        with cols[i % len(cols)]:
            poster = fetch_poster(title)
            if poster:
                st.image(poster, width=170)
            else:
                st.markdown(f"<div style='height:225px; background:#1f1f1f; border-radius:12px; display:flex; align-items:center; justify-content:center; color:#888;'>No poster</div>", unsafe_allow_html=True)

            st.caption(title)
            button_key = f"save_{section}_{slugify(title)}_{i}"
            if st.button("Save", key=button_key):
                if not st.session_state.user:
                    st.session_state.pending_favorite = title
                    st.info("Please log in to save favorites.")
                else:
                    add_favorite(st.session_state.user["id"], title)
                    st.success(f"Saved: {title}")


def render_admin_dashboard():
    users = get_all_users()
    if users.empty:
        st.caption("No users yet.")
        return

    st.subheader("Admin dashboard", divider="gray")
    admin_cols = st.columns(3)
    with admin_cols[0]:
        st.metric("Total users", int(len(users)))
    with admin_cols[1]:
        st.metric("Admins", int((users["role"] == "admin").sum()))
    with admin_cols[2]:
        st.metric("Saved titles", int(users["favorites"].sum()))

    st.dataframe(
        users.rename(columns={"username": "Username", "role": "Role", "favorites": "Favorites"}),
        use_container_width=True,
        hide_index=True,
    )


# -----------------------------------------------------------
# SESSION STATE
# -----------------------------------------------------------

if "user" not in st.session_state:
    st.session_state.user = None

if "modal_title" not in st.session_state:
    st.session_state.modal_title = None

if "pending_favorite" not in st.session_state:
    st.session_state.pending_favorite = None

if not st.session_state.user:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style='background:linear-gradient(135deg,#111111 0%,#1a1a1a 45%,#0f172a 100%); padding:2.5rem; border-radius:20px; border:1px solid #2a2a2a; margin-bottom:1.5rem;'>
            <div style='font-size:2.8rem; font-weight:800; color:#e50914; letter-spacing:0.04em;'>umty</div>
            <div style='font-size:1.1rem; color:#d1d5db; margin-top:0.6rem;'>Your next movie night starts here.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    login_tab, signup_tab = st.tabs(["Sign in", "Create account"])

    with login_tab:
        with st.form("landing_login_form"):
            username = st.text_input("Username", key="landing_username")
            password = st.text_input("Password", type="password", key="landing_password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)
            if submitted:
                ok, res = login(username, password)
                if ok:
                    st.session_state.user = res
                    st.rerun()
                else:
                    st.error(res)

    with signup_tab:
        with st.form("landing_signup_form"):
            name = st.text_input("Full name", key="landing_name")
            email = st.text_input("Email", key="landing_email")
            new_username = st.text_input("Username", key="landing_new_username")
            new_password = st.text_input("Password", type="password", key="landing_new_password")
            confirm_password = st.text_input("Confirm password", type="password", key="landing_confirm_password")
            signup = st.form_submit_button("Create account", use_container_width=True)
            if signup:
                if not all([name, email, new_username, new_password, confirm_password]):
                    st.error("Please fill in every field.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = create_user(name, email, new_username, new_password)
                    if ok:
                        st.success("Account created successfully. You can sign in now.")
                    else:
                        st.error(msg)

    st.stop()

# -----------------------------------------------------------
# SIDEBAR AUTH
# -----------------------------------------------------------

with st.sidebar:
    st.markdown("## :material/person: Account")

    if st.session_state.user:
        st.success(f"Signed in as {st.session_state.user['username']}")
        st.caption(st.session_state.user['name'])

        if st.session_state.user.get("role") == "admin":
            st.badge("Administrator", icon=":material/shield:", color="red")
            st.markdown("### Admin controls")
            st.write("Access to the admin dashboard is active.")
            st.metric("Users", len(pd.read_sql_query("SELECT id FROM users", sqlite3.connect(DB_PATH))))
            if st.button("Refresh admin panel", type="secondary"):
                st.rerun()
        else:
            st.badge("User", icon=":material/person:", color="blue")

        if st.button("Log out", type="secondary"):
            st.session_state.user = None
            st.session_state.pending_favorite = None
            st.rerun()
    else:
        st.subheader("Log in")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in")
            if submitted:
                ok, res = login(username, password)
                if ok:
                    st.session_state.user = res
                    st.session_state.modal_title = None
                    if st.session_state.pending_favorite:
                        add_favorite(res["id"], st.session_state.pending_favorite)
                        st.toast(f"Saved {st.session_state.pending_favorite} to favorites.")
                        st.session_state.pending_favorite = None
                    st.rerun()
                else:
                    st.error(res)

        st.subheader("Create account")
        with st.form("signup_form"):
            name = st.text_input("Full name")
            email = st.text_input("Email")
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            signup = st.form_submit_button("Create account")
            if signup:
                if not all([name, email, new_username, new_password, confirm_password]):
                    st.error("Please fill in every field.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = create_user(name, email, new_username, new_password)
                    if ok:
                        st.success("Account created successfully.")
                    else:
                        st.error(msg)

# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------

if st.session_state.user:
    welcome_name = st.session_state.user["name"] or st.session_state.user["username"]
    st.title(f"Welcome back, {welcome_name}", anchor=False)
else:
    st.title("umty", anchor=False)

st.caption("Trending movies, personal picks, and favorites built for your next watch night.")

with st.container(border=True):
    hero_cols = st.columns([1.4, 1])
    with hero_cols[0]:
        st.markdown("### Now streaming")
        st.markdown("Discover your next favorite film, build a watchlist, and let your saved picks power smarter recommendations.")
        if st.session_state.user:
            st.markdown("Your account is ready. Pick a mood, save your favorites, and keep the night rolling.")
        else:
            st.markdown("Sign in to unlock personalized suggestions, favorites, and a cleaner movie night experience.")
    with hero_cols[1]:
        st.markdown("#### Quick stats")
        stats_cols = st.columns(3)
        with stats_cols[0]:
            st.metric("Movies", len(movies_df))
        with stats_cols[1]:
            st.metric("Users", ratings_df["userId"].nunique())
        with stats_cols[2]:
            st.metric("Top rated", int(ratings_df["rating"].max()))

if st.session_state.user and st.session_state.user.get("role") == "admin":
    render_admin_dashboard()

categories = ["Trending", "Action", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Horror", "Mystery", "Romance"]
selected_category = st.pills("Browse by genre", categories, selection_mode="single", default="Trending")

if selected_category == "Trending":
    pop = ratings_df.groupby("movieId")["rating"].count().sort_values(ascending=False)
    ids = pop.head(20).index.tolist()
    titles = movies_df[movies_df["movieId"].isin(ids)]["title"].tolist()
else:
    mask = movies_df["genres"].str.contains(selected_category, case=False, na=False)
    titles = movies_df[mask]["title"].tolist()[:25]

st.subheader(f"{selected_category} picks", divider="gray")
render_row(titles, section="category")

# -----------------------------------------------------------
# AI RECOMMENDATIONS
# -----------------------------------------------------------

if st.session_state.user:
    st.subheader("AI recommendations", divider="gray")
    favs = get_favorites(st.session_state.user["id"])
    if favs:
        rec = []
        for f in favs:
            rec += similar(f)
        rec = list(dict.fromkeys(rec))
        rec = [r for r in rec if r not in favs]
        render_row(rec[:20], section="recommendations")
    else:
        st.caption("Add a few favorites to unlock personalized recommendations.")
else:
    st.subheader("Favorites", divider="gray")
    st.caption("Log in to save your watchlist and see recommendations tailored to your taste.")

if st.session_state.user:
    st.subheader("Your favorites", divider="gray")
    favs = get_favorites(st.session_state.user["id"])
    if favs:
        render_row(favs, section="favorites")
    else:
        st.caption("No favorites saved yet.")

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