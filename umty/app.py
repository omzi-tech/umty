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
from datetime import datetime, timezone

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

.stApp {
    background: radial-gradient(circle at top, #1a1a1a 0%, #0b0b0b 38%, #050505 100%);
    color: #f5f5f5;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: #111111;
    border-right: 1px solid rgba(255,255,255,0.08);
}

.section-title {
    color: white;
    font-size: 28px;
    font-weight: 700;
    margin: 20px 0 10px 0;
}

.hero-panel {
    background: linear-gradient(135deg, rgba(229,9,20,0.18), rgba(15,23,42,0.7));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 1.5rem 1.5rem 1rem 1.5rem;
    box-shadow: 0 20px 50px rgba(0,0,0,0.25);
}

.premium-shell {
    background: linear-gradient(135deg, rgba(17,17,17,0.96), rgba(24,24,27,0.88));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 28px;
    padding: 1.2rem;
    box-shadow: 0 26px 60px rgba(0,0,0,0.35);
    margin-bottom: 1rem;
}

.premium-badge {
    display: inline-block;
    padding: 0.34rem 0.7rem;
    border-radius: 999px;
    background: rgba(229,9,20,0.18);
    border: 1px solid rgba(229,9,20,0.45);
    color: #fca5a5;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.premium-title {
    margin-top: 0.8rem;
    font-size: clamp(2.1rem, 4vw, 4rem);
    line-height: 1.02;
    font-weight: 900;
    color: #fff;
    letter-spacing: -0.05em;
}

.premium-subtitle {
    color: #e5e7eb;
    font-size: 1rem;
    line-height: 1.7;
    max-width: 48rem;
    margin-top: 0.8rem;
}

.premium-feature-grid {
    margin-top: 1rem;
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.8rem;
}

.premium-feature {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 0.9rem 0.8rem;
}

.premium-feature .label {
    color: #9ca3af;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.premium-feature .value {
    margin-top: 0.3rem;
    color: white;
    font-size: 1.2rem;
    font-weight: 700;
}

.auth-shell {
    max-width: 640px;
    margin: 1rem auto 0 auto;
    background: rgba(17,17,17,0.9);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 1.5rem;
    box-shadow: 0 18px 45px rgba(0,0,0,0.28);
}

.auth-shell .stTabs [role="tablist"] button {
    color: #d1d5db;
    font-weight: 600;
}

.auth-shell .stTabs [role="tablist"] button[aria-selected="true"] {
    background: rgba(229,9,20,0.18);
    color: white;
    border-bottom: 2px solid #e50914;
}

.movie-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 0.8rem 0.6rem 0.75rem 0.6rem;
    height: 100%;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.movie-card:hover {
    transform: translateY(-2px);
    border-color: rgba(229,9,20,0.4);
}

.movie-card img {
    border-radius: 12px;
    width: 100%;
    max-width: 170px;
    height: 240px;
    object-fit: cover;
    display: block;
    margin: 0 auto;
}

.movie-title {
    color: #f3f4f6;
    font-size: 0.82rem;
    margin-top: 0.75rem;
    min-height: 2.2em;
    text-align: center;
    font-weight: 600;
    line-height: 1.3;
}

.no-poster {
    width: 100%;
    max-width: 170px;
    height: 240px;
    background: linear-gradient(160deg, #1d1d1d, #0f0f0f);
    border-radius: 12px;
    display: flex;
    justify-content: center;
    align-items: center;
    color: #9ca3af;
    margin: 0 auto;
    text-align: center;
    font-size: 0.8rem;
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

.stButton > button {
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.04);
    color: white;
    font-weight: 600;
}

.stButton > button:hover {
    border-color: rgba(229,9,20,0.45);
    background: rgba(229,9,20,0.1);
}

.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #e50914, #b70010);
    border: none;
}

.stMetric {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 0.6rem 0.8rem;
}

.stMetric > div {
    color: white;
}

.sidebar-card {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.catalog-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 1.1rem 0 0.8rem 0;
}

.catalog-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: white;
}

.catalog-badge {
    background: rgba(229,9,20,0.18);
    border: 1px solid rgba(229,9,20,0.4);
    color: #fca5a5;
    border-radius: 999px;
    padding: 0.28rem 0.55rem;
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.stream-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-bottom: 0.8rem;
}

.stream-pill {
    padding: 0.5rem 0.8rem;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.02);
    color: #e5e7eb;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
}

.stream-pill.active {
    background: linear-gradient(135deg, #e50914, #b70010);
    border-color: rgba(229,9,20,0.7);
    color: white;
}

.catalog-grid {
    margin-top: 0.4rem;
}

.dashboard-shell {
    background: linear-gradient(180deg, rgba(17,17,17,0.96), rgba(15,23,42,0.7));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    padding: 1.1rem 1.1rem 1.2rem 1.1rem;
    box-shadow: 0 22px 50px rgba(0,0,0,0.2);
    margin-bottom: 1.2rem;
}

.dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
}

.dashboard-kicker {
    color: #fca5a5;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700;
}

.dashboard-header h3 {
    margin: 0.2rem 0 0 0;
    font-size: 1.7rem;
    color: white;
    font-weight: 800;
}

.dashboard-chip {
    background: rgba(34,197,94,0.14);
    border: 1px solid rgba(34,197,94,0.35);
    color: #bbf7d0;
    border-radius: 999px;
    padding: 0.3rem 0.6rem;
    font-size: 0.7rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 700;
}

.dashboard-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.8rem;
    margin-bottom: 1rem;
}

.stat-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 0.9rem 0.8rem;
}

.stat-label {
    display: block;
    color: #9ca3af;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.stat-value {
    display: block;
    margin-top: 0.35rem;
    color: white;
    font-size: 1.8rem;
    font-weight: 800;
}

.table-panel {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 0.6rem;
}

.section-shell {
    background: linear-gradient(180deg, rgba(17,17,17,0.9), rgba(24,24,27,0.72));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 1rem 1rem 0.8rem 1rem;
    margin-top: 1rem;
    box-shadow: 0 16px 35px rgba(0,0,0,0.18);
}

.section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.9rem;
}

.section-kicker {
    color: #fca5a5;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 700;
}

.section-header h3 {
    margin: 0.25rem 0 0 0;
    color: white;
    font-size: 1.2rem;
    font-weight: 800;
}

.section-pill {
    background: rgba(229,9,20,0.12);
    border: 1px solid rgba(229,9,20,0.35);
    border-radius: 999px;
    padding: 0.28rem 0.55rem;
    color: #fca5a5;
    font-size: 0.66rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
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
        ("Administrator", "admin@example.com", admin_username, admin_password_hash, datetime.now(timezone.utc).isoformat()),
    )
    c.execute(
        "UPDATE users SET name='Administrator', email='admin@example.com', password_hash=?, role='admin' WHERE username=?",
        (admin_password_hash, admin_username),
    )
    conn.commit()
    conn.close()

setup_db()

def normalize_username(value):
    return (value or "").strip()


def create_user(name,email,username,password):
    name = (name or "").strip()
    email = (email or "").strip()
    username = normalize_username(username)
    password = (password or "").strip()

    if not name or not email or not username or not password:
        return False, "Please fill in all fields."

    try:
        conn = db_conn()
        c = conn.cursor()
        c.execute("INSERT INTO users(name,email,username,password_hash,created_at,role) VALUES (?,?,?,?,?,?)",
                  (name, email, username, hash_pw(password), datetime.now(timezone.utc).isoformat(), "user"))
        conn.commit()
        conn.close()
        return True, ""
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    except Exception:
        return False, "Unable to create account right now."


def login(username,password):
    username = normalize_username(username)
    password = (password or "").strip()

    if not username or not password:
        return False, "Please enter your username and password."

    conn = db_conn()
    c = conn.cursor()
    c.execute("SELECT id,name,email,password_hash,COALESCE(role,'user') FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if not row:
        return False, "No such user"
    uid, name, email, pwh, role = row
    if pwh != hash_pw(password):
        return False, "Wrong password"
    return True, {"id": uid, "name": name, "email": email, "username": username, "role": role}

def add_favorite(uid,title):
    conn = db_conn()
    c = conn.cursor()
    c.execute("INSERT INTO favorites(user_id,title,added_at) VALUES (?,?,?)",
              (uid,title,datetime.now(timezone.utc).isoformat()))
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
            st.markdown('<div class="movie-card">', unsafe_allow_html=True)
            if poster:
                st.image(poster, use_container_width=True)
            else:
                st.markdown('<div class="no-poster">No poster</div>', unsafe_allow_html=True)

            st.markdown(f'<div class="movie-title">{title}</div>', unsafe_allow_html=True)
            button_key = f"save_{section}_{slugify(title)}_{i}"
            if st.button("Save", key=button_key, use_container_width=True):
                if not st.session_state.user:
                    st.session_state.pending_favorite = title
                    st.info("Please log in to save favorites.")
                else:
                    add_favorite(st.session_state.user["id"], title)
                    st.success(f"Saved: {title}")
            st.markdown('</div>', unsafe_allow_html=True)


def render_admin_dashboard():
    users = get_all_users()
    if users.empty:
        st.caption("No users yet.")
        return

    st.markdown("<div class='dashboard-shell'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='dashboard-header'>
            <div>
                <div class='dashboard-kicker'>Control center</div>
                <h3>Admin dashboard</h3>
            </div>
            <div class='dashboard-chip'>Live</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    admin_cols = st.columns(3)
    with admin_cols[0]:
        st.markdown(f"<div class='stat-panel'><span class='stat-label'>Total users</span><span class='stat-value'>{int(len(users))}</span></div>", unsafe_allow_html=True)
    with admin_cols[1]:
        st.markdown(f"<div class='stat-panel'><span class='stat-label'>Admins</span><span class='stat-value'>{int((users['role'] == 'admin').sum())}</span></div>", unsafe_allow_html=True)
    with admin_cols[2]:
        st.markdown(f"<div class='stat-panel'><span class='stat-label'>Saved titles</span><span class='stat-value'>{int(users['favorites'].sum())}</span></div>", unsafe_allow_html=True)

    st.markdown("<div class='table-panel'>", unsafe_allow_html=True)
    st.dataframe(
        users.rename(columns={"username": "Username", "role": "Role", "favorites": "Favorites"}),
        use_container_width=True,
        hide_index=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


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
        <div class='hero-panel'>
            <div style='font-size:2.8rem; font-weight:800; color:#e50914; letter-spacing:0.04em;'>umty</div>
            <div style='font-size:1.1rem; color:#d1d5db; margin-top:0.6rem;'>Your next movie night starts here.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="auth-shell">', unsafe_allow_html=True)
    login_tab, signup_tab = st.tabs(["Sign in", "Create account"])

    with login_tab:
        with st.form("landing_login_form"):
            username = st.text_input("Username", key="landing_username")
            password = st.text_input("Password", type="password", key="landing_password")
            submitted = st.form_submit_button("Sign in", type="primary")
            if submitted:
                ok, res = login(username, password)
                if ok:
                    st.session_state.user = res
                    st.success(f"Welcome back, {res['username']}!")
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
            signup = st.form_submit_button("Create account", type="primary")
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

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# -----------------------------------------------------------
# SIDEBAR AUTH
# -----------------------------------------------------------

with st.sidebar:
    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
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
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------

base_titles = ["The Matrix", "Inception", "Fight Club", "Pulp Fiction", "Avatar", "Sabrina", "The Dark Knight", "Interstellar"]
if st.session_state.user:
    welcome_name = st.session_state.user["name"] or st.session_state.user["username"]
    st.title(f"Welcome back, {welcome_name}", anchor=False)
else:
    st.title("umty", anchor=False)

st.caption("Trending movies, personal picks, and favorites built for your next watch night.")

featured_titles = base_titles[:3]
feature_title = featured_titles[0] if featured_titles else "The Matrix"
feature_poster = fetch_poster(feature_title)

with st.container():
    hero_cols = st.columns([1.6, 1])
    with hero_cols[0]:
        st.markdown(
            """
            <div class='premium-shell'>
                <div class='premium-badge'>Featured tonight</div>
                <div class='premium-title'>""" + feature_title + """</div>
                <div class='premium-subtitle'>Discover your next favorite film, build a watchlist, and let your saved picks power smarter recommendations.</div>
                <div class='premium-feature-grid'>
                    <div class='premium-feature'>
                        <div class='label'>Movies</div>
                        <div class='value'>""" + f"{len(movies_df):,}" + """</div>
                    </div>
                    <div class='premium-feature'>
                        <div class='label'>Users</div>
                        <div class='value'>""" + f"{ratings_df['userId'].nunique():,}" + """</div>
                    </div>
                    <div class='premium-feature'>
                        <div class='label'>Top rated</div>
                        <div class='value'>""" + str(int(ratings_df["rating"].max())) + """</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.session_state.user:
            st.markdown("<div style='margin-top: 0.8rem; color: #e5e7eb;'>Your account is ready. Pick a mood, save your favorites, and keep the night rolling.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='margin-top: 0.8rem; color: #e5e7eb;'>Sign in to unlock personalized suggestions, favorites, and a cleaner movie night experience.</div>", unsafe_allow_html=True)
    with hero_cols[1]:
        if feature_poster:
            st.image(feature_poster, width=320)
        else:
            st.markdown("<div style='height: 360px; background: linear-gradient(160deg, #1d1d1d, #0f0f0f); border-radius: 20px; display:flex; align-items:center; justify-content:center; color:#9ca3af; font-size:1rem;'>No poster</div>", unsafe_allow_html=True)

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

st.markdown("""
<div class='section-shell'>
    <div class='section-header'>
        <div>
            <div class='section-kicker'>Catalog</div>
            <h3>Streaming catalog</h3>
        </div>
        <div class='section-pill'>Now playing</div>
    </div>
</div>
""", unsafe_allow_html=True)

pill_html = "".join(
    f"<span class='stream-pill {'active' if cat == selected_category else ''}'>{cat}</span>" for cat in categories
)
st.markdown(f"<div class='stream-pills'>{pill_html}</div>", unsafe_allow_html=True)
render_row(titles, section="category")

# -----------------------------------------------------------
# AI RECOMMENDATIONS
# -----------------------------------------------------------

if st.session_state.user:
    st.markdown(
        """
        <div class='section-shell'>
            <div class='section-header'>
                <div>
                    <div class='section-kicker'>For you</div>
                    <h3>AI recommendations</h3>
                </div>
                <div class='section-pill'>Smart picks</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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
    st.markdown(
        """
        <div class='section-shell'>
            <div class='section-header'>
                <div>
                    <div class='section-kicker'>Favorites</div>
                    <h3>Saved watchlist</h3>
                </div>
                <div class='section-pill'>Login required</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Log in to save your watchlist and see recommendations tailored to your taste.")

if st.session_state.user:
    st.markdown(
        """
        <div class='section-shell'>
            <div class='section-header'>
                <div>
                    <div class='section-kicker'>My list</div>
                    <h3>Your favorites</h3>
                </div>
                <div class='section-pill'>Saved</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
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