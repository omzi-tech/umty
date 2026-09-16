# umty

A Streamlit movie app with sign-in, favorites, genre browsing, and recommendations.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run umty/app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Select this repo and set the main file path to `umty/app.py`.
4. Add required dependencies from `requirements.txt`.
