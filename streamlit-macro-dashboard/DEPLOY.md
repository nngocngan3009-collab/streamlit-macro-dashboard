# Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (Public).
2. Go to https://share.streamlit.io/ → "Deploy an app".
3. Choose this repo, branch `main`, and entry file `app.py`.
4. In your app's Settings → Secrets, add:
   ```
   GEMINI_API_KEY = "your_real_key"
   ```
5. Click **Deploy**.

Each `git push` to the selected branch will auto-redeploy.