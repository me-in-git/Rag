Deploying to HuggingFace Spaces (Gradio)

1. Create a new Space on HuggingFace (https://huggingface.co/spaces) and choose the "Gradio" SDK.
2. Clone the Space's git repo locally or add it as a remote:

```bash
# from your repo root
git init
git add .
git commit -m "Prepare space"
# add the space remote from the UI and push, e.g.:
git remote add space https://huggingface.co/spaces/<username>/<space-name>.git
git push space main
```

3. Ensure `requirements.txt` is present at repo root (we added `gradio` and `rank_bm25`).
4. Ensure `space_app.py` is at repo root — Spaces will auto-run it. If your script is named differently, set `app_file` in Space settings.
5. Large model checkpoints should not be pushed to Spaces. Instead:
   - Use a hosted HF model (call `from_pretrained('<model-id>')`), or
   - Mount an external storage and load models at runtime (enterprise), or
   - Use smaller models that fit the Space's limits.

6. Once pushed, the Space will install dependencies and launch; you'll get a live URL in the Space UI.

If you want, I can prepare a minimal `.gitignore` for the Space and a small `README` explaining usage.
