"""Submission inference entrypoint.

Required env vars:
- API_BASE_URL
- MODEL_NAME
- HF_TOKEN
"""

from app.baseline_openai import main


if __name__ == "__main__":
    main()
