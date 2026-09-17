FROM python:3.11-slim

# System deps needed by pandas/scikit-learn wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better layer caching)
COPY src/requirements.txt ./src/requirements.txt
RUN pip install --no-cache-dir -r src/requirements.txt

# Copy the rest of the repo (models/, src/, backend/, data/ if present)
COPY . .

WORKDIR /app/backend

EXPOSE 8000

# -w 1: single worker is required — app state (the simulated wells) lives
# in-memory in one process. Running multiple workers would give each
# request a different, inconsistent view of the fleet.
# $PORT is set by Heroku at runtime; defaults to 8000 for plain `docker run`.
ENV PORT=8000
CMD uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1
