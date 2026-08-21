FROM python:3.11-slim

RUN pip install --no-cache-dir "poetry>=2.0"

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Install the locked dependencies first so this layer stays cached.
COPY pyproject.toml poetry.lock README.md ./
RUN poetry install --without train,dev && rm -rf $POETRY_CACHE_DIR

COPY . .

EXPOSE 8511

# The app serves on 8511 (see .streamlit/config.toml). Checked with urllib so
# the image does not need curl installed.
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8511/_stcore/health')"

ENTRYPOINT ["poetry", "run", "streamlit", "run", "app.py"]
