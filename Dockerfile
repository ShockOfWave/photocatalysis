FROM python:3.10-slim

RUN pip install poetry

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY . .

RUN poetry install --without train && rm -rf $POETRY_CACHE_DIR

EXPOSE 8511

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcode/health

ENTRYPOINT ["poetry", "run", "streamlit", "run", "app.py"]
