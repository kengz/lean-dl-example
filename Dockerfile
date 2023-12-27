FROM python:3.11-slim

ENV WORKDIR=/app
WORKDIR $WORKDIR

RUN apt-get update \
    && apt-get install -y curl nano \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry==1.6.1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install dependencies
ENV PATH="$WORKDIR/.venv/bin:$PATH"
COPY poetry.toml poetry.lock pyproject.toml ./
RUN poetry install --no-root --without dev && rm -rf $POETRY_CACHE_DIR

COPY . .
RUN poetry install --without dev

CMD ["python", "dl/train.py"]
