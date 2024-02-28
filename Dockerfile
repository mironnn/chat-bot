FROM python:3.12-slim-bullseye

# ---Install additional dedian dependencies---
RUN apt-get update

ENV LANG=en_US.UTF-8

COPY ./pyproject.toml /service/pyproject.toml
COPY ./poetry.lock /service/poetry.lock
COPY ./src /service/src/

WORKDIR /service

RUN pip install -U pip poetry && \
    poetry config virtualenvs.create false && \
    poetry install --without dev --no-interaction --no-ansi --no-root


EXPOSE 8080
CMD [ "python", "-m", "src.main"]
