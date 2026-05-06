FROM python:3.12-slim

COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY . /app/

RUN uv sync

CMD ["uv", "run", "streamlit", "run", "chat.py"]