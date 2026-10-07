FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv && uv sync --extra server --no-dev
COPY backend ./backend
COPY db ./db
COPY data ./data
COPY web_demo/data.json ./web_demo/data.json
CMD ["uv", "run", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
