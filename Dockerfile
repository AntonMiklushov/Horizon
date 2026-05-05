# Use a stable Python slim image instead of mutable latest tags.
FROM python:3.12-slim-bookworm

# Set working directory
WORKDIR /app

# Install uv for faster dependency management
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /uvx /usr/local/bin/

# Copy project files
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY data/config.example.json data/config.personal-news.example.json data/presets.json ./data/
COPY .env.example .env.example

# Install dependencies
RUN uv sync --frozen --no-dev

# Create volume mount points
VOLUME ["/app/data"]

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    UV_NO_SYNC=1

RUN useradd --create-home --uid 10001 horizon \
    && chown -R horizon:horizon /app/data
USER horizon

# Run the application
ENTRYPOINT ["uv", "run", "--no-sync", "horizon"]
CMD []
