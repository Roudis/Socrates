# STAGE 1: Builder
# Use a full python image to build dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies required for building python packages (e.g. PyMuPDF, Chroma)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python Dependency Manager (using pip for simplicity here, but Poetry is recommended)
# Install into a local directory that we can copy later
COPY requirements.txt.
RUN pip install --user --no-cache-dir -r requirements.txt

# STAGE 2: Runtime
# Use a slim image for production
FROM python:3.11-slim as runtime

WORKDIR /app

# Create a non-root user for security
RUN useradd -m scorates
USER scorates

# Copy installed packages from builder stage
COPY --from=builder /root/.local /home/scorates/.local

# Update PATH
ENV PATH=/home/scorates/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

# Copy Application Code
COPY --chown=scorates:scorates..

# Expose the port
EXPOSE 8000

# Start Command using Uvicorn (Production Server)
# We point to the FastAPI app wrapper that mounts Chainlit
CMD ["uvicorn", "app.presentation.api.main:app", "--host", "0.0.0.0", "--port", "8000"]