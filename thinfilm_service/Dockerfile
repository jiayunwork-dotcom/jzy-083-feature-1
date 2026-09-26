# Thin-film optics service: characteristic-matrix R/T/A over HTTP.
#
# Build and run with one command:
#   docker build -t thinfilm-service . && docker run --rm -p 8080:8080 thinfilm-service
# (or: docker compose up --build)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Runtime dependencies first, so code changes do not bust the pip layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY thinopt ./thinopt
COPY run.py ./

# Run as an unprivileged user.
RUN useradd --create-home appuser
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health')"

CMD ["python", "run.py"]
