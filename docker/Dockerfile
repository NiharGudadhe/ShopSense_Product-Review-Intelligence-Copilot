FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/src

COPY docker/requirements-api.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-api.txt

COPY src ./src

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "shopsense.api.app:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
