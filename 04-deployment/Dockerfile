FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

# Install dependencies
COPY Pipfile Pipfile.lock ./
RUN pip install pipenv && pipenv install --deploy --system

# Copy your script
COPY starter.py .

# Set the entrypoint (optional)
ENTRYPOINT ["python", "starter.py"]