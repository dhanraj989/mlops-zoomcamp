FROM mageai/mageai:latest

COPY requirements.txt .

RUN pip install -r requirements.txt