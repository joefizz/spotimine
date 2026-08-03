FROM python:3.11-slim

# ffmpeg is required by spotdl/yt-dlp for audio conversion (and brings ffprobe,
# which the download verifier needs).
# nodejs is a JavaScript runtime for yt-dlp: YouTube will not hand over stream
# URLs without solving a JS signature challenge, and with no runtime yt-dlp
# returns an empty format list — which looks exactly like "no Premium".
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persistent data lives in volumes; pre-create so permissions are right
RUN mkdir -p songs static/reports

EXPOSE 5000

CMD ["python", "app.py"]
