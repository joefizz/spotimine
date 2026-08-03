FROM python:3.11-slim

# ffmpeg is required by spotdl/yt-dlp for audio conversion, and brings ffprobe,
# which the download verifier needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Deno is yt-dlp's JavaScript runtime. YouTube will not hand over stream URLs
# without solving a JS signature challenge, and with no runtime yt-dlp returns
# an empty format list — which looks exactly like "this account has no Premium".
# It must be Deno specifically: node is nominally supported but does not solve
# the challenge even with yt-dlp-ejs installed (verified). The solver scripts
# themselves (yt-dlp-ejs) already arrive as a spotdl dependency.
COPY --from=denoland/deno:bin /deno /usr/local/bin/deno

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Persistent data lives in volumes; pre-create so permissions are right
RUN mkdir -p songs static/reports

EXPOSE 5000

CMD ["python", "app.py"]
