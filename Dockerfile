# syntax=docker/dockerfile:1
FROM python:3.11-slim

# ffmpeg is required by spotdl/yt-dlp for audio conversion, and brings ffprobe,
# which the download verifier needs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies get their own layer so only a requirements.txt change can
# invalidate them — this is by far the slowest step in the build. The pip cache
# mount means even a requirements change reuses already-downloaded wheels
# instead of pulling librosa/scipy/numba/matplotlib down again.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Deno is yt-dlp's JavaScript runtime. YouTube will not hand over stream URLs
# without solving a JS signature challenge, and with no runtime yt-dlp returns
# an empty format list — which looks exactly like "this account has no Premium".
# It must be Deno specifically: node is nominally supported but does not solve
# the challenge even with yt-dlp-ejs installed (verified). The solver scripts
# themselves (yt-dlp-ejs) already arrive as a spotdl dependency.
# Kept below the dependency layer on purpose: it is a cheap copy that changes
# independently, and above pip it would invalidate the expensive install.
COPY --from=denoland/deno:bin /deno /usr/local/bin/deno

COPY . .

# Persistent data lives in volumes; pre-create so permissions are right
RUN mkdir -p songs static/reports

EXPOSE 5000

CMD ["python", "app.py"]
