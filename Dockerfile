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

# sockseek (formerly sldl / slsk-batchdl) is the Soulseek source used as a second
# pass for tracks YouTube Music only serves at 128 kbps. Optional at runtime: the
# pass skips itself with an explanation if the binary or credentials are absent.
# Also kept below the dependency layer so bumping it does not trigger a full
# reinstall. TARGETARCH is supplied by BuildKit.
ARG TARGETARCH
ARG SOCKSEEK_VERSION=3.0.4
RUN set -u; \
    case "${TARGETARCH:-amd64}" in \
      amd64) ss_arch=x64 ;; \
      arm)   ss_arch=arm ;; \
      *)     ss_arch=""  ;; \
    esac; \
    if [ -z "$ss_arch" ]; then \
      echo "NOTE: sockseek publishes no linux build for '${TARGETARCH}'."; \
      echo "      The Soulseek pass will skip itself; everything else works."; \
    else \
      url="https://github.com/fiso64/sockseek/releases/download/v${SOCKSEEK_VERSION}/sockseek_${SOCKSEEK_VERSION}_linux-${ss_arch}.tar.gz"; \
      if python -c "import sys,urllib.request;urllib.request.urlretrieve(sys.argv[1],'/tmp/ss.tgz')" "$url" \
         && mkdir -p /tmp/ss && tar -xzf /tmp/ss.tgz -C /tmp/ss; then \
        bin="$(find /tmp/ss -maxdepth 3 -type f -name sockseek | head -1)"; \
        if [ -n "$bin" ]; then \
          mv "$bin" /usr/local/bin/sockseek && chmod +x /usr/local/bin/sockseek; \
          # sockseek is a .NET binary and aborts at startup without ICU. The
          # package is version-suffixed (libicu76 on trixie, 72 on bookworm), so
          # discover it rather than pinning a name that breaks on a base bump.
          apt-get update && \
          apt-get install -y --no-install-recommends \
            "$(apt-cache search --names-only '^libicu[0-9]+$' | head -1 | cut -d' ' -f1)" && \
          rm -rf /var/lib/apt/lists/*; \
          echo "installed sockseek ${SOCKSEEK_VERSION} (${ss_arch})"; \
        else \
          echo "WARNING: sockseek binary not found in the archive; Soulseek pass will skip."; \
        fi; \
      else \
        echo "WARNING: could not download sockseek; Soulseek pass will skip itself."; \
      fi; \
      rm -rf /tmp/ss /tmp/ss.tgz; \
    fi

COPY . .

# Persistent data lives in volumes; pre-create so permissions are right
RUN mkdir -p songs static/reports

EXPOSE 5000

CMD ["python", "app.py"]
