# Asterisk from Ubuntu's own package (no third-party image). Config files are
# rendered from .env by asterisk/render.py and mounted read-only at runtime.
FROM ubuntu:26.04
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends asterisk asterisk-modules ca-certificates \
 && rm -rf /var/lib/apt/lists/*
CMD ["asterisk", "-f", "-U", "asterisk", "-G", "asterisk", "-vvv"]
