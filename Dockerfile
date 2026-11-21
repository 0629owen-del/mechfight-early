# Mech Royale bot (mechfight/1). Build: docker build -t my-bot .
# Match sandbox: no network, read-only root (writable /tmp), uid 65534, 1 CPU, 512 MB, map at /arena.
FROM python:3.13-slim
# never write .pyc at runtime (read-only FS); bytecode is precompiled below at build time
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0
WORKDIR /bot
COPY . .
RUN python3 -m compileall -q /bot && chmod -R a+rX /bot
USER 65534:65534
ENTRYPOINT ["python3", "-u", "/bot/main.py"]
