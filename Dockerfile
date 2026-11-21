# Match sandbox: no network at runtime, read-only root, map at /arena.
FROM python:3.13-slim
WORKDIR /bot
COPY . .
ENTRYPOINT ["python3", "-u", "/bot/bot.py"]
