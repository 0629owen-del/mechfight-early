# Mech Royale bot (mechfight/1)

Python 3.13, standard library only. Entry point `main.py` (Docker `ENTRYPOINT python3 -u /bot/main.py`).
Design: `../mechfight-kit/docs/game-plan-v2.md`. Module interfaces: `CONTRACTS.md`.

## Layout

| File | Role |
| --- | --- |
| `main.py` | `Bot(hello, deadline)` / `act(obs, t0)`: binds config, runs the plan 6.1 pipeline (CONTRACTS section 2) with per-stage fallbacks, deadline accounting and R15.3 degrade levels |
| `runner.py` | stdin/stdout loop: newest-obs rule, tick echo, worker thread with hard-deadline guard, exception counter and T1 switch to SAFE, `ready` within the hello budget, action sanitizer, gc control |
| `safe.py` | X5 SAFE controller (pure function of obs + config) |
| `config.py` | plan section 10 constants, `bind(hello.config)`, `update_live(C, obs.self)`; the table block is generated |
| `contracts.py` | shared data types (Track, Ghost, Intent, ...) and the `Clock` budget helper |
| `mechfight.py` | kit wrapper, unmodified (types only; its `run()` is replaced by `runner.py`) |
| `tools/gen_config.py` | regenerates the table block of `config.py` from the plan |
| `tools/minisim.py`, `tools/run_matches.py` | local approximate match server and batch runner (`tools/MINISIM.md` says what it can and cannot measure) |
| `tests/` | `unittest` suites |

## Commands

```sh
python3 tools/gen_config.py            # after editing plan section 10 (--check to verify)
python3 -m unittest discover -s tests
cd ../mechfight-kit && python3 check/check.py -- python3 -u ../bot/main.py
docker build -t my-bot .               # then the sandboxed check from protocol.md section 1
```

Logs and telemetry go to stderr only (rate-limited JSONL). Debug drawings carry at most a short label.

## Baseline build

`BOT_SAFE_ONLY=1` in the environment keeps the bot on the SAFE controller for the whole match (walk to
the next circle's center, shoot the nearest mech, GP upgrade list). It exists to answer one question:
what does the rest of the code add? Locally:

```sh
python3 tools/run_matches.py -k 90 -j 5 --seed 100 --out runs/safe --bot "env BOT_SAFE_ONLY=1 python3 -u main.py"
```

For a baseline submission, add one line to the `Dockerfile` on a separate branch and submit that commit:

```dockerfile
ENV BOT_SAFE_ONLY=1
```

A submission is a public GitHub repository with the `Dockerfile` at its top level (protocol.md section 1,
20 MB at the submitted commit). Compare a candidate only with a baseline or incumbent submitted in the
same round (plan 14.5).
