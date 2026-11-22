# minisim: local approximate match server

`tools/minisim.py` plays a mechfight/1 match on this machine: 16 bot processes, the real map, the
real config (copied from the recorded `hello` in `mechfight-kit/check/stream.jsonl.gz`). It is the
bot team's only source of gameplay metrics (game-plan-v2 §14.4): real matches show no stderr and
debug drawings can't be downloaded. **Treat every number it produces as a hypothesis** (§14.5): the
physics is simplified and the opponents are scripted guesses at the built-ins.

Stdlib only. Speed: a full match with 16 Python bots takes about 5-15 s of wall time (40x real
time with the scripted bots); with heavier bots the slowest bot sets the pace (lockstep).

## Files

| File | What |
| --- | --- |
| `tools/minisim.py` | the server: geometry, rules (`World`), bot processes, ground-truth log |
| `tools/minisim_bots.py` | scripted opponents, one process each: `duelist`, `rusher`, `lobber`, `looter`; engagement knobs calibrated (see Calibration) |
| `tools/run_matches.py` | runs K matches (parallel) and aggregates metrics into `summary.json` + `summary.txt` |
| `tools/obs_recorder.py` | observer harness: the kit example bot, unchanged, dumping every observation to its stderr |
| `tools/calibrate.py` | observer-side metrics of minisim batches vs the one real recording (percentiles) |
| `tools/tracebot.py` | our bot plus a per-decision trace on stderr, for reading one bad match tick by tick |
| `tests/test_minisim.py` | rule sanity tests + smoke match (`python3 -m unittest tests.test_minisim -v`) |

## Usage

```sh
# one match: our bot in a random slot, 15 scripted built-in-like opponents
python3 tools/minisim.py --out runs/m1 --seed 7 --bot "python3 -u main.py"

# kit example bot vs duelists only, 60 s, wait forever for answers (debugging)
python3 tools/minisim.py --out runs/m2 --bot example --fill scripted:duelist --duration 60 --no-timeout

# several copies of our bot
python3 tools/minisim.py --out runs/m3 --bot "python3 -u main.py" --bot "python3 -u main.py" --fill scripted:mix

# a batch with aggregation (4 matches in parallel); args after -- go to minisim.py
python3 tools/run_matches.py -k 16 -j 4 --out runs/b1 --bot "python3 -u main.py" -- --cone half
python3 tools/run_matches.py --aggregate runs/b1          # re-aggregate existing results
```

Bot specs: a shell-style command line (leading `VAR=value` words become environment variables),
`example` (kit `python/example_bot.py` with the kit's `python/` on `PYTHONPATH`), or
`scripted:<arch>` (`duelist`, `rusher`, `lobber`, `looter`, or `mix`: a blend weighted towards
gun+laser duelists, the one recorded match's built-in loadout; calibrated engagement, see Calibration),
or `scripted:<arch>_v1` (e.g. `scripted:mix_v1`: the original fire-on-sight opponents, event-for-event
identical to the pre-calibration code). Slots are shuffled by the seed
(`--no-shuffle` keeps the given order). Bot processes run with the current directory as cwd and
receive `hello.map.dir` = the kit's `arena/` directory.

Options: `--mechs N`, `--duration S`, `--budget-ms MS`, `--no-timeout`, `--kit DIR`, `--quiet`,
and the rule toggles for spec ambiguities: `--cone half` (lightning cone ±cone_deg/2 instead of
±cone_deg, A-19), `--mortar-stack` (direct mortar hits also take splash, A-2),
`--drop-npc-blocks` (NPCs count as drop-ring occupants, A-6), and `--peace` (weapons and drones do no
damage to mechs, so every mech lives until the zone takes it: a test of rotation and endgame positioning,
see "What the batches cannot reach").

## Output

`<out>/slotNN.stderr`: every stderr line of the bot in slot NN, verbatim (our bot's JSONL telemetry).

`<out>/match.jsonl` (ground truth, one JSON object per line):

| `type` | Fields |
| --- | --- |
| `meta` | seed, n, duration, rule toggles, slots (slot, spec, label), zone circles |
| `shot` | t, by, weapon, target (intended mech: visible, nearest in angle within ~10° and in range; lightning: the mech it hit), range (3D, m), moving (target speed > 3 m/s), self_moving, hit, hit_ids, dmg, charge (laser); written when the shot resolves |
| `sight` | first time `by` sees `target`: t, mode (`near`/`wedge`), dist (eye to centre), hdist, target_loadout |
| `fight` | one per pair episode (starts at the first mech-on-mech damage, ends at a death or 10 s without damage between them): t0, t1, a, b, initiator, range at start, loadouts, hp/cubes at start, hp at end, damage each way, outcome (`kill`/`disengage`/`timeout`), winner |
| `death` | t, id, cause (`gun`/`laser`/`lightning`/`mortar`/`npc`/`zone`/`self_mortar`), by, killer, place, pos, killer_loadout, zone_taken |
| `take`, `upgrade`, `npc_kill`, `drop_call`, `drop_capture` | loot and progression events |
| `place` | per slot: place, score, kills, spec/label/name, hp, cubes, loadout, upgrades, damage dealt/taken, zone damage, death time, answered/missed/late/stale decisions, bad lines, answer time p50/p99/max (ms) |
| `summary` | wall_s, t_end, ticks, realtime factor |

`run_matches.py` aggregates: placement/score per spec (+ win/top-3 rate, kills, survival, missed
decisions, worst p99 answer time), hit rate by weapon × range bucket (and by target moving/still),
first-sighting distance distribution (median, quartiles, histogram, near/wedge share), fight
win/loss/draw by loadout × start-range bucket (per spec, overall, and loadout-vs-loadout), death
causes per spec.

## Modeled vs approximated

"exact" = implemented as protocol.md states it, with numbers read from `hello.config`.

| Area | Model here | Real server |
| --- | --- | --- |
| Protocol (§2, §6) | exact: hello (slot, per-slot seed, map, full config), ready within hello budget, lockstep obs → actions, 2 ticks per decision, tick echo, stale/late answers dropped, coast `coast_s` then idle, `end` with place and kills (sent at death), bad lines counted, names cut to 32 | exact rules; server-internal timing unknown |
| 50 ms budget | wall clock per decision from the write of `obs` to the read of the answer, all bots in parallel; `--no-timeout` waits | same rule; the scorer's CPU differs |
| Geometry | prisms rasterised to the 1 m terrain grid (cell centres): per cell a solid column (ground..top, including stacked parts) plus overhang intervals (bridges); features thinner than ~1 m can leak | exact polygons |
| Terrain | bilinear for feet; nearest vertex for shots and LOS | bilinear |
| Movement | exact accelerations, walk/boost/charge speeds, jump, jets (capped rise), energy drain/regen delay, gravity, map edge | exact |
| Collision | centre + 8 (12 when large) perimeter samples vs the grid; slide = axis split, then ±30°/±60° deflections; step-up `step_up`; slope limit on terrain rise under the centre; ceilings stop jets | exact cylinder sweep |
| Roof edges / eaves | perimeter samples support you when landing (edge landing); slid off at 0.6 m/tick unless moving onto the roof | as protocol |
| Mech overlap | pairwise horizontal push-apart, half each | unknown details |
| Fall damage | none (A-1) | unstated |
| Gun | exact projectile, Gaussian spread σ = `spread_deg` on yaw and pitch, exact fire-rate accumulator, mag, auto reload at 0, manual reload, range | exact |
| Laser | exact charge / min_charge / hold_s / cooldown / half speed / cook-off / pierce; hitscan vs cylinders and the grid | exact |
| Lightning | nearest target (mech, NPC, crate) by 3D distance from the muzzle, yaw cone ±`cone_deg` (default) or ±`cone_deg`/2 (`--cone half`), LOS through buildings, chain to nearest unhit within `chain_range` with falloff, energy, short; fires without a target | exact except the cone reading (A-19 unknown) |
| Mortar | ballistic shell (exact gravity/ttl), direct hit = `damage` (splash too with `--mortar-stack`), splash linear to `splash_edge` at `splash_radius` measured to the target cylinder, blocked by walls, knock/knock_up/drain scaled, own blast 50% and throws you | exact except stacking (A-2) |
| Shot origin | `radius + 0.5` along the aim (scaled), pulled in to a closer wall | exact |
| Rear armor | source direction (bullet origin, beam/arc origin, blast point) outside ±front_arc/2 of the aim yaw → × `rear_mult`; applies to NPC bullets too | exact |
| Cubes, size, upgrades | exact: +hp/+damage to `stat_cap`, size growth uncapped, offers at thresholds, 3 distinct eligible choices, weapon upgrades need the weapon, timeout applies choice 0, queueing; stats paths `mech.*`/`weapons.*` mul/add | exact (choice draw RNG unknown) |
| Health | exact regen delay/rate, no regen outside zone, zone resets the timer, armor not vs zone | exact |
| Zone | exact phases/dps/linear shrink; obs reports the phase-start circle, next circle and window (bot lerps); next centres uniform in the allowed disc, kept inside the bounds; radius 0 = nobody inside (A-9) | centre picking unknown |
| Vision | exact ranges/wedge/near/memory; eye → target centre or top; LOS march every 1.5 m on the grid; canopy = crowns as cylinders, blocked when exp(−canopyDensity·metres) < canopyBlockT; trunks ignored; item LOS cached 0.5 s per observer (or until it moves 4 m); projectiles use the same test | exact LOS; crown shape unknown |
| Beams | shown if the start, end or a segment midpoint passes the vision test | same rule |
| Sounds | exact kinds, ranges, near/mid/far bands, age, Gaussian bearing noise × `mech.hearing`; steps every `step_every` above `step_speed` (boost while boosting); jet every 0.3 s | exact except jet cadence |
| Events | exact types and fields; zone damage aggregated into one `damaged` per observation | zone event granularity unknown |
| NPCs | 12 drones: wander within leash, aggro on attack or on LOS within `aggro_range` while within `leash`, chase until `deaggro_range` / `chase_leash`, return home; straight-line walking (no pathfinding); bullets aimed at the centre (no lead) with spread; 2 cubes on death | pathing and aim unknown |
| Crates, loot | exact counts/HP/weapon chance/weights/dead-mech drops/take-drop rules/equip time/contested take; spawn points uniform on open ground in the start circle, roofs = random building tops ≤ 55 m | spawn distribution unknown |
| Supply drops | exact times/fall/ring/capture/freeze/restart/cubes/repair; landing point uniform within `spread` × next radius on open ground; ring = horizontal radius with feet −3..+10 m of the drop | details unknown |
| Placement | exact: death order, same-tick deaths share `alive`, clock end ranks survivors by HP (ties shared) | exact |
| Kill credit | mech whose damage took HP to ≤ 0 (own mortar: none) | unknown |
| Spawns | 16 points ≥ `spawn_min_separation` within `spawn_zone_fraction` × start radius on open ground (separation relaxed 5% per 10 failed attempts) | unknown algorithm (A-22) |

## Known gaps

- Opponent quality is the biggest unknown (Q-3/Q-11). Scripted bots lead perfectly with the gun and
  release the laser on target. Their engagement timing (when they start, how long they shoot, how
  they pause) is calibrated against ONE recorded match (see Calibration); the v1 fire-on-sight bots
  were about twice as lethal early (6 deaths by 36.8 s vs 3 real).
- NPC drones look too active early: the real observer heard no NPC gun in 36.8 s; minisim observers
  hear a median of 14-19 and are shot by NPCs in about a third of matches (placement/aggro, unchanged).
- No pathfinding for scripted bots and NPCs (stuck → jump/jet detour), so they rarely use roofs on purpose.
- Debug drawings are accepted and ignored (only line length is checked; caps not enforced).
- Answer times are measured on this machine with all bots sharing its CPUs; they are not the
  1-CPU container's times (use `tools/bench.py` / check.py in docker for CPU gates).
- Not modeled: exact polygon collision, tree trunks blocking sight, server tie-breaks we can't
  see, fall damage (assumed none), replay/spectator data.

## Calibration against the real recording (2026-11-21)

The only real match we have is `mechfight-kit/check/stream.jsonl.gz`: 300 observations (t = 0-36.8 s,
with a recording gap at 29.5-35.9 s) of slot 3, the kit example bot. One built-in (id 10, gun+laser)
engaged it. `tools/obs_recorder.py` runs the same example bot in minisim and dumps its observations;
`tools/calibrate.py` computes the same observer-side metrics from both streams (the minisim stream is
masked to the real stream's observation times) and prints the minisim median, IQR and the real value's
percentile.

```sh
python3 tools/run_matches.py -k 30 -j 4 --seed 1000 --out runs/cal --bot "python3 -u tools/obs_recorder.py"
python3 tools/calibrate.py runs/cal                  # add --spawn-band 100 to condition on spawn radius
python3 tools/run_matches.py -k 30 -j 4 --seed 1000 --out runs/cal_v1 --fill scripted:mix_v1 \
    --bot "python3 -u tools/obs_recorder.py"         # the pre-calibration opponents
python3 tools/calibrate.py runs/cal_v1 runs/cal      # side by side
```

Keep `-j` at 4-6 on a 10-core machine: overloaded runs miss 50 ms decisions and stop being reproducible.

### What the recording shows (observer's view)

- Alive 16 until 18.0 s, then 15, 14 at 22.7 s and 13 at 24.9 s: 1 death by 20 s, 3 by 36.8 s.
- 62 gun sounds (48 far, 14 mid, 0 near), first at 3.8 s; 6 lasers, 5 explosions, 1 mortar launch,
  2 crates, 3 deaths, no NPC gun.
- Built-in id 10, first seen at 4.9 s, 140 m away, aiming 1.2-1.8 rad off the observer (it had not seen
  it). Hit in the back at 5.7 s, it was aiming at the observer by 6.2 s, started a laser charge at 7.1 s
  and fired a full charge at 7.9 s. Never fired the gun while charging; gun shots 0.3-0.5 s apart during
  the laser cooldown. Fired 7.1-14.1 s (3 lasers, 12 gun rounds = one magazine), 18.5-19.8 s, 24.2-24.4 s
  and from about 35 s; between those it stayed in view and in range (91-150 m) without shooting.
- Damage taken from id 10: 4 lasers (315 each) and 9 gun hits (45 each) = 1665 over 7.9-36.8 s, at
  55-151 m (median 148 m). Laser 4 hits from 5 beams; gun about 9 hits from 11 rounds attributable to
  it. The observer ended at 9.5 HP (net HP loss 1390 after regen) and hit id 10 ten times.

### Knobs (`tools/minisim_bots.py`, top of file)

All are module constants read from `MINISIM_BOTS_<NAME>` (env) over the profile default;
`MINISIM_BOTS_PROFILE=v1` (or spec `scripted:<arch>_v1`) restores the original bots exactly.
"U(0, j)" = uniform random per bot or per event, from a separate RNG so v1 replays unchanged.

| Knob | v1 | v2 (default) | Evidence |
| --- | --- | --- | --- |
| `PEACE_S` + U(0, `PEACE_JITTER_S`) | 0 | 3 + U(0, 2) s | no shooting at all (mechs, crates, NPCs) and mechs ignored until then, unless a mech hits us; first gun heard at 3.8 s (v1: 0.2 s) |
| `REACT_S` + U(0, `REACT_JITTER_S`) | 0 | 0.5 + U(0, 0.6) s | new contact -> first trigger pull (turning extra); id 10: aim-on -> charge 0.9 s and 0.4 s |
| `ENGAGE_MAX_M` | none | 160 m | open fire only inside this range unless that mech hit us in the last 4 s; every real shot came from <= 151 m |
| `GUN_INTERVAL_S` ± `GUN_INTERVAL_JITTER_S` | 0 | 0.32 ± 0.1 s | gun cadence, any target (gives 0.3/0.4/0.5 s); id 10: 0.3-0.5 s, mean 0.36 s |
| `GUN_WHILE_CHARGING` | 1 | 0 | id 10 never fired the gun during its 4 observed charges |
| `BURST_S` + U(0, `BURST_JITTER_S`) | inf | 1 + U(0, 6) s | shoot this long, then pause: no new shots, keep moving and aiming, finish a laser charge |
| `PAUSE_S` + U(0, `PAUSE_JITTER_S`) | 0 | 4 + U(0, 7) s | id 10's bursts 7.0, 1.3, 0.2, 1.7+ s; pauses 4.4, 4.4, ~10.7 s with the target in view |
| `CONTACT_GAP_S` | 1 | 1 s | a mech missing from `mechs` this long is a new contact |

Not changed: aim (perfect lead; the real built-in was as accurate), archetype mix, movement bands.
The values were picked from 6 variants on seeds 1000-1029, then checked on seeds 2000-2029 (not used
for tuning).

### Before / after (30 matches each, observer = example bot, 15 x `scripted:mix`)

Cells: minisim median [IQR] and the real value's percentile (p). "cal" = seeds 1000-1029 (tuning),
"val" = seeds 2000-2029 (held out). Sounds are per 30 s of covered observation time.

| Metric, 0-36.8 s | Real | v1 cal | v2 cal | v1 val | v2 val |
| --- | --- | --- | --- | --- | --- |
| deaths by 20 s | 1 | 3 [2, 4] p5 | 1 [0, 1] p65 | 4 [3, 4] p2 | 1 [0, 1] p55 |
| deaths by 36.8 s | 3 | 6 [5, 7] p2 | 3 [2.3, 4] p47 | 6 [5.3, 7] p2 | 3.5 [2, 4] p40 |
| first death (s) | 18.0 | 8.1 [7.4, 11.4] p97 | 19.3 [15.5, 20.5] p40 | 8.6 [7.3, 11.4] p100 | 17.1 [13.0, 21.8] p53 |
| gun sounds heard | 62 | 191 [119, 237] p4 | 105 [66, 124] p17 | 193 [152, 253] p0 | 108 [89, 149] p14 |
| gun far / mid (median) | 48 / 14 | 126 / 39 | 81 / 20 | 126 / 55 | 81 / 33 |
| first gun heard (s) | 3.8 | 0.2 [0.1, 0.4] p93 | 4.1 [3.7, 4.7] p28 | 0.2 [0.1, 0.6] p93 | 4.2 [3.8, 4.6] p27 |
| laser sounds | 6 | 5 [2.5, 8.6] p61 | 3 [0, 5.8] p77 | 4 [2.3, 8] p67 | 2 [1.4, 5] p83 |
| NPC gun sounds | 0 | 19 [6, 33] p9 | 19 [8, 31] p8 | 10 [1, 22] p13 | 14 [6, 26] p10 |
| attacker: first sight -> first shot (s) | 2.9 | 1.2 [-0.1, 4.9] p58 | 2.3 [1.6, 4.3] p64 | 0.5 [0.1, 1.4] p92 | 1.8 [0.9, 4.5] p64 |
| same, ground truth from the attacker's own first sight (s) | - | 0.35 [0.25, 0.35] | 1.2 [1.0, 4.8] | 0.25 [0.18, 0.35] | 1.9 [0.9, 3.3] |
| attacker range at its hits (m) | 148 | 46 [40, 117] p91 | 122 [43, 141] p85 | 92 [86, 118] p92 | 74 [52, 93] p100 |
| attacker DPS while engaged | 56 | 117 [57, 151] p26 | 92 [48, 160] p28 | 137 [85, 183] p14 | 100 [50, 136] p26 |
| attacker DPS inside bursts | 191 | 123 [96, 151] p84 | 114 [64, 160] p89 | 144 [104, 183] p77 | 112 [86, 177] p83 |
| attacker bursts (3 s gap) | 3 | 1 [1, 1] p97 | 1 [1, 1] p100 | 1 [1, 1] p100 | 1 [1, 2] p96 |
| damage duty cycle (1 s bins) | 0.32 | 0.85 [0.68, 1] p6 | 0.9 [0.65, 1] p8 | 0.85 [0.7, 1] p0 | 0.65 [0.45, 1] p5 |
| attacker gun hit rate (sound estimate) | 0.82 | 1.0 [0.8, 1] p28 | 0.8 [0.26, 1] p53 | 1.0 [0.88, 1] p18 | 1.0 [0.98, 1] p20 |
| attacker laser hit rate | 0.8 | 1.0 (n=2) | 1.0 (n=5) | 1.0 (n=5) | 1.0 (n=6) |
| GT: shots per scripted bot-second, 0-60 s | - | 0.93 | 0.51 | 0.97 | 0.53 |
| GT: deaths by 60 s / 120 s | - | 8.5 / 12 | 6 / 9 | 8 / 11 | 6 / 10 |

Spawn position matters for sounds: the real observer spawned 347 m from the zone centre. With
`--spawn-band 100` (spawn radius 247-447 m; 38 of 60 matches per version, both seed sets pooled):
gun sounds v1 156 [105, 203] p3, v2 89 [65, 112] p22; deaths by 36.8 s v1 6 [5, 7] p3, v2 3 [2, 4] p47.

### What still does not match

- **Gun noise** stays about 1.5x the real count (real at p14-22). Longer pauses or a later, more spread
  onset (`PEACE_JITTER_S` 6) did not lower it much and pushed the first gun out of the IQR.
- **Burst structure seen by the observer:** in minisim the observer's engagements are mostly one burst
  (it dies, often 2-on-1, or the attacker moves on), so duty stays high. The real observer survived
  three bursts from a single attacker. The pauses exist in the ground truth (4-11 s by construction).
- **Engagement range:** real 148 m, minisim 74-122 m. Primary attackers in minisim are often rushers,
  lobbers or looters, and the example bot walks into its attacker.
- **Lasers:** the real attacker held a laser at 4.9 s; only ~10% of scripted bots hold one by 5 s (28%
  by 36.8 s), so laser sounds and laser hits on the observer are rarer than in the recording. That is
  the loot model, not engagement logic.
- **NPCs** (world model, unchanged): no NPC gun in the recording vs a median of 14-19 in minisim.

### Our bot on the calibrated opponents (seeds 100-129, 30 matches, 15 x `scripted:mix`)

| | HEAD 80b7597 | working tree (E1-E4) |
| --- | --- | --- |
| mean place (median), wins, top 3 | 5.00 (3), 11, 17 | 5.73 (6), 9, 13 |
| paired place difference (WT - HEAD) | | +0.73 +- 1.02 SE (11 better, 12 worse, 7 same) |
| deaths before 60 s | 7 | 7 |
| death causes (all deaths) | gun 8, laser 6, npc 3, mortar 1, zone 1 | npc 7, gun 7, laser 6, lightning 1 |
| macro time share | X1 75.8%, X2 23.1%, X3 1.1% | X1 72.4%, X3 14.9%, X2 12.6% |

On v1 opponents the same seeds gave mean place 8.43 / 8.53 and 18 / 19 deaths before 60 s. No
measurable difference between the versions; the working tree spends 14x more time in X3, and 3 of its 7
NPC deaths came in X3 (minisim NPCs look over-active, see above).

### Caveats

One recording, one observer, one attacker. Every percentile places a single draw in a distribution,
so p5-p95 is all "consistent"; the knob values are a plausible fit, not measurements, and several
(burst and pause lengths, peace jitter) rest on 2-4 observed events. Several metrics need the attacker
to be visible and are computed on 10-23 matches. Re-check against the real practice score (A/B
submissions) before trusting minisim-tuned early-game thresholds.

## Review round (2026-11-21)

Batches of 90 matches on seeds 100-189 against 15 x `scripted:mix` (calibrated). Every build takes the
same slot on the same seed, so differences are paired. "late ticks" = our missed + late answers summed
over the batch: it rises when other work runs on the machine, and a loaded batch plays a little worse,
so compare batches with similar counts.

| build (90 matches each) | mean place | wins | top 3 | dead < 60 s | kills | drops taken | fight time | flee time | zone deaths | late ticks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kit example bot | 8.76 | 3 | 18 | 39 | 0.48 | 0 | - | - | 2 | 15 |
| SAFE controller only | 9.09 | 2 | 18 | 41 | 0.39 | 0 | - | - | 2 | 17 |
| 80b7597 (before E1-E4) | 5.89 | 28 | 42 | 25 | 2.43 | 44 | 33% | 3% | 1 | 16 |
| e1be9c5 (committed, E1-E4) | 5.81 | 29 | 43 | 24 | 2.19 | 40 | 14% | 18% | 1 | 21 |
| f1: committed + armor + endgame point | 5.92 | 30 | 46 | 27 | 2.12 | 41 | 13% | 17% | 5 | 89 |
| f3: f1 without E1-E4 | 5.27 | 30 | 48 | 21 | 2.78 | 46 | 35% | 3% | 5 | 29 |
| g1: f1 + first round of later fixes | 6.33 | 27 | 35 | 25 | 2.04 | 30 | 16% | 17% | 0 | 50 |
| g2: g1, THIRD_PARTY_RULE summed | 6.41 | 17 | 33 | 27 | 1.73 | 20 | 15% | 18% | 3 | 98 |
| g3: g1, ENGAGE_MODEL live | 6.53 | 22 | 36 | 24 | 2.31 | 33 | 36% | 3% | 2 | 144 |
| h1: all fixes except leg execution | 6.16 | 25 | 38 | 23 | 2.02 | 39 | 15% | 18% | 5 | 16 |
| k1: final tree | 6.58 | 29 | 36 | 29 | 2.20 | 50 | 16% | 19% | 1 | 4 |
| k2: final tree, ENGAGE_MODEL live | 6.68 | 25 | 34 | 31 | 2.44 | 45 | 37% | 3% | 0 | 0 |

Paired differences in mean place (+ = worse than the build named second):

| build | compared with | mean place difference ± SE | better / worse / same |
| --- | --- | --- | --- |
| SAFE controller only | e1be9c5 (committed, E1-E4) | +3.28 ± 0.68 | 28 / 57 / 5 |
| kit example bot | e1be9c5 (committed, E1-E4) | +2.94 ± 0.64 | 27 / 59 / 4 |
| f1: committed + armor + endgame point | e1be9c5 (committed, E1-E4) | +0.11 ± 0.48 | 33 / 25 / 32 |
| e1be9c5 (committed, E1-E4) | 80b7597 (before E1-E4) | -0.08 ± 0.58 | 30 / 37 / 23 |
| f1: committed + armor + endgame point | f3: f1 without E1-E4 | +0.66 ± 0.53 | 33 / 28 / 29 |
| g1: f1 + first round of later fixes | g3: g1, ENGAGE_MODEL live | -0.20 ± 0.55 | 36 / 36 / 18 |
| k1: final tree | k2: final tree, ENGAGE_MODEL live | -0.10 ± 0.56 | 35 / 29 / 26 |
| g2: g1, THIRD_PARTY_RULE summed | g1: f1 + first round of later fixes | +0.08 ± 0.48 | 29 / 37 / 24 |
| h1: all fixes except leg execution | e1be9c5 (committed, E1-E4) | +0.34 ± 0.41 | 30 / 27 / 33 |
| k1: final tree | h1: all fixes except leg execution | +0.42 ± 0.48 | 25 / 37 / 28 |
| k1: final tree | e1be9c5 (committed, E1-E4) | +0.77 ± 0.55 | 29 / 33 / 28 |

What they say:

- **Baseline.** The SAFE controller alone and the kit's example bot both place about 9th; the full
  pipeline places about 6th. That is roughly three places, five standard errors. Nothing smaller than
  about one place can be told apart with 90 matches (the paired standard error is 0.4-0.6).
- **Armor and endgame-point fixes (f1).** No measurable change, as expected: armor only matters once the
  bot holds armor upgrades, and no 16-mech match here lasts to 375 s (latest end 348 s).
- **E1 + E2 (`ENGAGE_MODEL`).** Three paired runs put their effect on mean place at -0.27 +- 0.58,
  +0.66 +- 0.53 (that run also removed E3/E4) and -0.20 +- 0.55; the mean is +0.06 +- 0.32. They cut
  fight time from about 35% to 15% of the match and raise flee time from 3% to 18%, with the same result.
- **Third-party rule (`THIRD_PARTY_RULE` = "summed").** +0.64 +- 0.38 and +0.08 +- 0.48, fewer wins
  both times. LOSE (c) carries about 10% of X3 decisions; (a) 30%, (d) 25% and (g) 23% carry most.
- **Never initiating (`ALIVE_ENDGAME_N` = 16).** +0.23 +- 0.39: most fights are already self-defense
  or CORNERED.
- **Zone deaths.** With the old size rule (every cell near a wall closed once the mech has a cube) the
  bot died to the zone in 1-5 of 90 matches, usually while placed 2nd-5th. The final tree (k1) has none.
- **The final tree is not better here.** k1 places 6.58 against 5.81 for the committed build
  (+0.77 +- 0.55, 1.4 standard errors: not significant, but the wrong direction): the same 29 wins,
  more drops taken (50 against 40) and no zone deaths, but fewer top-3 finishes (36 against 43) and more
  deaths before 60 s (29 against 24). The tree before the leg-execution change (h1) sat between the two
  (6.16). In `--peace` the final tree is +0.81 +- 1.72 with two zone deaths before 375 s against none.
  The defects fixed are real and each has a test that fails without its fix (climbs through the arbiter
  1 of 14 -> 14 of 14), so the likely reading is that working climbs and jumps change where the bot goes
  (roofs, drops) and that this is not a gain against these opponents. That has not been traced.

### What the batches cannot reach

No 16-mech match here lasts to the endgame (375 s), and two-mech duels end in a kill too. Two tools
cover the late game:

- `--peace`: weapons and drones do no damage to mechs, the zone still does. All 16 mechs walk the whole
  match, crowd the final point and die in zone order, so placement measures rotation and endgame
  positioning only. Compare builds in separate runs on the same seeds (one build per run): with several
  builds in one match each gets a different spawn, and spawn decides more than the build does.
- `tools/tracebot.py`: the bot plus one stderr line every 2 s of match time (position, label, leg
  owner, move, boost, slack, stuck step, drops, path jobs) and one per finished path search. Every
  late-game defect found on 2026-11-21 came from reading such a trace of one bad match: the closed tight cells, the drop timeout that could not fire, CAPTURE's straight line, the
  45 m hover, the zone field that never reached the bot.

`--peace`, 16 matches on seeds 800-815, one of our builds per match against 15 x `scripted:mix`:

| build | matches | mean place | wins | top 3 | died to the zone before 375 s | median distance from the final point at death (after 390 s) |
| --- | --- | --- | --- | --- | --- | --- |
| e1be9c5 (committed) | 16 | 7.62 | 3 | 5 | 0 | 7.2 m |
| final tree | 16 | 8.44 | 5 | 6 | 2 | 7.8 m |

Paired difference (final − committed) over 16 seeds: +0.81 ± 1.72 places (6 better, 8 worse, 2 same).

Peace mode is harsher than a real match in one way: enemies that cannot be killed keep charging and
firing, so the bot's flee and cornered states run for the whole match.
