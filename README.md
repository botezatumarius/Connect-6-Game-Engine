# Connect 6 Game Engine

Practice 1 for *Techniques in Artificial Intelligence* (University of Alicante).
`mbo` is a Connect6 engine that uses alpha-beta search with iterative deepening under a time
limit, threat-based move generation, threat-space search, a transposition table, an opening book
and an evaluation function whose weights were tuned with a genetic algorithm. The report is
`docs/report-Marius Botezatu.pdf`.

## Repository structure

| Folder         | Contents |
|----------------|----------|
| `engine/`      | The engine. It started from the course template; `main.py` is the entry point. |
| `training/`    | Genetic algorithm that tunes the evaluation weights, and the opening book builder. |
| `experiments/` | Match runner, certified test positions, test battery, tournament and parameter studies. |
| `docs/`        | Report (`report.md`, the PDF and its figures). |
| `gui/`         | The ConnectMore GUI and tournament tool, plus the Cloudict engines in `gui/engines/`. Used unchanged. |

## Running the engine

The engine talks over stdin/stdout with the GUI protocol (`name`, `new`, `black`, `white`, `next`,
`move`, `depth`, `print`, `help`, `quit`). `depth d` sets the time limit to *d* seconds per turn.

```bash
cd engine
python main.py
```

The GUI launches engines as executables. Build ours with PyInstaller, then load
`engine/dist/mbo.exe` as the black or white engine:

```bash
cd engine
pyinstaller --onefile --name mbo main.py
```

The GUI must be started from inside `gui/`, because it loads its images by relative path:

```bash
cd gui
python ConnectMore.py
```

## Reproducing the training and the experiments

Every script writes CSV files next to itself, in a `results/` folder.

```bash
python training/genetic.py --generations 25 --population 16
python training/genetic.py --progress
python training/build_book.py --seconds 30
python experiments/positions.py
python experiments/battery.py v1 v2 v3 v4
python experiments/battery.py v5 --out battery_v5
python experiments/tournament.py v1 v2 v3 v4 cloudict
python experiments/tournament.py v4 v5 cloudict --openings 4 --out final
python experiments/sweeps.py
python experiments/plots.py
python docs/build_pdf.py
```

`v1`–`v5` are the engine at the end of sessions 1–5 (git tags `session-1` to `session-5`),
checked out automatically into `experiments/versions/`. The engine needs only the Python standard
library; the tools need the packages of `requirements.txt` (`pip install -r requirements.txt`).

## Upstream sources

The course repositories were copied without their git history:

- `engine/` started from [felixem/Connect6Engine](https://github.com/felixem/Connect6Engine) at commit `dd74352`
- `gui/` is [felixem/Connect6GUI](https://github.com/felixem/Connect6GUI) at commit `dc43f5a` (BSD-style licence, see `gui/LICENSE.txt`)
