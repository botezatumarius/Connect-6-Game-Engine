# Connect 6 Game Engine

Practice 1 for *Techniques in Artificial Intelligence* (University of Alicante).
This is a Connect6 game engine that uses adversarial search (Minimax, then Alpha-Beta and further optimisations).

## Repository structure

| Folder    | Contents |
|-----------|----------|
| `engine/` | Our game engine. It started from the course template and is where all development happens. |
| `gui/`    | The ConnectMore GUI and tournament tool, plus the prebuilt Cloudict opponent engines in `gui/engines/`. We use it but don't modify it. |

## Running

The engine talks over stdin/stdout. Start it in a terminal:

```bash
cd engine
python main.py
```

The GUI needs Python 3 with Tk. Run it from inside `gui/` because it loads images by relative path:

```bash
cd gui
python ConnectMore.py
```

The GUI launches engines as executables. To play against our engine in the GUI, build it with PyInstaller first:

```bash
cd engine
pyinstaller --onefile main.py
```

## Upstream sources

Both folders were copied from the course repositories without their git history:

- `engine/` from [felixem/Connect6Engine](https://github.com/felixem/Connect6Engine) at commit `dd74352`
- `gui/` from [felixem/Connect6GUI](https://github.com/felixem/Connect6GUI) at commit `dc43f5a` (BSD-style licence, see `gui/LICENSE.txt`)
