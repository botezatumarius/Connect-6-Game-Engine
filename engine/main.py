"""Entry point of the mbo Connect6 engine (also the PyInstaller target)."""
from game_engine import GameEngine


def main():
    """Run the engine on stdin/stdout until quit."""
    GameEngine().run()


if __name__ == "__main__":
    main()
