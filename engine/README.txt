================================================================================

Compile Notes
-------------

To generate a compatible executable with the game interface, execute:
    pyinstaller --onefile --name mbo main.py

The executable will be generated in dist/mbo.exe

================================================================================

Runtime Notes
-------------

The engine reads one command per line from stdin and answers on stdout:

        name        - print the name of the game engine (mbo).
        print       - print the board.
        exit/quit   - quit the game.
        black XXXX  - place the black stone(s) on XXXX, X is a letter from A to S.
        white XXXX  - place the white stone(s) on XXXX.
        next        - play the next move for the side to move.
        move XXXX   - the opponent played XXXX; play our reply.
        new black   - start a new game playing black (plays the first move).
        new white   - start a new game playing white.
        new xxx     - start a new game with no side assigned.
        depth d     - set the time limit per turn to d seconds (at most 25).
        vcf, unvcf  - accepted and ignored.
        help        - print this help.

The first letter of a cell is its row and the second its column, as in the GUI.
