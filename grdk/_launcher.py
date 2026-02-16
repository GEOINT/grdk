"""Launch Orange Canvas with PyQt6 backend."""

import sys

from grdk._pyqt6_bootstrap import install

install()

from Orange.canvas.__main__ import main as canvas_main


def main():
    canvas_main()


if __name__ == "__main__":
    main()
