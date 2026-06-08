"""Command-line entry points and one-off runner scripts for the project.

Each module here is meant to be executed directly (``python scripts/<name>.py``)
or imported as part of the ``scripts`` package (e.g. by the test-suite). The
actual modelling/data logic lives in the :mod:`src` package; these scripts are
thin orchestration layers around it.

Because the scripts live in a sub-directory, each one inserts the project root
onto ``sys.path`` at import time so that ``import src...`` resolves regardless of
the current working directory.
"""
