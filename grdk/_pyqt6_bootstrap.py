"""
PyQt6 bootstrap for Orange Canvas.

AnyQt 0.2.1 has native PyQt6 support in all its shim modules.
This module simply sets ``QT_API=pyqt6`` and validates that AnyQt
commits to the PyQt6 backend before Orange is imported.

Call :func:`install` **before** any ``from AnyQt.Qt*`` or Orange import.
"""

import os


def install():
    """Select PyQt6 as the Qt backend for AnyQt/Orange.

    Must be called **before** any ``from AnyQt.Qt*`` or Orange import.
    """
    os.environ["QT_API"] = "pyqt6"

    import AnyQt._api as _api

    if _api.USED_API != _api.QT_API_PYQT6:
        raise RuntimeError(
            f"Expected AnyQt to commit to pyqt6, got {_api.USED_API!r}"
        )

    # Prevent segfaults during Python interpreter shutdown.
    # During Py_FinalizeEx, sip iterates wrapped C++ objects and calls
    # sip_api_get_address on wrappers whose C++ side Qt already deleted,
    # causing a null-dereference segfault.  We register an atexit handler
    # that explicitly shuts down the QApplication *before* Python's module
    # finalisation starts destroying things in arbitrary order.
    import atexit

    def _cleanup_qt():
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.closeAllWindows()
            from PyQt6 import sip
            sip.delete(app)

    atexit.register(_cleanup_qt)
