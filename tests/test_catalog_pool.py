# -*- coding: utf-8 -*-
"""
Tests for grdk.catalog.pool — ThreadExecutorPool.

Author
------
Duane Smalley, PhD
duane.d.smalley@gmail.com

Created
-------
2026-02-06
"""

import pytest
from unittest import mock

from grdk.catalog.pool import ThreadExecutorPool


class TestThreadExecutorPool:

    def test_submit_returns_future(self):
        pool = ThreadExecutorPool(max_workers=2)
        try:
            # Submit a simple callable
            future = pool._executor.submit(lambda: 42)
            assert future.result(timeout=5) == 42
        finally:
            pool.shutdown(wait=True)

    def test_shutdown_is_safe(self):
        pool = ThreadExecutorPool(max_workers=1)
        pool.shutdown(wait=True)
        # Should not raise

    def test_submit_download_pip(self):
        pool = ThreadExecutorPool(max_workers=1)
        try:
            # Mock subprocess.run to avoid actually installing
            with mock.patch(
                'grdk.catalog.pool.subprocess.run'
            ) as mock_run:
                mock_run.return_value = mock.Mock(
                    returncode=0, stdout='installed', stderr=''
                )
                future = pool.submit_download("fake-package")
                result = future.result(timeout=10)
                assert result.returncode == 0
        finally:
            pool.shutdown(wait=True)
