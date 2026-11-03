"""Unit tests may build development stand-ins and tiny random models; nothing else may.

See adarq_flow/standins.py: this flag is set ONLY here, inside the pytest process (and
inherited by subprocesses the tests start).
"""

import os

os.environ["ADARQ_ALLOW_STAND_INS"] = "1"
