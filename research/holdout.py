"""
Canonical V3 final-holdout boundary. See V3_HOLDOUT_BOUNDARY.md -- this
constant must match that file exactly; it exists so every V3 script imports
one source instead of re-deriving the date and risking drift.

Reserved 2026-08-26T21:32:09Z, before any V3 data expansion or development.
"""
HOLDOUT_START_TS_MS = 1780012800000   # 2026-05-29T00:00:00Z
