# V3 Final Holdout Boundary — RESERVED BEFORE ANY DEVELOPMENT

**Recorded:** 2026-08-26T21:32:09Z, before fetching any expanded historical data, before writing any V3 strategy code, before any parameter selection.

**Development period (usable for everything — Stages 1-6):** all cached history from each symbol's earliest available data through **2026-05-28 23:59:59 UTC**.

**Final holdout period (Stage 7 ONLY, evaluated exactly once, after every other decision is locked):** **2026-05-29 00:00:00 UTC through 2026-08-26** (the last ~90 days of available history as of this research cycle).

**Rules governing this boundary:**
1. No chart data, no metric, no equity curve, no trade from the holdout period is read, printed, plotted, or inspected in any way until a candidate has already passed Stages 1-6 in full.
2. No parameter, threshold, or design decision in any V3 strategy may be adjusted after seeing holdout results.
3. This boundary is not moved later because results are disappointing. If nothing survives to Stage 7, the holdout is simply never used, and that is reported as-is.
4. This file is the single source of truth for the boundary. Any script computing it programmatically must match `2026-05-28T23:59:59Z`.

`HOLDOUT_START_TS_MS = 1780012800000` (2026-05-29T00:00:00Z, computed once here for scripts to import rather than re-deriving).
