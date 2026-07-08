"""
Multi-model contest tool with HTML output (mobile-friendly).
Live tables over the IMD 0830->0830 IST window:
  1. RAINFALL      - sum of hourly precip (21 contest stations)
  2. TN RAINFALL   - sum of hourly precip (15 TN highest-rainfall places)
  3. MAX TEMP      - max of hourly temp (10 TN stations)
Plus a BACKTEST (BACKTEST_DATE, default = yesterday IST) for a past day via the
Historical Forecast API.

Output: renders styled HTML tables INLINE in Colab (readable on a phone) and also
saves 'contest_report.html' you can open or share. Color coding: spread shaded
green->red (model agreement), rainfall means shaded blue by intensity.

ECMWF, GFS, ICON via Open-Meteo. No API key, no MCP. Colab has the deps.
WINDOW anchored to the current IST date: DAY_OFFSET=0 -> today's contest window.
"""

import requests
import pandas as pd
import json, os, time, math
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# Set True to print run diagnostics (drive mount, surge-regime decisions, saved-file
# path, module-import notices) above the rendered report. Default False = clean output;
# the surge/regime info is already shown in the report's guidance panel either way.
VERBOSE = False


def _log(*args, **kwargs):
    """print() only when VERBOSE. Keeps the routine chatter out of the cell output
    while leaving real error prints (which don't use this) untouched."""
    if VERBOSE:
        print(*args, **kwargs)

# -- RAINFALL stations: name -> (lat, lon) ------------------------------------
STATIONS = {
    "Kakki Dam (KL)":          (9.36, 77.16),
    "Peermade (KL)":           (9.58, 76.97),
    "Kakkayam Dam (KL)":       (11.54, 75.92),
    "Naladi (KA)":             (12.39, 75.49),
    "Hulikal (KA)":            (13.73, 75.01),
    "Kollur (KA)":             (13.865, 74.81),
    "Darbandora (GA)":         (15.36, 74.18),
    "Amboli (MH)":             (15.95, 73.99),
    "Gaganbawada (MH)":        (16.54, 73.83),
    "Tamini/Tamhini (MH)":     (18.46, 73.42),
    "Mahabaleshwar (MH)":      (17.92, 73.66),
    "Matheran (MH)":           (18.99, 73.27),
    "Kaprada (GJ)":            (20.43, 73.17),
    "Dharamshala (HP)":        (32.22, 76.32),
    "Sama (UK)":               (30.13, 80.01),
    "Mangan (SK)":             (27.51, 88.53),
    "Mawsynram (ML)":          (25.30, 91.58),
    "Alipurduar (WB)":         (26.49, 89.53),
    "Beki Mathanguri (AS)":    (26.77, 90.96),
    "Pasighat (AR)":           (28.07, 95.33),
    "Long Island (AN)":        (12.36, 92.93),
}

# -- TN highest-rainfall places: name -> (lat, lon) ---------------------------
TN_RAIN_STATIONS = {
    "Avalanche":            (11.29, 76.57),
    "Chinnakallar":         (10.30, 77.03),
    "Ooty":                 (11.41, 76.70),
    "Pandalur":             (11.54, 76.32),
    "Nalumukku":            (8.63, 77.32),
    "Hosur":                (12.74, 77.83),
    "Kilcheruvai":          (12.55, 79.20),   # VERIFY
    "Polur":                (12.51, 79.12),
    "Yercaud":              (11.78, 78.21),
    "Ponnai":               (12.90, 78.95),   # VERIFY
    "Kodaikanal Pachalur":  (10.18, 77.55),
    "Pudukkottai":          (10.38, 78.82),
    "Thirupuvanam":         (9.85, 78.27),
    "Colachel":             (8.17, 77.26),
    "Pilavakkal Dam":       (9.50, 77.55),
}

# -- MAX-TEMP stations (Tamil Nadu): name -> (lat, lon) -----------------------
TEMP_STATIONS = {
    "Madurai":        (9.93, 78.12),
    "Tiruttani":      (13.18, 79.61),
    "Vellore":        (12.92, 79.13),
    "Cuddalore":      (11.75, 79.77),
    "Meenambakkam":   (12.99, 80.18),
    "Nungambakkam":   (13.06, 80.24),
    "Palayamkottai":  (8.71, 77.73),
    "Tondi":          (9.74, 79.02),
    "Nagapattinam":   (10.77, 79.84),
    "Tiruchi":        (10.79, 78.70),
}

MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]
# NOTE (reverted): "ecmwf_aifs025" was added here briefly to bring AIFS (ECMWF's
# real graph-neural-network model) into the ensemble, but it doesn't work on
# this general multi-model /v1/forecast endpoint - confirmed empirically
# (06-Jul run: the aifs column came back all-empty, and its meta.json lookup
# silently matched some unrelated static archive instead of erroring, showing
# a run from Feb 2025). AIFS is actually served through ECMWF's SEPARATE
# /v1/ecmwf endpoint, not this shared one. Adding it properly means a second,
# independent fetch call merged into the tables - real feature, more plumbing,
# not a one-line change. Worth doing right rather than half-wired again.

# -- IMD TN/Puducherry/Karaikal bulletin stations: name -> (lat, lon) ---------
# Matches the 30-station "MAXIMUM TEMP" bulletin table (mausam.imd.gov.in) so
# rain-count guidance ("Obs: N stations") lines up with what IMD reports on.
# Where a station overlaps an existing dict (Chennai Nungambakkam/AP, Cuddalore,
# Nagapattinam, Palayamkottai, Tiruchirapalli AP, Tiruthani, Tondi, Vellore,
# Madurai City, Udagamandalam), coordinates are reused for consistency.
IMD_TN_STATIONS = {
    "Adiramapattinam":       (10.35, 79.35),
    "Chennai Nungambakkam":  (13.06, 80.24),
    "Chennai AP":            (12.99, 80.18),
    "Coimbatore AP":         (11.03, 76.97),
    "Coonoor":               (11.35, 76.80),
    "Cuddalore":             (11.75, 79.77),
    "Dharmapuri":            (12.13, 78.16),
    "Erode":                 (11.34, 77.73),
    "Kanyakumari":           (8.08, 77.57),
    "Karur Paramathi":       (10.96, 78.08),
    "Karaikal":              (10.92, 79.83),
    "Kodaikanal":            (10.24, 77.49),
    "Madurai City":          (9.93, 78.12),
    "Madurai Airport":       (9.83, 78.09),
    "Nagapattinam":          (10.77, 79.84),
    "Namakkal_AMFU":         (11.22, 78.17),
    "Palayamkottai":         (8.71, 77.73),
    "Pamban":                (9.28, 79.21),
    "Parangipettai":         (11.49, 79.77),
    "Puducherry":            (11.94, 79.83),
    "Salem":                 (11.66, 78.15),
    "Thanjavur":             (10.79, 79.14),
    "Tirupattur AWS":        (12.50, 78.57),
    "Tiruchirapalli AP":     (10.79, 78.70),
    "Tiruthani":             (13.18, 79.61),
    "Tondi":                 (9.74, 79.02),
    "Thoothukudi ARG":       (8.77, 78.13),
    "Udagamandalam":         (11.41, 76.70),
    "Valparai":              (10.33, 76.95),
    "Vellore":               (12.92, 79.13),
}
IMD_RAIN_THRESH = 2.5   # mm - IMD's own threshold for "rain observed" at a station

# --- Avalanche surge check (850 hPa wind + moisture) -------------------------
# Avalanche (TN_RAIN_STATIONS) sees its best rainfall during an active westerly
# monsoon surge off the Arabian Sea hitting the Nilgiris windward slope.
# SURGE_DIR band = SW-WSW (typical monsoon 850hPa flow); SURGE_WIND_MIN is a
# rough "moderate surge" cutoff, loosely anchored to IMD's own monsoon-onset
# wind criteria (~15-20kt at 925hPa) stepped up slightly for the 850hPa level.
AVALANCHE_LATLON = TN_RAIN_STATIONS["Avalanche"]
SURGE_DIR_MIN, SURGE_DIR_MAX = 200, 280   # degrees, "from" direction (SW-WNW)
SURGE_WIND_MIN = 8.0                       # m/s at 850hPa - rough moderate-surge cutoff

LABELS = {"ecmwf_ifs025": "ecmwf", "gfs_seamless": "gfs", "icon_seamless": "icon"}
META_DOMAINS = {
    "ecmwf_ifs025": "ecmwf_ifs025",
    "gfs_seamless": "ncep_gfs025",
    "icon_seamless": "dwd_icon",
}
# Single Runs API needs models with ONE well-defined run cycle. "gfs_seamless"
# and "icon_seamless" are blends of sub-models with different run schedules
# (e.g. ICON Global vs ICON-EU/D2), so a fixed run= timestamp doesn't map to
# a single archived run for them - this caused a 400 error. Use the plain
# global models instead, just for single-run fetches.
SINGLE_RUN_MODELS = ["ecmwf_ifs025", "gfs_global", "icon_global"]
SINGLE_RUN_LABELS = {"ecmwf_ifs025": "ecmwf", "gfs_global": "gfs", "icon_global": "icon"}

# Maps a Single-Run-API model id back to the "seamless" id used everywhere
# else in the script (LABELS, MODELS, build_table/build_temp_table column
# lookups), so downstream code doesn't need to know two different model lists.
# FIX (2026-07-07): fetch_single_run()/fetch_single_run_daily() previously
# requested `MODELS` (which includes gfs_seamless/icon_seamless) against the
# Single Runs API, which only accepts the plain global models with a fixed
# run= timestamp. That mismatch was silently producing a non-JSON response
# body, surfacing as `JSONDecodeError: Expecting value: line 1 column 1` and
# skipping ALL pinned-run backtests. Now we request SINGLE_RUN_MODELS and
# relabel the returned hourly keys back to the seamless names immediately.
_SINGLE_RUN_TO_SEAMLESS = {
    "ecmwf_ifs025": "ecmwf_ifs025",
    "gfs_global": "gfs_seamless",
    "icon_global": "icon_seamless",
}

DAY_OFFSET   = 0
WINDOW_START = 8.5
TIMEZONE     = "Asia/Kolkata"
CONF_WEIGHT  = 0.25     # (kept for reference; temp score uses it)
RAIN_RANK    = "max"    # rank rainfall by "max" (highest model; best for orographic spikes) or "mean"
USE_CLIMO_FLOOR = True   # lift known orographic monsters to a seasonal floor when models lowball them
CLIMO_FLOOR = {          # peak-monsoon daily expectation (mm) for stations the models are blind to.
    "Mawsynram (ML)": 100,   # June daily avg ~85-100mm; near-daily national topper in peak monsoon
    # Add others only if models PERSISTENTLY lowball them (check the backtest), e.g.:
    # "Alipurduar (WB)": 40, "Pasighat (AR)": 40, "Mangan (SK)": 40,
}

# --- Per-station rainfall bias (multiplicative) ------------------------------
USE_RAIN_BIAS   = True      # apply per-station orographic correction -> adds a 'corr' column
RAIN_BIAS_BASE  = "ecmwf"   # model the factor multiplies: "ecmwf"|"gfs"|"icon"|"mean"|"max"
RAIN_FACTOR_MIN, RAIN_FACTOR_MAX = 0.5, 3.0   # tightened from 0.2-8.0: a single noisy seed
                                               # day (e.g. Alipurduar 4.2x, Tamhini 7.3x) was
                                               # amplifying ECMWF's own miss instead of fixing it.
                                               # NOTE (01-Jul seed): Tamini/Tamhini and
                                               # Mahabaleshwar have now hit this 3.0 cap on
                                               # BACK-TO-BACK seed days (both want >3.4-6x
                                               # uncapped). RESOLVED 03-Jul: it persisted a
                                               # third straight day (Mahabaleshwar wanted
                                               # 4.6x on the exact day it won the contest) -
                                               # per-station cap overrides now live in
                                               # RAIN_FACTOR_MAX_STATION below.
RAIN_MIN_BASE   = 2.0       # skip ratio learning when base < this mm (ratio unstable)
RAIN_LEARN_RATE = 0.3       # EMA speed for refine_rain_bias()
# Per-station max-factor overrides for chronic under-forecast orographic monsters.
# These stations pin the global 3.0 cap on exactly the big-surge days they top the
# contest, so the global cap was choking the correction where it matters most.
# Raising the cap ONLY here keeps noisier stations (Alipurduar 4.2, Kakki 3.5 -
# both stored above 3.0 and INTENTIONALLY still clamped to it) from swinging.
# Applied at BOTH apply time (build_table) and learn time (refine_rain_bias).
# NOTE: the west-coast dampen still halves the pull at apply time, so e.g.
# Mahabaleshwar at factor 3.28 -> effective 2.14 on corr. If these keep
# lowballing on active-surge days, the next lever is a per-station dampen
# easing (e.g. 0.75 instead of 0.5 for this trio), not a further cap raise.
RAIN_FACTOR_MAX_STATION = {
    "Mahabaleshwar (MH)":   5.0,   # wanted 4.6x from one run, then 8.2x from the correctly
                                    # window-matched run (187 vs ecmwf 22.8) - worst miss yet
    "Tamini/Tamhini (MH)":  5.0,   # pinned 3.0 on back-to-back seed days
    "Kakkayam Dam (KL)":    5.0,   # wanted 3.7x, then 3.96x on the corrected pairing
    "Chinnakallar":         5.0,   # TN Valparai belt, legendarily under-modeled; stored 4.4
                                    # was silently clamped to 3.0 at apply time until now -
                                    # this makes it take full effect (no west-coast dampen
                                    # applies to TN stations, so watch its first few days)
    "Kakki Dam (KL)":       5.0,   # CONFIRMED 03-Jul (window-matched: 59 vs ecmwf 9.3,
                                    # ratio 6.34x). Was clamped to 3.0 losing real signal;
                                    # EMA now lands at 4.35 with the cap raised. (Earlier
                                    # add/revert of this same station was a date-mismatch
                                    # error on my part - this time the window is confirmed.)
    # Long Island (AN) is close behind (5.56x raw miss, EMA lands at 2.99 - just
    # under the old 3.0 cap) but doesn't need an override THIS round. Add it here
    # if next update pushes it over 3.0.
}

def _factor_cap(name):
    """Effective max factor for a station (per-station override, else global cap)."""
    return RAIN_FACTOR_MAX_STATION.get(name, RAIN_FACTOR_MAX)

# West-coast/Ghats stations: orographic convective rain is too day-to-day noisy for a
# static multiplier to track reliably (same station can get hammered one day, missed
# entirely the next). Dampen the learned factor's pull so ECMWF's raw read stays the
# dominant signal and the factor only nudges it, rather than multiplying ECMWF's own
# miss 3-7x. effective_factor = 1 + (factor - 1) * RAIN_WESTCOAST_DAMPEN
WEST_COAST_STATIONS = {
    "Kakki Dam (KL)", "Peermade (KL)", "Kakkayam Dam (KL)", "Naladi (KA)",
    "Hulikal (KA)", "Kollur (KA)", "Darbandora (GA)", "Amboli (MH)",
    "Gaganbawada (MH)", "Tamini/Tamhini (MH)", "Mahabaleshwar (MH)", "Matheran (MH)",
}
RAIN_WESTCOAST_DAMPEN = 0.5

# --- Surge-CONDITIONAL rainfall factor -------------------------------------
# For a few orographic stations the correction doesn't just vary in magnitude,
# it FLIPS SIGN with the synoptic regime: on an active Arabian-Sea westerly
# surge ECMWF UNDER-catches the windward-slope spike (factor >1), but on a
# quiet day it OVER-catches (factor <1). A single static multiplier is then
# guaranteed wrong in one regime (Avalanche: 0.5x quiet, 1.69x surge - see its
# STATION_RAIN_BIAS note). So for stations listed here we store TWO factors and
# pick per-day from the live 850hPa flow AT THAT STATION, using the same surge
# diagnostic as the Avalanche check. These OVERRIDE the STATION_RAIN_BIAS entry
# ONLY on days the surge state is detected successfully; if the 850 fetch fails
# we silently fall back to the plain STATION_RAIN_BIAS factor so nothing breaks.
USE_SURGE_FACTOR = True
STATION_SURGE_FACTOR = {
    # station: surge factor, quiet factor, 850hPa "from" dir band, min surge wind (m/s)
    # Arabian-Sea westerly surge onto the windward Nilgiri / Anamalai slopes.
    "Avalanche":    {"surge": 1.69, "quiet": 0.5,  "dir": (200, 290), "wind_min": 8.0},
    "Chinnakallar": {"surge": 1.61, "quiet": 1.0,  "dir": (200, 290), "wind_min": 8.0},
    # NOTE: each sub-factor still has only 1-2 samples. Avalanche's split is
    # real (sign flip observed). Chinnakallar's quiet=1.0 is a PLACEHOLDER prior
    # (trust raw ECMWF) - both its samples so far were surge-ish under-forecasts
    # (4.39x seed, 1.61x this event), so its quiet-day ratio is still unobserved.
    # refine_rain_bias() will learn whichever regime each backtest day falls in.
}

# --- Quiet-gate: don't let a knockdown (<1) factor suppress a strong signal ---
# The over-forecasts that justify a <1 factor happen on LOW-rain days (e.g.
# Avalanche ecmwf 27 -> actual 13). There's no evidence ECMWF over-forecasts a
# BIG orographic spike - on those it under-catches (Avalanche ecmwf 112 -> 189).
# So a <1 factor applies fully only while the base is modest, and RELAXES toward
# 1.0 (trust raw ECMWF) as the base climbs. Boosts (>=1) are never gated - we
# want them on big surge days. This is the fix for a quiet/misclassified day
# halving a large ECMWF read (e.g. Avalanche 79mm being cut to 40).
QUIET_GATE = True
QUIET_TRUST_BELOW = 30.0   # mm: at/under this base, a <1 factor applies in full
QUIET_TRUST_ABOVE = 60.0   # mm: at/over this base, factor -> 1.0 (trust ECMWF raw)


def _gate_quiet(f, base):
    """Relax a knockdown factor (<1) toward 1.0 as `base` (mm) rises past the
    trust window. Returns f unchanged for boosts (>=1) or when gating is off."""
    if not QUIET_GATE or f is None or base is None or f >= 1.0:
        return f
    if base >= QUIET_TRUST_ABOVE:
        return 1.0
    if base <= QUIET_TRUST_BELOW:
        return f
    frac = (base - QUIET_TRUST_BELOW) / (QUIET_TRUST_ABOVE - QUIET_TRUST_BELOW)
    return f + frac * (1.0 - f)


# --- Surge CLUSTERS: decide the regime ONCE per region ----------------------
# A single noisy 850 grid point shouldn't flip one station's regime while its
# neighbour disagrees (the Avalanche=quiet / Chinnakallar=surge split). Group
# stations that share a synoptic regime; the 850 flow is POOLED across all
# member points and classified once, then that single regime is handed to every
# member (even one whose own point lacked data). Members must be in
# STATION_SURGE_FACTOR and should share the same dir band / wind_min. A surge
# station NOT listed in any cluster still falls back to per-station detection.
SURGE_CLUSTERS = {
    "Nilgiri/Anamalai": ["Avalanche", "Chinnakallar"],
}
SURGE_CLUSTER_AGG = "mean"   # "mean" = classify from the regional vector-mean flow
                             # (noise-robust, recommended). "max" = surge if ANY member
                             # reads surge (more permissive; use if the mean is washing
                             # out a real surge hitting only part of the cluster).

# --- West-coast surge MOVEMENT (day-over-day migration banner) ---------------
# The heavy west-coast monsoon rain tracks the latitude of the strongest onshore
# 850hPa flow (the offshore-vortex / low-level-jet core), which migrates N-S day
# to day. We read that onshore flow at the west-coast Ghats stations, find where
# it PEAKS today vs yesterday, and print a one-line banner: "surge core moved
# from <station> to <station>" + the stations under today's core. Heuristic (an
# onshore-flow proxy, not a full moisture-flux integral) - a directional hint.
SHOW_SURGE_MOVEMENT = True
WESTCOAST_SURGE_DIR = (200, 300)   # 850hPa "from" band counted as onshore (Arabian Sea)
WESTCOAST_SURGE_MIN_IDX = 4.0      # below this peak index, call the whole coast "quiet"
WEST_COAST_CITIES = {              # nearby landmarks for a human-readable latitude label
    "Mumbai": 19.08, "Ratnagiri": 16.99, "Panaji/Goa": 15.49,
    "Mangalore": 12.87, "Kochi": 9.97,
}


# factor = typical (actual / base). >1 model UNDER-forecasts (orographic uplift),
# <1 OVER-forecasts. Multiplicative (not additive) because orographic errors scale
# with rainfall, not a fixed offset. GFS/ICON often print 0.0 at the wettest Ghats
# spots, so the default base is ECMWF, not the mean.
# Updated 03-Jul-2026 AIRF (EMA alpha=0.3), CORRECTED PAIRING. The actuals
# (Mahabaleshwar 187, Tamini 170, Matheran 117...) are the 2nd 08:30 -> 3rd
# 08:30 IST window - confirmed directly by the user. That matches the
# "Backtest rainfall (FROZEN run, gfs 2026-07-01 18Z)" table's ecmwf bases
# (67.5, 47.1, 22.8...), NOT the earlier corr table's bases (74.4, 87.9,
# 40.5...) used in the first EMA pass below - that first pass was scored
# against the wrong day's forecast and has been discarded. This update starts
# fresh from the pre-EMA baseline and applies ONE correct step.
# Mahabaleshwar's raw ECMWF miss this round was 8.2x (187 vs 22.8) - the
# worst yet - which alone pushed its EMA value to 4.36, now representable
# thanks to the raised per-station cap. Kakki Dam's raw miss (59 vs 9.3,
# 6.3x) similarly pushed it to 4.35, which NEEDS the same cap treatment -
# added below with solid, window-matched evidence this time. Long Island's
# raw miss (80 vs 14.4, 5.6x) lands its EMA at 2.99 - just under the global
# cap for now, but worth watching; it'll likely need the same override next
# update if this pattern continues.
STATION_RAIN_BIAS = {
    # --- TN stations (seeded 28-Jun TNRF: actual / ecmwf) ---
    "Chinnakallar": 3.56,   # EMA down from 4.4: TNRF actual 138 vs ecmwf 85.5 -> ratio 1.61;
                            # 0.7*4.4 + 0.3*1.61. cap 5.0 (RAIN_FACTOR_MAX_STATION) so not clamped.
                            # Still sits well above THIS event's 1.6x - seed day wanted 4.39x, so
                            # the station is regime-volatile; watch it converge over more days.
    "Nalumukku":    2.1,    # 26 vs 12.3
    "Pudukkottai":  1.2,    # 19 vs 15.6
    "Avalanche":    0.86,   # EMA up from 0.5: TNRF actual 189 vs ecmwf 111.9 -> ratio 1.69;
                            # 0.7*0.5 + 0.3*1.69. HIGHLY volatile: 0.5x on a quiet day (13 vs 27),
                            # 1.69x on this surge day (189 vs 112). A static factor can't win here -
                            # the sign of the correction flips with the surge. Prime candidate for
                            # a surge-conditional factor keyed off analyze_avalanche_surge().
    "Pandalur":     0.3,    # 8  vs 28.2  (over-forecast)
    # --- Main contest stations (EMA-refined 03-Jul AIRF, corrected pairing) ---
    "Pasighat (AR)":        2.2,   # unchanged, no actual this round
    "Alipurduar (WB)":      4.2,   # unchanged, no actual this round (still clamps to 3.0 - intentional)
    "Tamini/Tamhini (MH)":  2.93,  # 170 vs ecmwf 47.1 -> ratio 3.61
    "Gaganbawada (MH)":     1.96,  # 86 vs ecmwf 41.4 -> ratio 2.08
    "Kakkayam Dam (KL)":    2.73,  # 82 vs ecmwf 20.7 -> ratio 3.96
    "Mahabaleshwar (MH)":   4.36,  # 187 vs ecmwf 22.8 -> ratio 8.20 (worst miss yet);
                                     # only representable thanks to the raised per-station cap
    "Darbandora (GA)":      1.63,  # 68.4 vs ecmwf 24.6 -> ratio 2.78
    "Matheran (MH)":        2.04,  # 117 vs ecmwf 67.5 -> ratio 1.73
    "Long Island (AN)":     2.99,  # 80 vs ecmwf 14.4 -> ratio 5.56; right at the old 3.0 cap,
                                     # watch closely next update
    "Amboli (MH)":          1.77,  # 69 vs ecmwf 34.8 -> ratio 1.98
    "Mangan (SK)":          1.0,   # PENDING: actual 21.3, ecmwf base not visible in the shared table
    "Kakki Dam (KL)":       4.35,  # 59 vs ecmwf 9.3 -> ratio 6.34; NEW cap override added below -
                                     # was hard-clamped to 3.0 (losing real signal) before this
    "Beki Mathanguri (AS)": 0.7,   # unchanged, no actual this round
    "Hulikal (KA)":         1.27,  # 78 vs ecmwf 44.7 -> ratio 1.75
    "Kollur (KA)":          0.59,  # 39.8 vs ecmwf 38.4 -> ratio 1.04; near spot-on raw ECMWF
    "Sama (UK)":            3.0,   # unchanged, no actual this round
    "Naladi (KA)":          1.61,  # 58.8 vs ecmwf 25.2 -> ratio 2.33
    "Kaprada (GJ)":         0.86,  # 82 vs ecmwf 76.2 -> ratio 1.08; near spot-on raw ECMWF
    "Dharamshala (HP)":     1.34,  # unchanged, no actual this round
    # Peermade (KL): actual 43 but no ecmwf base visible in the shared table -
    # finish via REFINE on the next run (see RAIN_ACTUALS below).
    # Mawsynram: corr = ECMWF * factor so it scales proportionally with Euro
    # rather than hard-flooring at 100mm even when ECMWF shows a quiet day.
    # Climo floor (100mm) still applies to the 'max' column as a safety net.
    "Mawsynram (ML)":       3.0,   # unchanged, no actual this round
}
# 03-Jul AIRF actuals (window confirmed: 2nd 08:30 -> 3rd 08:30 IST). Most
# stations were already EMA'd directly into STATION_RAIN_BIAS above, scored
# against the correctly-matched backtest table. ONLY Peermade and Mangan
# remain here - their ecmwf base wasn't visible in the shared table. To
# finish them: flip REFINE_BIAS=True, set BACKTEST_DATE="2026-07-02" (the
# frozen run that predicts this exact window), run once, paste the printed
# factors for these two ONLY, then clear this dict and flip REFINE_BIAS back
# off. Do NOT re-add the stations already resolved above - a second EMA pass
# would double-count them.
# TNRF same-window update: Avalanche (189 vs ecmwf 111.9) and Chinnakallar
# (138 vs ecmwf 85.5) were EMA'd DIRECTLY into STATION_RAIN_BIAS above from the
# backtest table (bases read straight off it), so they are NOT listed here -
# adding them would double-count on the next REFINE pass.
RAIN_ACTUALS = {
    "Peermade (KL)": 43, "Mangan (SK)": 21.3,
}

TEMP_BIAS    = 0.0      # degC added to temp (retune vs actuals; daily-max now captures the peak)
TEMP_RANK    = "score"  # rank temp table by: "score" (mean adjusted for model spread,
                         # best for picking), "gfs", or "mean"
# Per-station correction (degC) added to temp = actual - model. Inland runs cool
# (positive), coastal runs hot (negative). SEEDED FROM ONE backtest day - refine by
# AVERAGING (actual - forecast) over several days. Note: this makes that one day's
# backtest match trivially; judge it on FUTURE days, not the seed day.
#
# 01-Jul-2026 (2nd pass, 17:30 IST run): proper EMA update (alpha=0.3) applied
# on top of the morning's one-time bootstrap injection, scored against the
# 17:30 IST IMD obs vs that morning's 'score' forecast. Tondi's injection had
# overshot (fcst 38.0 vs actual 36.7, +1.3 too warm) and correctly pulled back
# down; every other station nudged up further (still running cold). This
# replaces the earlier morning bootstrap values as the new baseline.
# 03-Jul-2026 (1730 IST IMD obs vs that morning's live 'score' forecast, same
# methodology as the 01-Jul pass). Tiruchi and Palayamkottai REVERSED
# direction - both had been running cold and are now over-forecasting by
# 1.6-1.8 deg, so their bias pulled back down. Vellore's cold-bias widened
# sharply (+1.9 resid) and got the biggest bump. Everyone else nudged up
# further (still net cold).
# 04-Jul-2026 (1730 IST IMD obs vs morning 'score'). Split day: inland/southern
# hot spots ran WARMER than modelled (Madurai +2.2, Tondi +2.0 -> big bumps up),
# northern/coastal ran COOLER (Vellore -1.9, Cuddalore -1.1 -> pulled down).
# Tiruchi spot-on (held 1.9). Meenambakkam & Nungambakkam HELD (0.1 / 0.0):
# raw ECMWF nailed both Chennai stations (32.2 actual), so no station-bias to
# chase - the small score warmth was gfs/icon + the +0.7 ecmwf nudge, a
# model-spread issue not a station one. CAUTION: Vellore ran 2.8 below its OWN
# normal max today (cloud/rain anomaly, not a warm-bias), so its 1.5->0.9 pull
# is the least-trustworthy move here - half-step it if it whipsaws.
# 07-Jul-2026 (1730 IST IMD obs vs that morning's live 'score' forecast, EMA
# alpha=0.3). Madurai/Tiruchi/Tiruttani/Cuddalore/Vellore had all drifted too
# WARM (overshooting actuals by 0.9-1.7 deg after several up-only EMA passes
# chasing earlier cold-bias days) and got pulled back down. Chennai stations
# + Nagapattinam + Palayamkottai were all within +0.3 (essentially spot-on,
# tiny nudge up). Tondi's residual (+2.1) was NOT applied to its base bias
# here - that miss traces to the late (16:00 IST) sea-breeze onset not
# triggering the no-breeze bump at all (binary check), not to a wrong base
# bias. Folding the full +2.1 into Tondi's flat bias would double-count once
# the graded late-breeze bump below is applied, and would wrongly inflate
# Tondi's forecast on a normal-onset day. See _seabreeze_lateness_bump().
# 08-Jul-2026 EMA update, scored against a PINNED 18Z run (2026-07-06T18:00Z)
# vs today's 1730 IST actuals - the first real backtest using the fixed
# backtest_temp_single_run() / fetch_single_run_daily() (see the workaround
# note on fetch_single_run_daily for why the daily endpoint 400'd before this).
# This is a cleaner signal than the usual live-forecast-vs-actual comparison:
# it isolates ONE specific evening run's skill rather than whatever run
# happened to be freshest when the report was generated. Madurai/Cuddalore/
# Nagapattinam/Palayamkottai got the biggest upward pulls (18Z run still ran
# meaningfully warm vs actual at these); Tiruchi/Tiruttani/Vellore pulled back
# down slightly; Chennai stations barely moved (already close). Tondi's small
# +0.8 resid folded in normally this time (no late-breeze complication noted
# for this run).
# 08-Jul-2026 TARGETED Madurai-only follow-up (same day, separate from the EMA
# pass above): two independent same-direction misses both said Madurai's
# forecast runs warm - the 18Z backtest (-1.1) AND today's live forecast
# (-0.8) vs the SAME day's actual (38.4). That's consistent signal, not noise,
# so Madurai got one extra targeted pull (avg of both residuals, one EMA step:
# 2.6 + 0.3*(-0.95) = 2.3) on top of the general pass, while every other
# station was left untouched.
# SUPERSEDED 08-Jul-2026 (later same day): the 2.3 flat pull above was itself
# replaced after a proper 28-day 18Z backtest (madurai_temp_backtest.py)
# showed the per-model biases at Madurai differ enough (ecmwf +2.0, gfs +2.7,
# icon +2.3) that a single flat number was structurally the wrong tool - see
# MODEL_STATION_BIAS above, which now carries Madurai's correction instead.
# Left at 0.0 here (not deleted) so the key stays present/documented; do NOT
# set this back to a nonzero value without removing the MODEL_STATION_BIAS
# Madurai entries first, or Madurai will get double-corrected.
STATION_TEMP_BIAS = {
    "Tiruchi": 1.2, "Madurai": 0.0, "Palayamkottai": 1.2, "Meenambakkam": 0.3,
    "Tiruttani": 0.8, "Vellore": 0.4, "Cuddalore": 2.1, "Tondi": 0.6,
    "Nungambakkam": 0.2, "Nagapattinam": 1.8,
}
TEMP_LEARN_RATE = 0.3   # how fast STATION_TEMP_BIAS adapts per day (EMA); lower = smoother

# --- Temporary ECMWF exclusion from temp score (self-expiring) ---------------
# ECMWF's recent runs have shown a repeated cold skew at the temp stations
# (flagged two days running at Chennai). Until the IST date below (EXCLUSIVE),
# ECMWF is dropped from the temp mean/spread/score - the ecmwf column still
# DISPLAYS so you can watch whether the skew persists, it just stops dragging
# the rank down. Self-expiring: after the date it silently reverts to all three
# models, no cleanup needed. Set to None (or a past date) to disable early.
# NOTE: applies to TEMP only. Rain corr stays ECMWF-based on purpose - the
# learned rain factors are actual/ECMWF ratios (corr breaks without it), and
# ECMWF remains the best rain model on surge days (e.g. Tamhini today).
# ENDED 07-Jul-2026 (was 2026-07-13): today's evidence flipped - ECMWF ran only
# -1.4 vs actual (39.0) while GFS/ICON both ran +1.2 hot, so ECMWF was the BEST
# model today, not the worst. Excluding it was pulling the score too warm, the
# opposite of what the exclusion was meant to fix. Set to today's date so the
# exclusion ends immediately (all three models vote again from this run forward).
# If ECMWF resumes a persistent cold skew on a future day, re-add a fresh
# self-expiring window then rather than reviving this one.
TEMP_EXCLUDE_ECMWF_UNTIL = "2026-07-07"


def _temp_ecmwf_excluded():
    if not TEMP_EXCLUDE_ECMWF_UNTIL:
        return False
    try:
        until = datetime.fromisoformat(TEMP_EXCLUDE_ECMWF_UNTIL).date()
        return datetime.now(ZoneInfo(TIMEZONE)).date() < until
    except Exception:
        return False

# Per-(station, model) correction (degC), applied to ONE model's raw value before
# it feeds into mean/spread/score - unlike STATION_TEMP_BIAS which shifts all three
# models equally. Use this when one specific model is the outlier, not the station
# as a whole. SEEDED 01-Jul (1 day only): ECMWF ran cold vs gfs/icon at Nungambakkam
# and Meenambakkam; GFS ran cold vs ecmwf/icon at Tiruchi. Values = ~50% of the gap
# between that model and the other two models' average (same conservative logic as
# the station-level bootstrap). Confirm over a few more days before trusting fully.
MODEL_STATION_BIAS = {
    ("Meenambakkam", "ecmwf"):  0.7,
    ("Nungambakkam", "ecmwf"):  0.7,
    ("Tiruchi", "gfs"):         1.0,
    ("Tondi", "gfs"):           0.6,   # PLACEHOLDER (user obs: GFS chronically under-reads
                                        # Tondi's max). Not yet quantified from per-model data -
                                        # send a few days of Tondi gfs-vs-actual to calibrate.
                                        # Stacks with the no-sea-breeze bump below, so on a
                                        # no-breeze day GFS gets base+bump+this; watch for overwarm.
    # Madurai, CALIBRATED 08-Jul from a 28-day 18Z-pinned backtest (standalone
    # madurai_temp_backtest.py) - not a single-day seed like the entries above.
    # All three raw models run persistently COLD at Madurai, but by different,
    # fairly consistent amounts: ecmwf mean bias +1.99 (std 1.51, noisiest but
    # most accurate overall, MAE 2.02), gfs +2.69 (std 1.11, very consistent
    # offset), icon +2.29 (std 1.12, also consistent). The flat STATION_TEMP_BIAS
    # this replaced (2.3) was ~the average of these three - correct in aggregate,
    # but under-corrected gfs/icon and over-corrected ecmwf. Giving each model
    # its own correction here should tighten the spread column too (previously
    # the raw per-model disagreement passed through unchanged under a flat shift;
    # now the models should converge closer once each is individually corrected).
    # STATION_TEMP_BIAS["Madurai"] was zeroed out to avoid double-correcting -
    # do not re-add a flat Madurai entry there without removing/adjusting these.
    ("Madurai", "ecmwf"):       2.0,
    ("Madurai", "gfs"):         2.7,
    ("Madurai", "icon"):        2.3,
}

# --- No-sea-breeze conditional max-temp bump --------------------------------
# Some coastal stations run notably HOTTER than the models on days the sea
# breeze fails to set up: no marine air arrives to cap the afternoon max (this
# happens under an offshore/westerly surge that suppresses the Bay breeze).
# This is a CONDITIONAL error - a flat STATION_TEMP_BIAS would over-warm the
# station on the many NORMAL days the breeze does show. So the extra warmth is
# added ONLY when the LIVE sea-breeze detector reports "No Sea Breeze today" for
# that station (the climatological fallback never triggers it, by design - we
# only boost when we actually observe no breeze). Keys off the same string the
# temp table already shows in its 'sea breeze' column.
USE_NO_SEABREEZE_BUMP = True
NO_SEABREEZE_TEMP_BUMP = {
    "Tondi": 1.6,   # narrow peninsula, normally earliest/strongest breeze; overshoots HARD
                    # on no-breeze/surge days. Net no-breeze correction = base 0.4 + 1.6 = +2.0.
                    # Sized to the 04-Jul no-breeze evidence (actual 36.2 vs a +0.4-corrected
                    # 34.2 = +2.0 miss, i.e. ~+2.4 total wanted) tempered slightly off the full
                    # single-day residual. First conservative seed (0.6) undershot and left
                    # no-breeze days looking unchanged; this makes them actually run hot. Base
                    # STATION_TEMP_BIAS stays 0.4 so breeze days are untouched. If it overshoots
                    # a no-breeze day, refine/EMA it down; if it keeps missing hot, push toward 2.0.
}


def _no_seabreeze(sb_state):
    """True if the (live-detected) sea-breeze string means no breeze today."""
    return isinstance(sb_state, str) and "No Sea Breeze" in sb_state


# --- NEW 07-Jul: graded bump for a LATE (but technically-detected) breeze ----
# 07-Jul exposed a gap: Tondi's live detector found an onset at 16:00 IST
# (normal window 11:30-12:00) - a real onset string, so _no_seabreeze() read
# False and NO_SEABREEZE_TEMP_BUMP never fired, even though a breeze arriving
# 4+ hours late has almost no time to cap the afternoon max and behaves close
# to a true no-breeze day (actual ran +2.1 over forecast, vs a full no-breeze
# station's typical +2.0 net correction). A flat late/on-time cutoff would
# just move the cliff edge rather than removing it, so this instead RAMPS the
# bump in proportional to how many hours late the onset is past the station's
# normal window, capping at the full NO_SEABREEZE_TEMP_BUMP once lateness
# reaches LATE_BREEZE_FULL_BUMP_HOURS. On-time or early onsets get 0 - normal
# days are untouched, same as before this fix.
USE_LATE_SEABREEZE_BUMP = True
LATE_BREEZE_FULL_BUMP_HOURS = 4.0   # hours past the normal window's end = full bump


def _seabreeze_lateness_bump(name, sb_state):
    """Partial (ramped) bump for an onset that IS detected but arrives late.
    Returns 0.0 if disabled, if sb_state isn't a real "HH:MM IST" onset string
    (e.g. it's a no-breeze string or a raw '-'), or if the station has no
    climatological onset window to compare against (e.g. 'negligible')."""
    if not (USE_LATE_SEABREEZE_BUMP and isinstance(sb_state, str)
            and sb_state.endswith("IST") and "No Sea Breeze" not in sb_state):
        return 0.0
    normal = SEA_BREEZE_ONSET.get(name)
    if not normal or "-" not in normal or normal == "negligible":
        return 0.0
    try:
        onset_hm = sb_state.split(" ")[0]
        onset_dt = datetime.strptime(onset_hm, "%H:%M")
        window_end_hm = normal.split("-")[1].strip().split(" ")[0]
        window_end_dt = datetime.strptime(window_end_hm, "%H:%M")
    except Exception:
        return 0.0
    late_hours = (onset_dt - window_end_dt).total_seconds() / 3600.0
    if late_hours <= 0:
        return 0.0   # on-time or early - no adjustment
    frac = min(late_hours / LATE_BREEZE_FULL_BUMP_HOURS, 1.0)
    return round(frac * NO_SEABREEZE_TEMP_BUMP.get(name, 0.0), 2)


# Expected sea breeze onset (local IST), by distance/exposure from the coast.
# Climatological FALLBACK ONLY - used if the live wind fetch fails. The live
# path (compute_sea_breeze, wired in __main__) detects the actual daily onset
# from hourly wind direction/speed and overrides this. ALSO used (as of 07-Jul)
# as the reference "normal window" for _seabreeze_lateness_bump() above, even
# when the live path succeeds - i.e. it now does double duty as both the
# fallback value AND the lateness yardstick.
SEA_BREEZE_ONSET = {
    "Tondi":          "11:30-12:00 IST",  # narrow peninsula, coast on both sides - earliest, strongest
    "Nungambakkam":   "12:00-12:30 IST",  # Chennai, ~2-3 km from coast
    "Cuddalore":      "12:00-12:30 IST",  # directly on coast
    "Nagapattinam":   "12:00-12:30 IST",  # directly on coast
    "Meenambakkam":   "12:30-13:00 IST",  # Chennai, ~7-8 km inland from coast
    "Palayamkottai":  "14:00-14:30 IST",  # ~30 km from Gulf of Mannar, delayed/weaker
    "Tiruchi":        "14:30-15:00 IST",  # ~60 km inland, moderate delay
    "Tiruttani":      "15:00-15:30 IST",  # ~85 km NW of Chennai, weak by arrival
    "Madurai":        "15:00-15:30 IST",  # ~100 km inland, weak/inconsistent
    "Vellore":        "negligible",       # ~140 km inland, rarely reaches with any strength
}

# --- Live sea breeze detection (hourly wind direction/speed) ------------------
# TN east coast faces the Bay of Bengal; onshore wind (sea breeze) arrives FROM
# the east-ish quadrant. Band is intentionally wide (ENE through SSE) since
# individual station coastlines aren't perfectly north-south.
SEABREEZE_DIR_MIN, SEABREEZE_DIR_MAX = 45, 165     # degrees, "from" direction
SEABREEZE_SPEED_MIN = 10                            # km/h, min sustained onshore speed
SEABREEZE_SEARCH_START, SEABREEZE_SEARCH_END = 9, 19   # IST hour window to search
# WORKFLOW CHANGE (08-Jul): this dict used to need manual daily edits (paste
# IMD actuals in, watch for a "paste this back" bias-refinement printout, copy
# it back into STATION_TEMP_BIAS by hand). That's gone now - the actuals come
# in as a daily IMD bulletin screenshot in chat instead, and the bias math /
# STATION_TEMP_BIAS / MODEL_STATION_BIAS edits get made directly in the script
# from there - same as the 07-Jul EMA pass and the Madurai per-model fix.
# Left EMPTY on purpose: an empty dict means refine_temp_bias()/log_scoreboard()
# silently no-op (both check `if not actuals: return`), so the backtest and
# live report still run and display normally - you just won't see the
# "Temp bias refinement" / "paste this back" text block cluttering the output
# anymore. No need to ever hand-edit this dict again.
ACTUALS = {}
REFINE_BIAS = False     # print bias-refinement suggestions during the backtest. Keep OFF
                        # for clean output. Turn ON only when re-tuning, and set BACKTEST_DATE
                        # to the SAME day as your RAIN_ACTUALS/ACTUALS - otherwise it scores
                        # the actuals against the wrong day's forecast.
BACKTEST_DATE = (datetime.now(ZoneInfo(TIMEZONE)).date() - timedelta(days=1)).isoformat()
OUTFILE = "contest_report.html"
# Live Chennai rain probability, pulled from chennai_rain.py's analyze(). That
# file must sit in the SAME folder as this script (upload both to Colab).
# If the import or the live fetch fails for any reason, falls back to manual
# entry below (or "N/A" if that's also unset) so the report never breaks.
try:
    from chennai_rain import analyze as _chennai_rain_analyze
except Exception as e:
    _log(f"chennai_rain.py not found/importable ({type(e).__name__}); "
         f"place it in the same folder to enable live Chennai rain probability.")
    _chennai_rain_analyze = None
CHENNAI_RAIN_PROB_MANUAL = None   # fallback only, e.g. 70 for 70% - used if the live call fails
# --- Freeze & replay: save the EXACT run you pick from, and backtest against it ---
FREEZE_FORECAST = True   # each run saves the live contest-day forecast (last run before
                         # your deadline wins). Backtest then replays THAT run, not the
                         # hindsight Historical API - a true forecast-skill comparison.
USE_DRIVE = True         # mount Google Drive so frozen forecasts persist across sessions
FROZEN_DIR = "/content/drive/MyDrive/contest_frozen"  # falls back to /content if Drive off
# ------------------------------------------------------------------------------

# --- ECMWF-vs-other-models running scoreboard --------------------------------
# Tracks, for the stations listed, how raw ECMWF compared to the OTHER models'
# mean against the actual, each day - so a debate like "did ECMWF nail Chennai
# today" doesn't rely on memory. Logged automatically whenever ACTUALS is
# populated for the backtest date, INDEPENDENT of REFINE_BIAS (that flag only
# controls whether biases get updated; this is a pure observation log, always on
# by default). Idempotent - re-running the same day/station is a no-op.
LOG_SCOREBOARD = True
SCOREBOARD_STATIONS = {"Meenambakkam", "Nungambakkam"}
SCOREBOARD_FILE_NAME = "ecmwf_scoreboard.json"


# --- Retry-with-backoff for flaky Colab connections -------------------------
# A stalled/dropped connection to Open-Meteo previously hung indefinitely
# inside requests' socket read (observed 02-Jul: KeyboardInterrupt needed to
# escape a hung fetch(STATIONS, ...) call). This wraps every GET so transient
# network errors (timeout, connection reset, 5xx) retry with exponential
# backoff instead of hanging or crashing the whole run on one bad request.
# A genuine manual stop (Colab's stop button / Ctrl-C) raises KeyboardInterrupt,
# which is a BaseException, NOT caught by `except requests.exceptions.*` below -
# so you can still cancel a cell by hand at any point, retries or not.
#
# 04-Jul HARDENING (Open-Meteo throttling Colab IPs -> timeouts):
#   * Timeout is now a (connect, read) TUPLE, not a scalar. The throttle/block
#     symptom is a CONNECT hang (the IP never gets a SYN-ACK - see Open-Meteo
#     issues #1651/#1669), so a SHORT connect cap is what actually makes a bad
#     fetch fail fast instead of sitting for the full 30s x every attempt.
#   * 4xx responses are NO LONGER retried. A 429 (rate-limit) or 400 (bad param)
#     won't fix itself on retry - retrying just burns the contest window and
#     hammers an IP that's already being throttled. Fail fast, loud, labelled.
#   * 5xx + timeouts + connection errors still retry with backoff as before.
RETRY_ATTEMPTS = 3        # total tries per request (1 initial + 2 retries)
RETRY_BACKOFF_BASE = 2    # seconds; wait doubles each retry (2s, 4s, ...)
CONNECT_TIMEOUT = 6       # s to establish the TCP/TLS connection. A throttled/blocked
                          # Colab IP hangs HERE, so keep it short -> fail fast, retry.
READ_TIMEOUT = 20         # s to wait on the response body once connected. This is a
                          # per-read-gap timeout (not a total cap), so large-but-flowing
                          # multi-station responses are unaffected; only a dead-silent
                          # socket trips it.


def _normalize_timeout(timeout):
    """Return a (connect, read) tuple. Accepts None (module defaults), a bare
    int/float (kept as the READ timeout, short CONNECT_TIMEOUT prepended - so
    old callers passing timeout=30/15 still behave sanely), or a ready tuple."""
    if timeout is None:
        return (CONNECT_TIMEOUT, READ_TIMEOUT)
    if isinstance(timeout, (int, float)):
        return (CONNECT_TIMEOUT, timeout)
    return timeout


def _get_with_retry(url, params=None, timeout=None, attempts=RETRY_ATTEMPTS):
    to = _normalize_timeout(timeout)
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            r = requests.get(url, params=params, timeout=to)
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            # 4xx = client error: do NOT retry (won't self-heal, wastes the window).
            if status is not None and 400 <= status < 500:
                if status == 429:
                    raise RuntimeError(
                        "Open-Meteo 429: rate-limited (shared Colab IP throttle). "
                        "Restart the Colab runtime to reroll the IP, or wait a few "
                        "minutes, before retrying."
                    ) from e
                body = getattr(e.response, "text", "") or ""
                raise RuntimeError(
                    f"Open-Meteo HTTP {status} (client error, not retried): {body[:200]}"
                ) from e
            # 5xx = transient server-side: fall through to the retry/backoff path.
            last_exc = e
        except requests.exceptions.RequestException as e:
            # Timeout / connection reset / refused / DNS - the throttle-or-blocked
            # symptom. Retry with backoff.
            last_exc = e
        if attempt < attempts:
            wait = RETRY_BACKOFF_BASE ** attempt
            print(f"  !! request failed ({type(last_exc).__name__}: {last_exc}); "
                  f"retry {attempt}/{attempts - 1} in {wait}s...")
            time.sleep(wait)
    # Exhausted all attempts on a timeout/connection/5xx error.
    raise RuntimeError(
        f"Open-Meteo unreachable after {attempts} tries (likely IP/rate-limit "
        f"block, not a global outage - try restarting the Colab runtime): {last_exc}"
    ) from last_exc


def fetch(stations, hourly_var):
    lats = ",".join(str(lat) for lat, lon in stations.values())
    lons = ",".join(str(lon) for lat, lon in stations.values())
    params = {"latitude": lats, "longitude": lons, "hourly": hourly_var,
              "models": ",".join(MODELS), "timezone": TIMEZONE,
              "forecast_days": max(DAY_OFFSET, 0) + 3, "precipitation_unit": "mm"}
    r = _get_with_retry("https://api.open-meteo.com/v1/forecast", params=params)
    data = r.json()
    return data if isinstance(data, list) else [data]


def fetch_historical(stations, hourly_var, date_str):
    d = datetime.fromisoformat(date_str).date()
    lats = ",".join(str(lat) for lat, lon in stations.values())
    lons = ",".join(str(lon) for lat, lon in stations.values())
    params = {"latitude": lats, "longitude": lons, "hourly": hourly_var,
              "models": ",".join(MODELS), "timezone": TIMEZONE,
              "start_date": date_str, "end_date": (d + timedelta(days=1)).isoformat(),
              "precipitation_unit": "mm"}
    r = _get_with_retry("https://historical-forecast-api.open-meteo.com/v1/forecast",
                        params=params)
    data = r.json()
    return data if isinstance(data, list) else [data]


# --- Single Runs API: pin an EXACT past model run by UTC init time ----------
# Unlike fetch() (always the freshest run right now) or fetch_historical()
# (a stitched multi-run time series), this returns exactly what ONE specific
# run said - e.g. "yesterday's 18Z run" - independent of when you happen to
# execute the script. run_iso format: "YYYY-MM-DDTHH:MM" UTC, synoptic hours
# only (00/06/12/18 for the global models here). Archive coverage: ECMWF IFS
# HRES from Mar-2024; GFS/ICON from Sep-2025 - any 2026 date is covered.
# Free tier, no API key, same rate limits as the live Forecast API.
SINGLE_RUNS_BASE = "https://single-runs-api.open-meteo.com/v1/forecast"


def fetch_single_run(stations, hourly_var, run_iso, forecast_days=3):
    """Fetch one pinned model run's hourly data. Uses SINGLE_RUN_MODELS
    (ecmwf_ifs025, gfs_global, icon_global) since the Single Runs API rejects
    the "seamless" blends used elsewhere (gfs_seamless/icon_seamless) with a
    fixed run= timestamp - previously this silently returned a non-JSON body
    (JSONDecodeError) because the wrong model list (MODELS) was requested
    here. The returned hourly keys are relabelled back to the seamless names
    (_gfs_global -> _gfs_seamless, _icon_global -> _icon_seamless) so every
    downstream consumer (build_table, build_temp_table, etc., which all loop
    over MODELS/LABELS) keeps working without any changes."""
    lats = ",".join(str(lat) for lat, lon in stations.values())
    lons = ",".join(str(lon) for lat, lon in stations.values())
    params = {"latitude": lats, "longitude": lons, "hourly": hourly_var,
              "models": ",".join(SINGLE_RUN_MODELS), "timezone": TIMEZONE,
              "run": run_iso, "forecast_days": forecast_days, "precipitation_unit": "mm"}
    r = _get_with_retry(SINGLE_RUNS_BASE, params=params)
    data = r.json()
    data = data if isinstance(data, list) else [data]
    for loc in data:
        hourly = loc.get("hourly", {})
        for src, dst in _SINGLE_RUN_TO_SEAMLESS.items():
            if src == dst:
                continue
            for key in list(hourly.keys()):
                if key.endswith(f"_{src}"):
                    new_key = key[: -len(src)] + dst
                    hourly[new_key] = hourly.pop(key)
    return data


def fetch_single_run_daily(stations, run_iso, target_date=None, forecast_days=3):
    """Daily temperature_2m_max for ONE pinned run, shaped like the real daily
    endpoint's response so build_temp_table() can consume it unmodified.

    WORKAROUND (found 08-Jul): the Single Runs API rejects `daily=` output
    unless `run` happens to land exactly at 00:00 in the requested timezone -
    a real 18Z-style run in Asia/Kolkata NEVER satisfies that (18Z UTC =
    23:30 IST, never 00:00), so calling the daily endpoint directly with an
    18Z run always 400s ("'daily' is only supported ... when 'run' starts at
    00:00 in the requested timezone"). Fetches HOURLY temperature_2m for the
    pinned run instead (no such restriction on hourly) and computes the
    calendar-day max itself for `target_date` - identical result to what the
    daily endpoint would have returned, without hitting the restriction.
    target_date defaults to today's live contest date (respecting DAY_OFFSET).
    """
    td = target_date or (datetime.now(ZoneInfo(TIMEZONE)).date() + timedelta(days=DAY_OFFSET))
    if isinstance(td, str):
        td = datetime.fromisoformat(td).date()
    target = td.isoformat()
    raw = fetch_single_run(stations, "temperature_2m", run_iso, forecast_days=forecast_days)
    out = []
    for loc in raw:
        hourly = loc.get("hourly", {})
        times = [datetime.fromisoformat(t) for t in hourly.get("time", [])]
        daily = {"time": [target]}
        for mdl in MODELS:
            series = hourly.get(f"temperature_2m_{mdl}")
            vals = []
            if series is not None:
                vals = [series[i] for i, t in enumerate(times)
                        if t.date().isoformat() == target and i < len(series)
                        and series[i] is not None]
            daily[f"temperature_2m_max_{mdl}"] = [max(vals) if vals else None]
        out.append({**loc, "daily": daily})
    return out


def single_run_report(run_iso, target_date=None, save=True):
    """Build rain/TN-rain/temp tables from ONE specific pinned run (e.g.
    "2026-07-02T18:00" for yesterday's 18Z) for a given contest window, using
    the same bias correction / climo floor / ranking as the live report.
    target_date defaults to today's IST contest window (DAY_OFFSET=0).
    Returns (html_string, {name: DataFrame}); also saves + displays inline
    when save=True, same as the main run. Call this directly in a cell:
        single_run_report("2026-07-02T18:00")
    """
    td = target_date or (datetime.now(ZoneInfo(TIMEZONE)).date() + timedelta(days=DAY_OFFSET))
    if isinstance(td, str):
        td = datetime.fromisoformat(td).date()
    rain_df, win = build_table(fetch_single_run(STATIONS, "precipitation", run_iso),
                               STATIONS, "precipitation", "sum", td)
    tn_df, _ = build_table(fetch_single_run(TN_RAIN_STATIONS, "precipitation", run_iso),
                           TN_RAIN_STATIONS, "precipitation", "sum", td)
    temp_df = build_temp_table(fetch_single_run_daily(TEMP_STATIONS, run_iso, target_date=td),
                               TEMP_STATIONS, td, bias=TEMP_BIAS, rank=TEMP_RANK)
    sections = [(f"Rainfall (mm) &mdash; single run {run_iso}Z", rain_df, "rain"),
                (f"TN rainfall (mm) &mdash; single run {run_iso}Z", tn_df, "rain"),
                (f"Max temp (\u00b0C) &mdash; single run {run_iso}Z", temp_df, "temp")]
    html = build_report({"pinned run": f"{run_iso}Z"}, win, sections)
    if save:
        fname = f"single_run_{run_iso.replace(':', '')}.html"
        with open(fname, "w") as f:
            f.write("<!doctype html><meta name='viewport' "
                    "content='width=device-width,initial-scale=1'>" + html)
        print(f"Saved {fname}")
        try:
            from IPython.display import HTML, display
            display(HTML(html))
        except Exception:
            pass
    tables = {"rainfall": rain_df, "tn_rainfall": tn_df, "temp": temp_df}
    return html, tables


# --- Standalone backtest against one exact pinned run (e.g. 18Z) -------------
# Unlike the OLD BACKTEST_DATE mechanism (removed 08-Jul - it replayed a
# FROZEN daily snapshot of whatever run happened to be live when you ran the
# script that day, e.g. a 06Z GFS run, or fell back to the Historical API's
# blended multi-run series), this pins ONE specific model cycle for ALL THREE
# tables (rainfall, TN rainfall, temp) - "did last night's 18Z run predict
# today correctly" - the cleanest apples-to-apples test of a single run's
# skill, independent of whatever happened to be freshest when you ran it.
#
# IMPORTANT FIX (temp table specifically): sea-breeze detection is pinned to
# the SAME run_iso (via fetch_single_run for wind), not the climatological
# SEA_BREEZE_ONSET fallback. Without this, a real no-breeze day in the pinned
# run would silently miss NO_SEABREEZE_TEMP_BUMP / _seabreeze_lateness_bump,
# making the backtest's residual look like a bias problem when it was
# actually a missed conditional bump.
SINGLE_RUN_BACKTEST = "AUTO"   # "AUTO" = yesterday's 18Z (UTC), computed fresh every
                                # run below - no date to edit, ever. Set an explicit
                                # "YYYY-MM-DDTHH:MM" string instead to pin one specific
                                # run manually; set to None/"" to turn ALL pinned-run
                                # backtests off (rainfall, TN rainfall, and temp).


def _default_18z_run_iso():
    """Yesterday's 18Z (UTC) run, computed from the current moment - i.e. always
    the most recent evening run before today's contest window. Used when
    SINGLE_RUN_BACKTEST == "AUTO" so the daily backtest never needs a
    manually-edited date string."""
    y = datetime.now(timezone.utc).date() - timedelta(days=1)
    return f"{y.isoformat()}T18:00"


def backtest_rain_single_run(run_iso, target_date=None, show=True):
    """Backtest RAINFALL + TN RAINFALL against one exact pinned model run (e.g.
    yesterday's 18Z) - the rainfall equivalent of backtest_temp_single_run.
    Replaces the old frozen-snapshot/Historical-API rainfall backtest, which
    could show whatever run happened to be live at freeze time (e.g. a 06Z
    GFS run) rather than a consistent 18Z pin. target_date defaults to the
    live contest date. Always displays both tables inline; returns
    (rain_df, tn_df)."""
    td = target_date or (datetime.now(ZoneInfo(TIMEZONE)).date() + timedelta(days=DAY_OFFSET))
    if isinstance(td, str):
        td = datetime.fromisoformat(td).date()

    rain_df, win = build_table(fetch_single_run(STATIONS, "precipitation", run_iso),
                               STATIONS, "precipitation", "sum", td)
    tn_df, _ = build_table(fetch_single_run(TN_RAIN_STATIONS, "precipitation", run_iso),
                           TN_RAIN_STATIONS, "precipitation", "sum", td)

    if show:
        try:
            from IPython.display import HTML, display
            display(HTML(CSS + f'<div class="cr"><h2>Backtest rainfall (mm) &mdash; '
                         f'single run {run_iso}Z &mdash; {win}</h2>'
                         + html_table(rain_df, "rain")
                         + f'<h2>Backtest TN rainfall (mm) &mdash; single run {run_iso}Z</h2>'
                         + html_table(tn_df, "rain") + '</div>'))
        except Exception:
            print(f"(Display unavailable outside Colab; returning the DataFrames instead.)")

    return rain_df, tn_df


def backtest_temp_single_run(run_iso, target_date=None, actuals=None, show=True, refine=False):
    """Backtest MAX TEMP ONLY against one exact pinned model run (e.g. yesterday's
    18Z), rather than the blended Historical API series or today's freshest live
    run. target_date defaults to the live contest date (today, respecting
    DAY_OFFSET). actuals defaults to the module-level ACTUALS dict (empty by
    default as of 08-Jul - see ACTUALS' comment for the screenshot-based
    workflow). Always displays the table inline; prints refine_temp_bias()
    suggestions against actuals if any are available (does NOT apply them)."""
    td = target_date or (datetime.now(ZoneInfo(TIMEZONE)).date() + timedelta(days=DAY_OFFSET))
    if isinstance(td, str):
        td = datetime.fromisoformat(td).date()
    act = actuals if actuals is not None else ACTUALS

    try:
        wind_data = fetch_single_run(TEMP_STATIONS, "winddirection_10m,windspeed_10m", run_iso)
        sb = compute_sea_breeze(wind_data, TEMP_STATIONS, td)
    except Exception as e:
        print(f"Pinned-run ({run_iso}Z) sea breeze fetch failed ({type(e).__name__}); "
              f"falling back to climatological onset times for this backtest only.")
        sb = None

    temp_data = fetch_single_run_daily(TEMP_STATIONS, run_iso, target_date=td)
    df = build_temp_table(temp_data, TEMP_STATIONS, td, bias=TEMP_BIAS, rank=TEMP_RANK,
                          sea_breeze=sb)

    if show:
        try:
            from IPython.display import HTML, display
            display(HTML(CSS + f'<div class="cr"><h2>Backtest temps (\u00b0C) &mdash; '
                         f'single run {run_iso}Z vs actual ({td})</h2>'
                         + html_table(df, "temp") + '</div>'))
        except Exception:
            print(f"(Display unavailable outside Colab; returning the DataFrame instead.)")

    if act:
        refine_temp_bias(df, act, source=f"single run {run_iso}Z")
    return df


def window_bounds(contest_date):
    h = int(WINDOW_START)
    m = int(round((WINDOW_START - h) * 60))
    start = datetime.combine(contest_date, datetime.min.time()).replace(hour=h, minute=m)
    return start, start + timedelta(days=1)


def fetch_avalanche_850(past_days=3):
    """850hPa wind speed/direction + RH at Avalanche, ECMWF only (primary model
    per the station-bias notes elsewhere in this script)."""
    lat, lon = AVALANCHE_LATLON
    params = {"latitude": lat, "longitude": lon,
              "hourly": "wind_speed_850hPa,wind_direction_850hPa,relative_humidity_850hPa",
              "models": "ecmwf_ifs025", "timezone": TIMEZONE, "wind_speed_unit": "ms",
              "past_days": past_days, "forecast_days": 2}
    r = _get_with_retry("https://api.open-meteo.com/v1/forecast", params=params)
    j = r.json()
    return j if isinstance(j, list) else [j]


def analyze_avalanche_surge(contest_date):
    """Compare today's vs yesterday's 850hPa wind/moisture over the Avalanche
    contest window - the same diagnostic IMD uses for monsoon surge strength,
    applied at station scale. Returns a dict, or None if the fetch fails."""
    try:
        data = fetch_avalanche_850()
    except Exception as e:
        print(f"Avalanche surge fetch failed: {type(e).__name__}: {e}")
        return None
    loc = data[0]
    hourly = loc.get("hourly", {})

    def hget(base):
        # Open-Meteo suffixes hourly keys with the model id when 'models' is
        # passed explicitly for some endpoints/params but not always for a
        # single model - try both so this doesn't silently return None.
        return hourly.get(f"{base}_ecmwf_ifs025", hourly.get(base))

    times_raw = hourly.get("time", [])
    times = [datetime.fromisoformat(t) for t in times_raw]
    spd = hget("wind_speed_850hPa")
    drc = hget("wind_direction_850hPa")
    rh = hget("relative_humidity_850hPa")
    if not (times and spd and drc and rh):
        print(f"Avalanche surge: missing data - keys available: {list(hourly.keys())}")
        return None

    def window_avg(day):
        ws, we = window_bounds(day)
        idx = [i for i, t in enumerate(times) if ws <= t < we]
        s = [spd[i] for i in idx if i < len(spd) and spd[i] is not None]
        d = [drc[i] for i in idx if i < len(drc) and drc[i] is not None]
        h = [rh[i] for i in idx if i < len(rh) and rh[i] is not None]
        if not (s and d and h):
            return None
        return {"speed": round(sum(s) / len(s), 1), "dir": round(sum(d) / len(d)),
                "rh": round(sum(h) / len(h))}

    today = window_avg(contest_date)
    yday = window_avg(contest_date - timedelta(days=1))
    if today is None:
        return None
    onshore = SURGE_DIR_MIN <= today["dir"] <= SURGE_DIR_MAX
    strong = onshore and today["speed"] >= SURGE_WIND_MIN
    verdict = "active surge" if strong else ("weak/marginal" if onshore else "not westerly")
    trend = None
    if yday is not None:
        d_spd = round(today["speed"] - yday["speed"], 1)
        d_rh = today["rh"] - yday["rh"]
        if d_spd > 1 or d_rh > 5:
            trend = "stronger than yesterday"
        elif d_spd < -1 or d_rh < -5:
            trend = "weaker than yesterday"
        else:
            trend = "similar to yesterday"
    return {"today": today, "yday": yday, "onshore": onshore, "verdict": verdict, "trend": trend}


# --- Generalized per-station surge detection (feeds STATION_SURGE_FACTOR) -----
# Same 850hPa diagnostic as analyze_avalanche_surge, but for an arbitrary point
# and either the live forecast or a past (historical) window. Cached so the same
# station/date isn't refetched across the several build_table calls in one run.
_SURGE_CACHE = {}


def _fetch_point_850(lat, lon, contest_date, historical=False):
    """Hourly 850hPa wind speed/dir at one point over the contest day (+neighbours),
    ECMWF only. Uses the live Forecast API for current/near dates, the Historical
    Forecast API for past ones (so refine can classify a backtest day)."""
    params = {"latitude": lat, "longitude": lon,
              "hourly": "wind_speed_850hPa,wind_direction_850hPa",
              "models": "ecmwf_ifs025", "timezone": TIMEZONE, "wind_speed_unit": "ms"}
    if historical:
        params["start_date"] = contest_date.isoformat()
        params["end_date"] = (contest_date + timedelta(days=1)).isoformat()
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    else:
        params["past_days"] = 1
        params["forecast_days"] = max(DAY_OFFSET, 0) + 2
        url = "https://api.open-meteo.com/v1/forecast"
    j = _get_with_retry(url, params=params).json()
    return j[0] if isinstance(j, list) else j


def _point_850_uv_samples(lat, lon, contest_date, historical=False):
    """Hourly 850hPa wind (u, v) components over the contest window at one point.
    Vector components let readings be averaged correctly (scalar direction means
    are wrong near 0/360). Returns (us, vs) or ([], []) if unavailable."""
    loc = _fetch_point_850(lat, lon, contest_date, historical)
    hourly = loc.get("hourly", {})

    def hget(base):
        return hourly.get(f"{base}_ecmwf_ifs025", hourly.get(base))

    times = [datetime.fromisoformat(t) for t in hourly.get("time", [])]
    spd, drc = hget("wind_speed_850hPa"), hget("wind_direction_850hPa")
    if not (times and spd and drc):
        return [], []
    ws, we = window_bounds(contest_date)
    us, vs = [], []
    for i, t in enumerate(times):
        if ws <= t < we and i < len(spd) and i < len(drc) \
                and spd[i] is not None and drc[i] is not None:
            th = math.radians(drc[i])            # meteorological "from" direction
            us.append(-spd[i] * math.sin(th))
            vs.append(-spd[i] * math.cos(th))
    return us, vs


def _classify_uv(u, v, dir_band, wind_min):
    """Classify a mean (u, v) flow as 'surge' or 'quiet'."""
    speed = math.hypot(u, v)
    direction = math.degrees(math.atan2(-u, -v)) % 360
    onshore = dir_band[0] <= direction <= dir_band[1]
    state = "surge" if (onshore and speed >= wind_min) else "quiet"
    return state, {"speed": round(speed, 1), "dir": round(direction), "onshore": onshore}


def station_surge_state(lat, lon, contest_date, dir_band, wind_min, historical=False):
    """Classify the 850hPa flow over one station's window as 'surge' or 'quiet'.
    Returns (state, details) or (None, None) if the data is unavailable."""
    key = (round(lat, 3), round(lon, 3), contest_date.isoformat(), historical, dir_band, wind_min)
    if key in _SURGE_CACHE:
        return _SURGE_CACHE[key]
    us, vs = _point_850_uv_samples(lat, lon, contest_date, historical)
    if not us:
        _SURGE_CACHE[key] = (None, None)
        return None, None
    state, det = _classify_uv(sum(us) / len(us), sum(vs) / len(vs), dir_band, wind_min)
    _SURGE_CACHE[key] = (state, det)
    return state, det


def compute_surge_states(stations, contest_date, historical=False):
    """Detect surge regimes for every STATION_SURGE_FACTOR station in `stations`.
    CLUSTERED stations get ONE pooled decision (see SURGE_CLUSTERS); the rest are
    classified per-station. Returns {name: (state, details)}; stations with no
    850 data are omitted (build_table then uses the static factor)."""
    out = {}
    if not USE_SURGE_FACTOR:
        return out
    present = stations
    clustered = set()

    # 1) cluster decisions: pool 850 flow across members, classify once
    for cname, members in SURGE_CLUSTERS.items():
        mem = [m for m in members if m in STATION_SURGE_FACTOR and m in present]
        if not mem:
            continue
        cfg0 = STATION_SURGE_FACTOR[mem[0]]           # members share dir/wind_min
        member_uv, pooled_u, pooled_v = {}, [], []
        for m in mem:
            lat, lon = present[m]
            try:
                mu, mv = _point_850_uv_samples(lat, lon, contest_date, historical)
            except Exception as e:
                print(f"  cluster {cname}: {m} 850 fetch failed ({type(e).__name__})")
                mu, mv = [], []
            if mu:
                member_uv[m] = (sum(mu) / len(mu), sum(mv) / len(mv))
                pooled_u += mu
                pooled_v += mv
        if not member_uv:
            print(f"  cluster {cname}: no 850 data; members fall back to static factor.")
            continue
        if SURGE_CLUSTER_AGG == "max":
            per = {m: _classify_uv(u, v, cfg0["dir"], cfg0["wind_min"])
                   for m, (u, v) in member_uv.items()}
            surge_dets = [d for (s, d) in per.values() if s == "surge"]
            if surge_dets:
                state, det = "surge", max(surge_dets, key=lambda d: d["speed"])
            else:
                state = "quiet"
                det = max((d for (s, d) in per.values()), key=lambda d: d["speed"])
        else:                                          # "mean": regional vector-mean flow
            state, det = _classify_uv(sum(pooled_u) / len(pooled_u),
                                      sum(pooled_v) / len(pooled_v),
                                      cfg0["dir"], cfg0["wind_min"])
        per_member = {m: round(math.hypot(u, v), 1) for m, (u, v) in member_uv.items()}
        det = {**det, "cluster": cname, "members": per_member}
        for m in mem:                                  # hand the ONE regime to ALL members
            out[m] = (state, det)
            clustered.add(m)
        pm = ", ".join(f"{k} {v}m/s" for k, v in per_member.items())
        _log(f"  surge cluster {cname}: {state} "
             f"({SURGE_CLUSTER_AGG} {det['speed']}m/s @ {det['dir']}deg; members: {pm})")

    # 2) surge stations not in any cluster: per-station detection
    for name, (lat, lon) in stations.items():
        if name in clustered or name not in STATION_SURGE_FACTOR:
            continue
        cfg = STATION_SURGE_FACTOR[name]
        try:
            st, dt = station_surge_state(lat, lon, contest_date, cfg["dir"], cfg["wind_min"], historical)
            if st is not None:
                out[name] = (st, dt)
            else:
                print(f"  surge state for {name}: no 850 data, using static factor.")
        except Exception as e:
            print(f"  surge state for {name} failed ({type(e).__name__}); static factor.")
    return out


# --- West-coast surge movement: where is the onshore-flow core, and did it move?
def _westcoast_members():
    """West-coast Ghats stations (name -> (lat, lon)), from WEST_COAST_STATIONS."""
    return {n: STATIONS[n] for n in WEST_COAST_STATIONS if n in STATIONS}


def fetch_westcoast_850(members):
    """One batched ECMWF 850hPa fetch (wind + RH) for all west-coast points,
    with past_days so both yesterday's and today's windows are covered."""
    lats = ",".join(str(la) for la, lo in members.values())
    lons = ",".join(str(lo) for la, lo in members.values())
    params = {"latitude": lats, "longitude": lons,
              "hourly": "wind_speed_850hPa,wind_direction_850hPa,relative_humidity_850hPa",
              "models": "ecmwf_ifs025", "timezone": TIMEZONE, "wind_speed_unit": "ms",
              "past_days": 2, "forecast_days": max(DAY_OFFSET, 0) + 2}
    j = _get_with_retry("https://api.open-meteo.com/v1/forecast", params=params).json()
    return j if isinstance(j, list) else [j]


def _surge_index(hourly, day):
    """Mean onshore-flow index over the day's 0830->0830 window at one station:
    wind_speed x RH/100 for hours the 850 flow is onshore (from the Arabian Sea),
    else 0. A rough moisture-laden-upslope-flow proxy. None if no data."""
    def hget(b):
        return hourly.get(f"{b}_ecmwf_ifs025", hourly.get(b))

    times = [datetime.fromisoformat(t) for t in hourly.get("time", [])]
    spd, drc, rh = hget("wind_speed_850hPa"), hget("wind_direction_850hPa"), hget("relative_humidity_850hPa")
    if not (times and spd and drc):
        return None
    ws, we = window_bounds(day)
    vals = []
    for i, t in enumerate(times):
        if ws <= t < we and i < len(spd) and i < len(drc) \
                and spd[i] is not None and drc[i] is not None:
            onshore = WESTCOAST_SURGE_DIR[0] <= drc[i] <= WESTCOAST_SURGE_DIR[1]
            r = rh[i] / 100.0 if (rh and i < len(rh) and rh[i] is not None) else 1.0
            vals.append(spd[i] * r if onshore else 0.0)
    return (sum(vals) / len(vals)) if vals else None


def surge_movement(contest_date):
    """Locate the west-coast onshore-flow core today vs yesterday and how it's
    changing. Core = the index-WEIGHTED CENTROID latitude (not the single peak
    station, which jitters between adjacent Ghats stations when the whole belt is
    active). Also tracks peak intensity and per-station day-over-day change so a
    PARKED surge still shows a live trend. Returns a dict or None."""
    if not SHOW_SURGE_MOVEMENT:
        return None
    members = _westcoast_members()
    if not members:
        return None
    try:
        data = fetch_westcoast_850(members)
    except Exception as e:
        print(f"Surge-movement fetch failed ({type(e).__name__}); banner skipped.")
        return None
    today_idx, yday_idx = {}, {}
    for name, loc in zip(members.keys(), data):
        h = loc.get("hourly", {})
        ti = _surge_index(h, contest_date)
        yi = _surge_index(h, contest_date - timedelta(days=1))
        if ti is not None:
            today_idx[name] = ti
        if yi is not None:
            yday_idx[name] = yi
    if not today_idx:
        return None

    def centroid(idxmap):
        den = sum(idxmap.values())
        return (sum(members[n][0] * v for n, v in idxmap.items()) / den) if den > 1e-6 else None

    t_cen, y_cen = centroid(today_idx), (centroid(yday_idx) if yday_idx else None)
    t_peak = max(today_idx.values())
    y_peak = max(yday_idx.values()) if yday_idx else None
    top = sorted(today_idx.items(), key=lambda kv: kv[1], reverse=True)
    deltas = {n: today_idx[n] - yday_idx.get(n, 0.0) for n in today_idx}
    building = max(deltas.items(), key=lambda kv: kv[1]) if deltas else None
    return {"members": members, "today_idx": today_idx, "yday_idx": yday_idx,
            "t_cen": t_cen, "y_cen": y_cen, "t_peak": t_peak, "y_peak": y_peak,
            "top": top, "deltas": deltas, "building": building,
            "contest_date": contest_date}


def _core_label(members, lat):
    """'near <nearest station>[, ~<city>]' for a core latitude."""
    stn = min(members, key=lambda n: abs(members[n][0] - lat))
    city = min(WEST_COAST_CITIES.items(), key=lambda kv: abs(kv[1] - lat))
    tag = f", \u2248{city[0]}" if abs(city[1] - lat) <= 1.2 else ""
    return f"near {stn}{tag}"


def surge_movement_banner_html(mv):
    """Two-line HTML banner: today's surge-core latitude, how it drifted vs
    yesterday, whether it's building/fading, today's heaviest stations and where
    it's strengthening fastest - plus a date stamp so it's obviously fresh."""
    if not mv or mv.get("t_cen") is None:
        return ""
    members, t_cen = mv["members"], mv["t_cen"]
    quiet = mv["t_peak"] < WESTCOAST_SURGE_MIN_IDX

    trend = []
    if mv["y_cen"] is not None:
        d = t_cen - mv["y_cen"]
        trend.append(f"drifted <b>{'north' if d > 0 else 'south'} ~{abs(d):.1f}\u00b0</b>"
                     if abs(d) >= 0.5 else "<b>steady</b> N-S")
    if mv["y_peak"]:
        ratio = mv["t_peak"] / mv["y_peak"]
        trend.append("intensity <b>building</b>" if ratio > 1.15
                     else "intensity <b>fading</b>" if ratio < 0.85
                     else "intensity steady")
    trend_txt = (" &mdash; " + " &middot; ".join(trend)) if trend else ""

    cd = mv["contest_date"]
    stamp = (f"<span style='color:#7f8a99'>&nbsp; [{cd - timedelta(days=1):%d %b} "
             f"\u2192 {cd:%d %b}]</span>")
    line1 = (f"\U0001f30a Surge core <b>~{t_cen:.1f}\u00b0N</b> "
             f"({_core_label(members, t_cen)}){trend_txt}{stamp}")

    if quiet:
        line2 = "Onshore flow weak coast-wide today &mdash; no strong orographic signal"
    else:
        names = [n for n, _ in mv["top"][:3]]
        line2 = f"Heaviest today: <b>{', '.join(names)}</b>"
        b = mv.get("building")
        if b and b[1] > 1.0:   # only if a station is meaningfully strengthening
            line2 += f" &middot; building fastest: <b>{b[0]}</b>"
    return f'<div class="surge-banner">{line1}<br>{line2}</div>'


def build_table(data, stations, var_prefix, agg, contest_date, bias=0.0, surge_states=None):
    names = list(stations.keys())
    apply_bias = USE_RAIN_BIAS and any(n in STATION_RAIN_BIAS for n in names)
    rows, warned = [], set()
    start, end = window_bounds(contest_date)
    window_label = f"{start:%Y-%m-%d %H:%M} -> {end:%Y-%m-%d %H:%M} IST"
    for name, loc in zip(names, data):
        hourly = loc.get("hourly", {})
        times = [datetime.fromisoformat(t) for t in hourly.get("time", [])]
        if not times:
            rows.append({"Station": name}); continue
        idx = [i for i, t in enumerate(times) if start <= t < end]
        row, vals = {"Station": name}, []
        for mdl in MODELS:
            label = LABELS.get(mdl, mdl)
            series = hourly.get(f"{var_prefix}_{mdl}")
            if series is None:
                if mdl not in warned:
                    keys = [k for k in hourly if k.startswith(var_prefix)]
                    print(f"  !! model '{mdl}' returned no data. Keys: {keys}")
                    warned.add(mdl)
                row[label] = None; continue
            wv = [series[i] for i in idx if i < len(series) and series[i] is not None]
            if wv:
                agg_val = sum(wv) if agg == "sum" else max(wv)
                row[label] = round(agg_val + bias, 1)
                vals.append(row[label])
            else:
                row[label] = None
        if vals:
            row["mean"] = round(sum(vals) / len(vals), 1)
            row["max"] = round(max(vals), 1)
            if USE_CLIMO_FLOOR:
                fl = CLIMO_FLOOR.get(name)
                if fl is not None and fl > row["max"]:
                    row["max"] = float(fl)   # climo floor overrides a lowballed model max
            row["spread"] = round(max(vals) - min(vals), 1)
            if apply_bias:
                f = STATION_RAIN_BIAS.get(name)
                # Surge-conditional override: when a regime was detected for this
                # station, use its surge/quiet factor instead of the static one.
                if (USE_SURGE_FACTOR and surge_states and name in STATION_SURGE_FACTOR
                        and name in surge_states):
                    state = surge_states[name][0]
                    f = STATION_SURGE_FACTOR[name]["surge" if state == "surge" else "quiet"]
                if f is not None:
                    bmap = {"ecmwf": row.get("ecmwf"), "gfs": row.get("gfs"),
                            "icon": row.get("icon"), "mean": row.get("mean"),
                            "max": row.get("max")}
                    bv = bmap.get(RAIN_BIAS_BASE)
                    if bv is not None and not pd.isna(bv):
                        f_eff = _gate_quiet(f, bv)   # relax a <1 factor on strong base
                        ff = max(RAIN_FACTOR_MIN, min(_factor_cap(name), f_eff))
                        if name in WEST_COAST_STATIONS:
                            ff = 1 + (ff - 1) * RAIN_WESTCOAST_DAMPEN
                        row["corr"] = round(bv * ff, 1)
                    else:
                        row["corr"] = row.get("max")
                else:   # station with no factor: pass through the existing rank metric
                    row["corr"] = row.get(RAIN_RANK if RAIN_RANK in ("mean", "max") else "max")
        rows.append(row)
    df = pd.DataFrame(rows)
    if apply_bias and "corr" in df.columns:   # corr first = the bias-corrected forecast
        cols = [c for c in df.columns if c != "corr"]
        if "Station" in cols:
            i = cols.index("Station") + 1
            cols = cols[:i] + ["corr"] + cols[i:]
        else:
            cols = ["corr"] + cols
        df = df[cols]
    rank = ("corr" if (apply_bias and "corr" in df.columns)
            else (RAIN_RANK if RAIN_RANK in df.columns else "mean"))
    if rank in df.columns:
        df = df.sort_values(rank, ascending=False, na_position="last").reset_index(drop=True)
    return df, window_label


def fetch_daily_temp(stations, historical=False, date_str=None):
    """Pull the model's TRUE daily max (temperature_2m_max), not max-of-hourly."""
    lats = ",".join(str(lat) for lat, lon in stations.values())
    lons = ",".join(str(lon) for lat, lon in stations.values())
    params = {"latitude": lats, "longitude": lons, "daily": "temperature_2m_max",
              "models": ",".join(MODELS), "timezone": TIMEZONE}
    if historical:
        params["start_date"] = date_str
        params["end_date"] = date_str
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    else:
        params["forecast_days"] = max(DAY_OFFSET, 0) + 3
        url = "https://api.open-meteo.com/v1/forecast"
    r = _get_with_retry(url, params=params)
    data = r.json()
    return data if isinstance(data, list) else [data]


def build_temp_table(data, stations, contest_date, bias=0.0, rank=TEMP_RANK, sea_breeze=None):
    target = contest_date.isoformat()
    rows, warned = [], set()
    excl_ecmwf = _temp_ecmwf_excluded()
    for name, loc in zip(stations.keys(), data):
        daily = loc.get("daily", {})
        times = daily.get("time", [])
        di = times.index(target) if target in times else (0 if times else None)
        sb_state = (sea_breeze or SEA_BREEZE_ONSET).get(name, "-")
        sb = STATION_TEMP_BIAS.get(name, 0.0)
        if USE_NO_SEABREEZE_BUMP and _no_seabreeze(sb_state):
            sb += NO_SEABREEZE_TEMP_BUMP.get(name, 0.0)   # hotter when the breeze fails to cap the max
        else:
            # NEW 07-Jul: breeze DID arrive but possibly late - ramp a partial
            # bump in proportional to lateness (see _seabreeze_lateness_bump).
            # Returns 0.0 for a normal on-time onset, so this is a no-op on
            # ordinary days.
            sb += _seabreeze_lateness_bump(name, sb_state)
        row, vals = {"Station": name}, []
        for mdl in MODELS:
            label = LABELS.get(mdl, mdl)
            series = daily.get(f"temperature_2m_max_{mdl}")
            if series is None:
                if mdl not in warned:
                    print(f"  !! temp model '{mdl}' returned no data.")
                    warned.add(mdl)
                row[label] = None; continue
            v = series[di] if (di is not None and di < len(series)) else None
            if v is not None:
                v = v + MODEL_STATION_BIAS.get((name, label), 0.0)
            row[label] = round(v + bias + sb, 1) if v is not None else None
            if row[label] is not None and not (excl_ecmwf and label == "ecmwf"):
                vals.append(row[label])   # excluded model still displays, just no vote
        if vals:
            row["mean"] = round(sum(vals) / len(vals), 1)
            row["spread"] = round(max(vals) - min(vals), 1)
            row["score"] = round(row["mean"] - CONF_WEIGHT * row["spread"], 1)
        row["sea breeze"] = sb_state
        rows.append(row)
    df = pd.DataFrame(rows)
    if "score" in df.columns:   # score ("Bias adjusted") + sea breeze first (after Station)
        cols = [c for c in df.columns if c not in ("score", "sea breeze")]
        i = cols.index("Station") + 1 if "Station" in cols else 0
        lead = [c for c in ("score", "sea breeze") if c in df.columns]
        cols = cols[:i] + lead + cols[i:]
        df = df[cols]
    sort_col = rank if rank in df.columns else ("score" if "score" in df.columns else "mean")
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=False, na_position="last").reset_index(drop=True)
    return df


def _fmt_delta(diff, kind):
    """Format a day-over-day delta for display, e.g. '70.0 mm more',
    '0.7\u00b0C less'."""
    if diff is None or pd.isna(diff):
        return "\u2013"
    unit = "mm" if kind == "rain" else "\u00b0C"
    if abs(diff) < (0.5 if kind == "rain" else 0.1):
        return "no change"
    direction = "more" if diff > 0 else "less"
    return f"{abs(diff):.1f} {unit} {direction}"


def add_delta_column(df, value_col, prev_df, kind):
    """Add a 'delta' column comparing value_col in df to the SAME station's
    value_col in prev_df - yesterday's frozen forecast for its own contest
    window, loaded via load_frozen(). This tracks day-over-day movement of
    the forecast itself (is a surge building or fading at this station?),
    not forecast-vs-actual skill. Always appended as the LAST column.
    Missing prior-day data or a station absent from it yields '\u2013'."""
    if prev_df is None or prev_df.empty or value_col not in df.columns:
        df["delta"] = "\u2013"
        return df
    prev_lookup = dict(zip(prev_df.get("Station", []), prev_df.get(value_col, [])))
    deltas = []
    for _, row in df.iterrows():
        pv = prev_lookup.get(row["Station"])
        cv = row.get(value_col)
        if pv is None or cv is None or pd.isna(pv) or pd.isna(cv):
            deltas.append("\u2013")
        else:
            deltas.append(_fmt_delta(cv - pv, kind))
    df["delta"] = deltas
    return df


def refine_temp_bias(forecast_df, actuals, rank_col=TEMP_RANK, alpha=TEMP_LEARN_RATE, source=""):
    """Compare corrected forecast vs actuals; print residuals + an updated
    STATION_TEMP_BIAS (EMA: new = old + alpha*(actual - corrected_forecast))."""
    if not actuals:
        return
    new_bias = dict(STATION_TEMP_BIAS)
    print(f"\nTemp bias refinement (rank col = {rank_col}, learn rate = {alpha})")
    if source:
        print(f"  scoring against: {source}")
        if "Historical API" in source:
            print("  WARNING: this is the hindsight API forecast, NOT a frozen run you")
            print("  picked from. Don't mix API-day and frozen-day residuals when tuning.")
    print(f"{'Station':<16}{'fcst':>7}{'actual':>8}{'resid':>7}   new bias")
    for _, r in forecast_df.iterrows():
        name = r["Station"]
        fc = r.get(rank_col)
        if name not in actuals or fc is None or pd.isna(fc):
            continue
        resid = round(actuals[name] - fc, 1)
        updated = round(STATION_TEMP_BIAS.get(name, 0.0) + alpha * resid, 1)
        new_bias[name] = updated
        print(f"{name:<16}{fc:>7.1f}{actuals[name]:>8.1f}{resid:>+7.1f}   {updated:+.1f}")
    items = ",\n    ".join(f'"{k}": {v}' for k, v in new_bias.items())
    print("\nPaste this back into the config:\nSTATION_TEMP_BIAS = {\n    " + items + ",\n}")
    return new_bias


def refine_rain_bias(forecast_df, actuals, base=RAIN_BIAS_BASE, alpha=RAIN_LEARN_RATE,
                     source="", surge_states=None):
    """Compare actual rainfall vs the BASE model and suggest updated MULTIPLICATIVE
    factors (EMA: new = old*(1-alpha) + (actual/base)*alpha). Skips near-zero base,
    where the ratio is unstable. Per-station cap overrides (RAIN_FACTOR_MAX_STATION)
    apply here too, so the Ghats toppers can learn past the global cap.

    Surge-aware: for a station in STATION_SURGE_FACTOR, if `surge_states` gives its
    regime for the scored day, the EMA updates ONLY that regime's sub-factor (surge
    or quiet) rather than the static STATION_RAIN_BIAS entry - so the two regimes
    learn independently. Prints a STATION_RAIN_BIAS block AND (if any fired) a
    STATION_SURGE_FACTOR block to paste back."""
    if not actuals:
        return
    new = dict(STATION_RAIN_BIAS)
    new_surge = {k: dict(v) for k, v in STATION_SURGE_FACTOR.items()}
    print(f"\nRain bias refinement (base = {base}, learn rate = {alpha}, multiplicative)")
    if source:
        print(f"  scoring against: {source}")
    print(f"{'Station':<16}{'base':>7}{'actual':>8}{'ratio':>7}{'regime':>9}   new factor")
    for _, r in forecast_df.iterrows():
        name = r["Station"]
        if name not in actuals:
            continue
        bv = r.get(base)
        if bv is None or pd.isna(bv):
            continue
        if bv < RAIN_MIN_BASE:
            print(f"{name:<16}{bv:>7.1f}{actuals[name]:>8.1f}    base<{RAIN_MIN_BASE}, skipped")
            continue
        ratio = actuals[name] / bv
        # route to the regime sub-factor if this is a surge-conditional station
        regime = ""
        if (USE_SURGE_FACTOR and surge_states and name in STATION_SURGE_FACTOR
                and name in surge_states):
            regime = surge_states[name][0]
            key = "surge" if regime == "surge" else "quiet"
            old = STATION_SURGE_FACTOR[name][key]
            updated = round(max(RAIN_FACTOR_MIN, min(_factor_cap(name),
                                                     old * (1 - alpha) + ratio * alpha)), 2)
            new_surge[name][key] = updated
        else:
            old = STATION_RAIN_BIAS.get(name, 1.0)
            updated = round(max(RAIN_FACTOR_MIN, min(_factor_cap(name),
                                                     old * (1 - alpha) + ratio * alpha)), 2)
            new[name] = updated
        print(f"{name:<16}{bv:>7.1f}{actuals[name]:>8.1f}{ratio:>7.2f}{regime:>9}   {updated}")
    items = ",\n    ".join(f'"{k}": {v}' for k, v in new.items())
    print("\nPaste this back into the config:\nSTATION_RAIN_BIAS = {\n    " + items + ",\n}")
    if surge_states and any(n in surge_states for n in STATION_SURGE_FACTOR):
        sitems = ",\n    ".join(f'"{k}": {v}' for k, v in new_surge.items())
        print("\nAnd the surge-conditional block:\nSTATION_SURGE_FACTOR = {\n    " + sitems + ",\n}")
    return new


def fetch_wind(stations, historical=False, date_str=None):
    """Pull hourly wind direction/speed (default/best-match model - sea breeze
    detection doesn't need multi-model comparison, just a reasonable single read)."""
    lats = ",".join(str(lat) for lat, lon in stations.values())
    lons = ",".join(str(lon) for lat, lon in stations.values())
    params = {"latitude": lats, "longitude": lons,
              "hourly": "winddirection_10m,windspeed_10m", "timezone": TIMEZONE}
    if historical:
        params["start_date"] = date_str
        params["end_date"] = date_str
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    else:
        params["forecast_days"] = max(DAY_OFFSET, 0) + 2
        url = "https://api.open-meteo.com/v1/forecast"
    r = _get_with_retry(url, params=params)
    data = r.json()
    return data if isinstance(data, list) else [data]


def compute_sea_breeze(wind_data, stations, contest_date):
    """Detect the actual onshore wind flip for contest_date from hourly wind
    direction/speed: first hour in the search window where direction falls in
    the onshore band AND speed clears the threshold, sustained into the next
    hour too (avoids flagging a single noisy reading). Returns 'not detected'
    for stations where no qualifying onshore flip occurs that day (common for
    deep-inland stations, or if synoptic winds override the local breeze)."""
    result = {}
    for name, loc in zip(stations.keys(), wind_data):
        hourly = loc.get("hourly", {})
        times = [datetime.fromisoformat(t) for t in hourly.get("time", [])]
        dirs = hourly.get("winddirection_10m", [])
        speeds = hourly.get("windspeed_10m", [])
        day_idx = [i for i, t in enumerate(times) if t.date() == contest_date]
        onset = None
        for pos, i in enumerate(day_idx):
            t = times[i]
            if not (SEABREEZE_SEARCH_START <= t.hour <= SEABREEZE_SEARCH_END):
                continue
            d = dirs[i] if i < len(dirs) else None
            s = speeds[i] if i < len(speeds) else None
            if d is None or s is None:
                continue
            onshore = SEABREEZE_DIR_MIN <= d <= SEABREEZE_DIR_MAX
            if onshore and s >= SEABREEZE_SPEED_MIN:
                nxt = day_idx[pos + 1] if pos + 1 < len(day_idx) else None
                nxt_ok = (nxt is not None and nxt < len(dirs) and dirs[nxt] is not None
                          and SEABREEZE_DIR_MIN <= dirs[nxt] <= SEABREEZE_DIR_MAX)
                if nxt_ok:
                    onset = t
                    break
        result[name] = f"{onset:%H:%M} IST" if onset else "No Sea Breeze today"
    return result



def build_guidance_html(temp_df, tn_df, rain_df, chennai_rain_prob=None, imd_rain_df=None, avalanche_surge=None, surge_states=None, extra_html=""):
    """Quick-glance pick summary, pulled from tables the script already built -
    no extra fetches. Rain tables are already sorted descending by their rank
    column (corr if bias-applied, else RAIN_RANK), so top rows = top rows."""

    def row_for(df, station):
        m = df[df["Station"] == station]
        return m.iloc[0] if not m.empty else None

    def temp_line(station):
        r = row_for(temp_df, station)
        if r is None or pd.isna(r.get("score")):
            return f"<b>{station}</b>: no data"
        sb = r.get("sea breeze", "-")
        sb_text = sb if sb == "No Sea Breeze today" else f"sea breeze {sb}"
        return f"<b>{station}</b>: {r['score']:.1f}&deg;C &middot; {sb_text}"

    chennai_temps = "<br>".join(temp_line(s) for s in ("Nungambakkam", "Meenambakkam"))

    top_temps = temp_df.sort_values("score", ascending=False).head(3)
    tn_temp_top3 = "<br>".join(
        f"{i+1}. <b>{r['Station']}</b> &ndash; {r['score']:.1f}&deg;C"
        for i, (_, r) in enumerate(top_temps.iterrows()) if not pd.isna(r.get("score"))
    ) or "no data"

    def rain_top3(df):
        rank_col = "corr" if "corr" in df.columns else (RAIN_RANK if RAIN_RANK in df.columns else "mean")
        top = df.head(3)
        lines = []
        for i, (_, r) in enumerate(top.iterrows()):
            v = r.get(rank_col)
            vs = f"{v:.1f}" if v is not None and not pd.isna(v) else "&ndash;"
            lines.append(f"{i+1}. <b>{r['Station']}</b> &ndash; {vs} mm")
        return "<br>".join(lines) or "no data"

    tn_rain_top3 = rain_top3(tn_df)
    airf_rain_top3 = rain_top3(rain_df)

    prob_html = (f'<div class="g-label">Chennai rain probability</div>'
                 f'<div class="g-body">{chennai_rain_prob}%</div>'
                 if chennai_rain_prob is not None else "")

    imd_count, imd_names, imd_breakdown = count_rain_stations(imd_rain_df) if imd_rain_df is not None else (0, [], [])
    imd_line = f"Obs: {imd_count} ({', '.join(imd_names)})" if imd_names else f"Obs: {imd_count}"
    imd_detail = "<br>".join(
        f"{b['Station']}: ecmwf {b['ecmwf']:.1f} &middot; gfs {b['gfs']:.1f} &middot; icon {b['icon']:.1f}"
        if b['ecmwf'] is not None and b['gfs'] is not None and b['icon'] is not None else f"{b['Station']}: incomplete data"
        for b in imd_breakdown
    )

    if avalanche_surge is None:
        surge_line = "no data (live fetch unavailable)"
    else:
        t = avalanche_surge["today"]
        surge_line = f"{t['speed']} m/s @ {t['dir']}&deg;, RH {t['rh']}% &mdash; {avalanche_surge['verdict']}"
        if avalanche_surge["trend"]:
            surge_line += f" ({avalanche_surge['trend']})"

    # Surge-conditional factor decisions (which regime factor fired today).
    sf_html = ""
    if surge_states:
        def _base_for(nm):
            for d in (tn_df, rain_df):
                if d is not None and "Station" in d and "ecmwf" in d:
                    hit = d[d["Station"] == nm]
                    if not hit.empty and not pd.isna(hit.iloc[0]["ecmwf"]):
                        return hit.iloc[0]["ecmwf"]
            return None
        lines = []
        for nm, (st, det) in surge_states.items():
            fac = STATION_SURGE_FACTOR[nm]["surge" if st == "surge" else "quiet"]
            base = _base_for(nm)
            eff = _gate_quiet(fac, base)
            colour = "#f0a020" if st == "surge" else "#56b6ff"
            cluster_tag = f" &middot; {det['cluster']} cluster" if det.get("cluster") else ""
            line = (f"<b>{nm}</b>: <span style='color:{colour}'>{st}</span> "
                    f"&rarr; factor {fac} (850 {det['speed']} m/s @ {det['dir']}&deg;{cluster_tag})")
            if base is not None and abs(eff - fac) > 0.01:
                line += (f" &middot; <span style='color:#9aa3ad'>gated to {eff:.2f} "
                         f"on {base:.0f}mm base</span>")
            lines.append(line)
        sf_html = ('<div class="g-label">Surge-conditional factors</div>'
                   f'<div class="g-body">{"<br>".join(lines)}</div>')

    return (
        '<div class="guidance"><h2>&#127919; Guidance</h2>'
        '<div class="g-label">Chennai temps</div>'
        f'<div class="g-body">{chennai_temps}</div>'
        '<div class="g-label">TN temperature &ndash; top 3</div>'
        f'<div class="g-body">{tn_temp_top3}</div>'
        + prob_html
        + '<div class="g-label">TN rainfall toppers &ndash; top 3</div>'
        f'<div class="g-body">{tn_rain_top3}</div>'
        '<div class="g-label">All-India rainfall toppers &ndash; top 3</div>'
        f'<div class="g-body">{airf_rain_top3}</div>'
        f'<div class="g-label">IMD TN stations &ndash; rain forecast (&ge;{IMD_RAIN_THRESH}mm)</div>'
        f'<div class="g-body">{imd_line}</div>'
        + (f'<div class="g-body" style="font-size:11px;color:#9aa3ad;margin-top:4px">{imd_detail}</div>' if imd_detail else '')
        + '<div class="g-label">Avalanche surge check (850hPa)</div>'
        + f'<div class="g-body">{surge_line}</div>'
        + sf_html
        + extra_html
        + '</div>'
    )



def count_rain_stations(df, thresh=IMD_RAIN_THRESH):
    """Count + list IMD bulletin stations forecast to clear the rain threshold,
    using an ECMWF+GFS mean only - ICON excluded (unreliable for TN rainfall).
    Also returns a per-station ecmwf/gfs/icon breakdown for the flagged
    stations, so a call can be sanity-checked against what each model actually
    showed (e.g. one model spiking alone vs genuine multi-model agreement)."""
    if df is None or df.empty:
        return 0, [], []
    cols = [c for c in ("ecmwf", "gfs") if c in df.columns]
    if not cols:
        return 0, [], []
    ge_mean = df[cols].mean(axis=1, skipna=True)
    tmp = df.assign(_ge_mean=ge_mean)
    hits = tmp[tmp["_ge_mean"].notna() & (tmp["_ge_mean"] >= thresh)]
    hits = hits.sort_values("_ge_mean", ascending=False)
    breakdown = [
        {"Station": r["Station"], "ecmwf": r.get("ecmwf"), "gfs": r.get("gfs"), "icon": r.get("icon")}
        for _, r in hits.iterrows()
    ]
    return len(hits), list(hits["Station"]), breakdown



def fetch_run_info():
    out, now = {}, datetime.now(timezone.utc)
    for mdl in MODELS:
        label = LABELS.get(mdl, mdl)
        domain = META_DOMAINS.get(mdl, mdl)
        try:
            meta = _get_with_retry(f"https://api.open-meteo.com/data/{domain}/static/meta.json",
                                   timeout=15, attempts=2).json()
            ts = meta.get("last_run_initialisation_time")
            if ts is None:
                out[label] = "run unavailable (check domain)"; continue
            init = datetime.fromtimestamp(ts, tz=timezone.utc)
            age_h = (now - init).total_seconds() / 3600
            avail = meta.get("last_run_availability_time")
            astr = ""
            if avail:
                ap = datetime.fromtimestamp(avail, tz=ZoneInfo("America/Los_Angeles"))
                astr = f", live {ap:%H:%M %Z}"
            out[label] = f"{init:%Y-%m-%d %HZ} ({age_h:.0f}h ago{astr})"
        except Exception as e:
            out[label] = f"run unavailable ({type(e).__name__})"
    return out


# ---------------------------- HTML rendering ----------------------------------
# Stations to visually pin in every table (blue left-bar + tint), so you can spot
# your must-watch rows without hunting - they still sort in their normal position.
HIGHLIGHT_STATIONS = {"Meenambakkam", "Nungambakkam"}

CSS = """
<style>
.cr{font-family:-apple-system,Segoe UI,Roboto,Helvetica,sans-serif;
    background:#0f1115;color:#e6e6e6;padding:12px;border-radius:8px;max-width:760px}
.cr h1{font-size:18px;margin:0 0 4px} .cr h2{font-size:15px;margin:22px 0 4px;color:#9ecbff}
.cr .meta{font-size:12px;color:#9aa3ad;line-height:1.5;margin-bottom:6px}
.cr table{border-collapse:collapse;width:100%;font-size:13px}
.cr th,.cr td{padding:6px 8px;text-align:right;border-bottom:1px solid #262a33;white-space:nowrap}
.cr th{background:#1a1d24;color:#cfd2d6;position:sticky;top:0}
.cr td:first-child,.cr th:first-child{text-align:left;white-space:normal}
.cr tr:nth-child(even) td{background:#14171f}
.cr tr.hl td{background:rgba(158,203,255,.14) !important}
.cr tr.hl td:first-child{border-left:3px solid #9ecbff;font-weight:700;color:#cfe3ff}
.cr .surge-banner{margin:8px 0 12px;padding:9px 12px;border-radius:8px;
    background:#0d1826;border:1px solid #2b5a8c;color:#cfe3ff;font-size:13px;line-height:1.55}
.cr .surge-banner b{color:#9ecbff}
.cr .mean{font-weight:700}
.cr .score{font-weight:700;color:#7ee787;border-left:2px solid #2ea043}
.cr .corr{font-weight:700;color:#56d4dd;border-left:2px solid #1f9aa3}
.cr .climo{font-weight:700;color:#f0a020;border-left:2px solid #d29922}
.cr .legend{font-size:11px;color:#8a929c;margin-top:8px}
.cr .guidance{margin-top:18px;padding:12px 14px;border:1px solid #2ea043;border-radius:8px;
              background:#101a12}
.cr .guidance h2{color:#7ee787;margin:0 0 10px;font-size:15px}
.cr .guidance .g-label{color:#9aa3ad;font-size:11px;text-transform:uppercase;
                        letter-spacing:.04em;margin:10px 0 3px}
.cr .guidance .g-label:first-of-type{margin-top:0}
.cr .guidance .g-body{font-size:13px;line-height:1.6}
</style>
"""


def _fmt(v):
    return "&ndash;" if v is None or (isinstance(v, float) and pd.isna(v)) else f"{v:.1f}"


def _spread_bg(v):
    if v is None or pd.isna(v): return ""
    if v < 10:  return "background:rgba(46,160,67,.30)"
    if v < 30:  return "background:rgba(210,153,34,.32)"
    return "background:rgba(248,81,73,.35)"


def html_table(df, kind):
    display_headers = {"score": "Bias adjusted"}   # header text only - internal column key stays 'score'
    cols = list(df.columns)
    head = "".join(f"<th>{display_headers.get(c, c)}</th>" for c in cols)
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c] if c in row else None
            if c == "Station":
                cells.append(f"<td>{v}</td>")
            elif c == "spread" and kind == "rain":
                cells.append(f'<td style="{_spread_bg(v)}">{_fmt(v)}</td>')
            elif c == "mean":
                style = ""
                if kind == "rain" and v is not None and not pd.isna(v):
                    a = min(v / 80, 1) * 0.55
                    style = f'background:rgba(56,139,253,{a:.2f})'
                cells.append(f'<td class="mean" style="{style}">{_fmt(v)}</td>')
            elif c == "max":
                floored = (kind == "rain" and USE_CLIMO_FLOOR
                           and row.get("Station") in CLIMO_FLOOR
                           and v is not None and not pd.isna(v)
                           and abs(v - CLIMO_FLOOR[row["Station"]]) < 0.05)
                # warn when climo floor is overriding models by >2x (models may be right)
                floor_val = CLIMO_FLOOR.get(row.get("Station")) if kind == "rain" else None
                model_max = row.get("max") if not floored else None
                # get raw model max before floor was applied (stored in mean/ecmwf columns)
                raw_vals = [row.get(m) for m in ("ecmwf","gfs","icon") if row.get(m) is not None]
                raw_model_max = max(raw_vals) if raw_vals else None
                climo_warning = (floored and floor_val is not None and raw_model_max is not None
                                 and floor_val > raw_model_max * 2)
                display_val = _fmt(v)
                if climo_warning:
                    display_val += " ⚠️"
                cells.append(f'<td class="{"climo" if floored else "score"}" '
                             f'title="{"Climo floor overriding models by >2x — models may be right" if climo_warning else ""}">'
                             f'{display_val}</td>')
            elif c == "corr":
                cells.append(f'<td class="corr">{_fmt(v)}</td>')
            elif c == "score":
                cells.append(f'<td class="score">{_fmt(v)}</td>')
            elif c == "sea breeze":
                cells.append(f'<td style="white-space:nowrap;color:#9ecbff">{v if v else "-"}</td>')
            elif c == "delta":
                color = ("color:#f0a020" if isinstance(v, str) and "more" in v
                         else "color:#56b6ff" if isinstance(v, str) and "less" in v
                         else "color:#8a929c")
                cells.append(f'<td style="white-space:nowrap;{color}">{v}</td>')
            else:
                cells.append(f"<td>{_fmt(v)}</td>")
        station = row["Station"] if "Station" in row else None
        tr = '<tr class="hl">' if station in HIGHLIGHT_STATIONS else "<tr>"
        body.append(tr + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_report(run_info, window_label, sections, bt_date=None, guidance_html=None, banner_html=""):
    now = datetime.now(ZoneInfo(TIMEZONE))
    runs = "<br>".join(f"<b>{k}</b>: {v}" for k, v in run_info.items())
    parts = [CSS, '<div class="cr">',
             "<h1>Contest forecast</h1>",
             f'<div class="meta">Window: <b>{window_label}</b><br>'
             f'Generated {now:%Y-%m-%d %H:%M} IST<br>{runs}</div>']
    if banner_html:
        parts.append(banner_html)
    for title, df, kind in sections:
        parts.append(f"<h2>{title}</h2>")
        parts.append(html_table(df, kind))
    if bt_date:
        parts.append(f'<div class="meta" style="margin-top:14px">Backtest source: '
                     f'Historical Forecast API &middot; compare vs IMD actuals.</div>')
    rank_desc = ("<b>corr</b> = per-station bias-corrected estimate (teal)"
                 if USE_RAIN_BIAS else f"<b>{RAIN_RANK}</b>")
    parts.append(f'<div class="legend">Rainfall ranked by {rank_desc}. '
                 'max = highest single model (catches orographic spikes the mean buries). '
                 'corr = ECMWF &times; learned factor (dampened for west coast/Ghats to keep ECMWF dominant; '
                 'a few chronic under-forecast toppers carry a raised per-station factor cap). '
                 'Amber max = climatological floor active. '
                 '⚠️ = climo floor overriding models by &gt;2x — check map, models may be right. '
                 'Temp ranked by Bias adjusted. spread shaded green&rarr;red (red = models disagree). '
                 'sea breeze = live-detected onshore wind flip for that day (falls back to '
                 'climatological estimate if the wind fetch fails) - coastal stations flip '
                 'earliest/strongest, inland stations later/weaker; "No Sea Breeze today" = no '
                 'qualifying onshore flip detected that day. A LATE onset (still detected, just '
                 'past the normal window) gets a partial no-breeze bump ramped by how many hours '
                 'late it is (full bump at 4h+ late). '
                 'delta = change vs yesterday\'s forecast for that station (amber = building, '
                 'blue = fading) - tracks day-over-day forecast movement, not forecast accuracy. '
                 '&ndash; = no data / no prior-day forecast to compare.</div>')
    if guidance_html:
        parts.append(guidance_html)
    parts.append('</div>')
    return "".join(parts)


def mount_drive():
    """Mount Google Drive for persistence; fall back to local /content if unavailable."""
    global FROZEN_DIR
    if not USE_DRIVE:
        FROZEN_DIR = "/content/contest_frozen"
        return
    try:
        import contextlib, io
        from google.colab import drive
        if VERBOSE:
            drive.mount("/content/drive")
        else:   # swallow Colab's "Mounted at /content/drive" line
            with contextlib.redirect_stdout(io.StringIO()):
                drive.mount("/content/drive")
    except Exception as e:
        FROZEN_DIR = "/content/contest_frozen"
        print(f"Drive not mounted ({type(e).__name__}); frozen files won't persist across sessions.")


def freeze_snapshot(date_str, run_info, window, tables):
    """Save the exact forecast tables (what you picked from) for a contest date."""
    os.makedirs(FROZEN_DIR, exist_ok=True)
    payload = {
        "contest_date": date_str,
        "frozen_at": datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds"),
        "runs": run_info,
        "window": window,
        "tables": {k: v.to_dict(orient="records") for k, v in tables.items()},
    }
    with open(os.path.join(FROZEN_DIR, f"{date_str}.json"), "w") as f:
        json.dump(payload, f)


def load_frozen(date_str):
    """Return (payload, {name: DataFrame}) for a frozen date, or None if not saved."""
    p = os.path.join(FROZEN_DIR, f"{date_str}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        payload = json.load(f)
    tables = {k: pd.DataFrame(v) for k, v in payload["tables"].items()}
    return payload, tables


# --- AIFS via ECMWF's DEDICATED endpoint (isolated test, NOT wired into the ---
# --- main report yet - run test_aifs() by hand and confirm real numbers    ---
# --- come back before merging this into build_table()/build_temp_table(). ---
# Prior attempt failed because "ecmwf_aifs025" isn't served on the shared
# /v1/forecast endpoint (the one MODELS uses) - AIFS is scoped to ECMWF's own
# /v1/ecmwf endpoint instead (confirmed via Open-Meteo's docs: that endpoint's
# model dropdown lists "ECMWF IFS HRES 9km", "ECMWF IFS 0.25 deg", and
# "ECMWF AIFS 0.25 deg Single"). This fetches from THAT endpoint. Unverified:
# the exact model= string here is a best-effort match to Open-Meteo's naming
# convention, not something fetchable/testable from this sandbox (no live
# network access here beyond pages already linked in conversation) - hence
# the standalone test function below instead of wiring it in blind again.
AIFS_ENDPOINT = "https://api.open-meteo.com/v1/ecmwf"
AIFS_MODEL = "ecmwf_aifs025"


def fetch_aifs_precip(stations):
    """Hourly precipitation from AIFS via ECMWF's dedicated endpoint."""
    lats = ",".join(str(lat) for lat, lon in stations.values())
    lons = ",".join(str(lon) for lat, lon in stations.values())
    params = {"latitude": lats, "longitude": lons, "hourly": "precipitation",
              "models": AIFS_MODEL, "timezone": TIMEZONE,
              "forecast_days": max(DAY_OFFSET, 0) + 3, "precipitation_unit": "mm"}
    r = _get_with_retry(AIFS_ENDPOINT, params=params)
    data = r.json()
    return data if isinstance(data, list) else [data]


def fetch_aifs_daily_temp(stations):
    """Daily temperature_2m_max from AIFS via ECMWF's dedicated endpoint."""
    lats = ",".join(str(lat) for lat, lon in stations.values())
    lons = ",".join(str(lon) for lat, lon in stations.values())
    params = {"latitude": lats, "longitude": lons, "daily": "temperature_2m_max",
              "models": AIFS_MODEL, "timezone": TIMEZONE,
              "forecast_days": max(DAY_OFFSET, 0) + 3}
    r = _get_with_retry(AIFS_ENDPOINT, params=params)
    data = r.json()
    return data if isinstance(data, list) else [data]


def test_aifs(stations=None):
    """STANDALONE diagnostic - run this alone in a Colab cell:  test_aifs()
    Prints raw AIFS precipitation + daily max temp for a couple of stations so
    you can SEE whether real numbers come back before we touch the main report.
    A working result looks like real mm/degC values with today's date in the
    time array. A broken result looks like empty lists, all-null values, or an
    HTTP error printed by _get_with_retry (400 = wrong model/endpoint string,
    which would mean the endpoint fix still needs adjusting)."""
    test_stations = stations or {"Mahabaleshwar (MH)": STATIONS["Mahabaleshwar (MH)"],
                                 "Nungambakkam": TEMP_STATIONS["Nungambakkam"]}
    print(f"Testing AIFS via {AIFS_ENDPOINT} (models={AIFS_MODEL})\n")
    try:
        precip = fetch_aifs_precip(test_stations)
        for name, loc in zip(test_stations.keys(), precip):
            h = loc.get("hourly", {})
            times = h.get("time", [])
            series = h.get(f"precipitation_{AIFS_MODEL}", h.get("precipitation"))
            if not times or series is None:
                print(f"  RAIN  {name}: NO DATA - keys returned: {list(h.keys())}")
            else:
                today_vals = series[:24] if series else []
                print(f"  RAIN  {name}: OK - first day, e.g. {times[0]}={today_vals[0] if today_vals else '?'}, "
                     f"24h sample sum={sum(v for v in today_vals if v is not None):.1f}mm")
    except Exception as e:
        print(f"  RAIN  fetch FAILED: {type(e).__name__}: {e}")

    try:
        temp = fetch_aifs_daily_temp(test_stations)
        for name, loc in zip(test_stations.keys(), temp):
            d = loc.get("daily", {})
            times = d.get("time", [])
            series = d.get(f"temperature_2m_max_{AIFS_MODEL}", d.get("temperature_2m_max"))
            if not times or series is None:
                print(f"  TEMP  {name}: NO DATA - keys returned: {list(d.keys())}")
            else:
                print(f"  TEMP  {name}: OK - {times[0]}={series[0]}\u00b0C ... {times[-1]}={series[-1]}\u00b0C")
    except Exception as e:
        print(f"  TEMP  fetch FAILED: {type(e).__name__}: {e}")

    print("\nIf both show OK with sane mm/\u00b0C values dated today or later, AIFS is reachable -"
         " tell me and I'll wire it into the main tables as a 4th column.")
    print("If either shows NO DATA or FAILED, paste the output back and we'll adjust the model/endpoint string.")


# --- ECMWF-vs-other-models scoreboard ----------------------------------------
def _scoreboard_path():
    return os.path.join(FROZEN_DIR, SCOREBOARD_FILE_NAME)


def _load_scoreboard():
    p = _scoreboard_path()
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return json.load(f)


def _save_scoreboard(entries):
    os.makedirs(FROZEN_DIR, exist_ok=True)
    with open(_scoreboard_path(), "w") as f:
        json.dump(entries, f, indent=1)


def log_scoreboard(date_str, temp_df, actuals):
    """Append one entry per SCOREBOARD_STATIONS station: raw ECMWF vs the mean of
    the OTHER models, scored against `actuals`, for `date_str`. Reads the table's
    'ecmwf'/'gfs'/'icon' columns (these already carry the common station bias +
    sea-breeze bump, so the comparison between models stays apples-to-apples -
    only MODEL_STATION_BIAS differs between them). Idempotent per (date, station);
    safe to call every run. No-op if LOG_SCOREBOARD is off or actuals is empty."""
    if not LOG_SCOREBOARD or not actuals or temp_df is None or temp_df.empty:
        return
    entries = _load_scoreboard()
    have = {(e["date"], e["station"]) for e in entries}
    changed = False
    for name in SCOREBOARD_STATIONS:
        if name not in actuals or (date_str, name) in have:
            continue
        hit = temp_df[temp_df["Station"] == name]
        if hit.empty:
            continue
        r = hit.iloc[0]
        ecmwf, gfs, icon = r.get("ecmwf"), r.get("gfs"), r.get("icon")
        other = [v for v in (gfs, icon) if v is not None and not pd.isna(v)]
        if ecmwf is None or pd.isna(ecmwf) or not other:
            continue
        other_mean = sum(other) / len(other)
        actual = actuals[name]
        entries.append({
            "date": date_str, "station": name,
            "ecmwf": round(ecmwf, 1), "other_mean": round(other_mean, 1),
            "actual": round(actual, 1),
            "ecmwf_err": round(ecmwf - actual, 1),
            "other_err": round(other_mean - actual, 1),
        })
        changed = True
    if changed:
        _save_scoreboard(entries)


def _scoreboard_tally(entries):
    ecmwf_wins = other_wins = ties = 0
    for e in entries:
        d = abs(e["ecmwf_err"]) - abs(e["other_err"])
        if d < -0.05:
            ecmwf_wins += 1
        elif d > 0.05:
            other_wins += 1
        else:
            ties += 1
    return ecmwf_wins, other_wins, ties


def print_scoreboard(stations=None):
    """Print the full day-by-day ECMWF-vs-other-models log + tally. Call directly
    in a cell:  print_scoreboard()  or  print_scoreboard(["Meenambakkam"])"""
    entries = _load_scoreboard()
    if stations:
        entries = [e for e in entries if e["station"] in stations]
    if not entries:
        print("No scoreboard entries yet.")
        return
    entries.sort(key=lambda e: (e["station"], e["date"]))
    print(f"{'Date':<12}{'Station':<16}{'ecmwf':>7}{'other':>7}{'actual':>8}"
         f"{'ecmwf_err':>11}{'other_err':>11}  winner")
    for e in entries:
        d = abs(e["ecmwf_err"]) - abs(e["other_err"])
        w = "ecmwf" if d < -0.05 else "other" if d > 0.05 else "tie"
        print(f"{e['date']:<12}{e['station']:<16}{e['ecmwf']:>7.1f}{e['other_mean']:>7.1f}"
             f"{e['actual']:>8.1f}{e['ecmwf_err']:>+11.1f}{e['other_err']:>+11.1f}  {w}")
    ew, ow, tw = _scoreboard_tally(entries)
    print(f"\nTally: ECMWF closer {ew}x, other-models-mean closer {ow}x, tie {tw}x (n={len(entries)})")


def scoreboard_summary_html(stations=None, label="Chennai (Meena/Nunga)"):
    """Short guidance-panel line: running win/loss tally, pointer to the full log."""
    entries = _load_scoreboard()
    if stations:
        entries = [e for e in entries if e["station"] in stations]
    if not entries:
        return ""
    ew, ow, tw = _scoreboard_tally(entries)
    return (f'<div class="g-label">ECMWF scoreboard &ndash; {label}</div>'
           f'<div class="g-body">ECMWF closer <b>{ew}x</b> &middot; '
           f'other-models-mean closer <b>{ow}x</b> &middot; tie {tw}x (n={len(entries)}) '
           f'<span style="color:#9aa3ad">&middot; print_scoreboard() for the day-by-day log</span></div>')


if __name__ == "__main__":
    mount_drive()
    live_date = datetime.now(ZoneInfo(TIMEZONE)).date() + timedelta(days=DAY_OFFSET)
    run_info = fetch_run_info()

    # Detect the 850hPa surge regime for surge-conditional stations (Avalanche,
    # Chinnakallar) so their bias factor flips with the synoptic state today.
    # (compute_surge_states logs the decision itself when VERBOSE; the regime also
    # appears in the report's guidance panel.)
    surge_states = compute_surge_states({**STATIONS, **TN_RAIN_STATIONS}, live_date)

    rain_df, win = build_table(fetch(STATIONS, "precipitation"),
                               STATIONS, "precipitation", "sum", live_date,
                               surge_states=surge_states)
    tn_df, _ = build_table(fetch(TN_RAIN_STATIONS, "precipitation"),
                           TN_RAIN_STATIONS, "precipitation", "sum", live_date,
                           surge_states=surge_states)
    imd_rain_df, _ = build_table(fetch(IMD_TN_STATIONS, "precipitation"),
                                 IMD_TN_STATIONS, "precipitation", "sum", live_date)
    try:
        live_sea_breeze = compute_sea_breeze(fetch_wind(TEMP_STATIONS), TEMP_STATIONS, live_date)
    except Exception as e:
        print(f"Live sea breeze fetch failed ({type(e).__name__}); using climatological fallback.")
        live_sea_breeze = None
    temp_df = build_temp_table(fetch_daily_temp(TEMP_STATIONS), TEMP_STATIONS,
                               live_date, bias=TEMP_BIAS, rank=TEMP_RANK,
                               sea_breeze=live_sea_breeze)
    bias_tag = f", +{TEMP_BIAS}" if TEMP_BIAS else ""
    if _temp_ecmwf_excluded():
        bias_tag += f", ecmwf excluded until {TEMP_EXCLUDE_ECMWF_UNTIL}"

    # Day-over-day delta: compare each station's forecast to yesterday's
    # frozen forecast for its own contest window (tracks surge movement,
    # not accuracy). Silently skipped (all '-') if no prior-day snapshot exists.
    prev_date_str = (live_date - timedelta(days=1)).isoformat()
    prev_frozen = load_frozen(prev_date_str)
    prev_tables = prev_frozen[1] if prev_frozen else {}
    rain_value_col = "corr" if "corr" in rain_df.columns else RAIN_RANK
    tn_value_col = "corr" if "corr" in tn_df.columns else RAIN_RANK
    rain_df = add_delta_column(rain_df, rain_value_col, prev_tables.get("rainfall"), "rain")
    tn_df = add_delta_column(tn_df, tn_value_col, prev_tables.get("tn_rainfall"), "rain")
    temp_df = add_delta_column(temp_df, TEMP_RANK, prev_tables.get("temp"), "temp")

    sections = [("Rainfall (mm)", rain_df, "rain"),
                ("TN rainfall (mm)", tn_df, "rain"),
                (f"Max temp (\u00b0C, daily max, rank={TEMP_RANK}{bias_tag})", temp_df, "temp")]

    # Freeze the live forecast you're picking from (last run before deadline wins).
    if FREEZE_FORECAST:
        try:
            freeze_snapshot(live_date.isoformat(), run_info, win,
                            {"rainfall": rain_df, "tn_rainfall": tn_df, "temp": temp_df,
                             "imd_rain": imd_rain_df})
        except Exception as e:
            print(f"Freeze skipped: {type(e).__name__}")

    # Pinned-18Z backtest (rainfall, TN rainfall, temp - all from the SAME
    # run), replacing the old BACKTEST_DATE frozen-snapshot/Historical API
    # mechanism entirely. That old path could silently show whatever run
    # happened to be live at freeze time (e.g. a 06Z GFS run) rather than a
    # consistent 18Z pin, which is what actually caused the confusing
    # "Backtest rainfall (FROZEN run, gfs ... 06Z)" table on 08-Jul. Only
    # runs automatically if SINGLE_RUN_BACKTEST is set; otherwise call
    # backtest_rain_single_run(...) / backtest_temp_single_run(...) by hand.
    if SINGLE_RUN_BACKTEST:
        try:
            _run_iso = (_default_18z_run_iso() if SINGLE_RUN_BACKTEST == "AUTO"
                       else SINGLE_RUN_BACKTEST)
            backtest_rain_single_run(_run_iso, target_date=live_date)
            backtest_temp_single_run(_run_iso, target_date=live_date)
        except Exception as e:
            print(f"Pinned-run backtest skipped ({type(e).__name__}: {e})")

    chennai_rain_prob = CHENNAI_RAIN_PROB_MANUAL
    if _chennai_rain_analyze is not None:
        try:
            chennai_rain_prob = _chennai_rain_analyze().get("final_pop")
        except Exception as e:
            print(f"Live Chennai rain probability fetch failed ({type(e).__name__}); "
                  f"falling back to manual value.")

    avalanche_surge = analyze_avalanche_surge(live_date)

    banner_html = surge_movement_banner_html(surge_movement(live_date))

    guidance_html = build_guidance_html(temp_df, tn_df, rain_df, chennai_rain_prob=chennai_rain_prob,
                                        imd_rain_df=imd_rain_df, avalanche_surge=avalanche_surge,
                                        surge_states=surge_states,
                                        extra_html=scoreboard_summary_html(SCOREBOARD_STATIONS))
    html = build_report(run_info, win, sections, guidance_html=guidance_html,
                        banner_html=banner_html)

    with open(OUTFILE, "w") as f:
        f.write("<!doctype html><meta name='viewport' "
                "content='width=device-width,initial-scale=1'>" + html)
    _log(f"Saved {OUTFILE} (open from the Colab Files panel on the left).")

    try:
        from IPython.display import HTML, display
        display(HTML(html))
    except Exception:
        print("Open the saved HTML file to view the tables.")
