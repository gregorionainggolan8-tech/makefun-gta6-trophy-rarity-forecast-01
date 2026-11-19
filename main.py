"""Perkiraan rarity trophy GTA VI (PS5) di Exophase pada target case.

Cara kerja, singkatnya:
  * Untuk tiap trophy, ubah riwayat earn-rate menjadi titik-titik
    (waktu-sejak-rilis, logit(rate)) lalu pasang garis lurus terhadap
    log-waktu: logit(rate) ~ a + b * ln(1 + jam sejak rilis).
    Garis ini menjenuh secara alami (rate menuju 100% ketika b > 0)
    dan mengikuti tren yang benar-benar teramati di history, bukan
    sekadar mengulang rate terakhir.
  * Pembacaan terbaru diberi bobot lebih (paruh bobot 24 jam), karena
    tren minggu peluncuran berubah cepat: awalnya pemain baru membanjiri
    dan menekan rate ke bawah, belakangan perolehan trophy yang menang.
  * Kalau history belum memberi info (mis. semua rate sama karena halaman
    belum di-refresh), kemiringan garis diambil dari prior per tier:
    platinum nyaris datar, bronze cerita naik cepat, gold/silver di tengah.
    Makin banyak snapshot berbeda di history, makin besar bobot data
    dibanding prior.
  * Lebar interval (dalam logit) tumbuh bersama horizon, ketidakpastian
    kemiringan, sisa (residual) fit, guncangan refresh yang tak terlihat,
    dan penalti bila history datar/pendek.

Hanya pustaka standar Python 3.13, tanpa jaringan, tanpa acak: dua run
pada case yang sama menghasilkan byte yang sama persis.
"""

import json
import math
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Level kuantil yang diminta case; dipakai bila case tidak menyebutkannya.
LEVELS_DEFAULT = [0.05, 0.25, 0.5, 0.75, 0.95]

# Batas logit: halaman mencetak 2 desimal, 0% dan 100% tidak punya logit.
LO = 0.01
HI = 99.99
LOGIT_LO = math.log(LO / (100.0 - LO))
LOGIT_HI = math.log(HI / (100.0 - HI))

# Prior kemiringan garis logit-vs-log-waktu, per tier (satuan logit per
# satuan ln(1 + jam sejak rilis)). Alasan: platinum butuh menamatkan
# segalanya sehingga nyaris tidak bergerak dalam seminggu; trophy cerita
# bronze diselesaikan banyak pemain di hari-hari pertama; gold/silver
# berada di antaranya. Ini prior, bukan jawaban: data yang informatif
# selalu mengalahkannya.
PRIOR_SLOPE = {
    "platinum": 0.15,
    "gold": 0.60,
    "silver": 0.80,
    "bronze": 1.10,
}
PRIOR_SLOPE_FALLBACK = 0.90

# Median darurat bila sebuah trophy tidak punya satu pun rate di history.
TIER_FALLBACK_RATE = {
    "platinum": 0.20,
    "gold": 0.50,
    "silver": 0.50,
    "bronze": 1.00,
}

# Bobot prior dibanding data (dalam satuan "snapshot berbeda").
PRIOR_WEIGHT = 3.0
# Ketidakpastian kemiringan bila data tidak memberi info sama sekali.
PRIOR_SLOPE_SD = 0.80
# Penyebaran gerak acak pada skala logit, per akar hari.
WALK_SD_PER_SQRT_DAY = 0.50
# Guncangan satu refresh Exophase yang tak terlihat (pemain baru masuk,
# rate bergeser tiba-tiba), tumbuh bersama horizon.
SHOCK_BASE = 0.45
# Penalti lebar bila history hanya punya satu snapshot berbeda (halaman
# belum pernah di-refresh: tren sebenarnya tidak terlihat sama sekali).
FLAT_PENALTY = 0.70
TWO_SNAPSHOT_PENALTY = 0.35
# Batas kemiringan hasil blend, supaya ekstrapolasi 4,5 hari tidak liar.
SLOPE_MIN = -0.50
SLOPE_MAX = 3.00
# Batas lebar interval (logit) dan lebar minimum agar tidak terlalu yakin.
SPREAD_MIN = 0.15
SPREAD_MAX = 2.50
# Paruh bobot recency dalam jam: pembacaan 24 jam lalu berbobot setengah.
RECENCY_HALF_LIFE_HOURS = 24.0


def parse_moment(text):
    """ISO 8601 (dengan Z) menjadi datetime UTC; None bila gagal."""
    if not isinstance(text, str):
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def clamp_pct(value):
    return min(HI, max(LO, value))


def logit_of(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def inv_normal(p):
    """Kuantil normal baku (aproksimasi rasional Acklam, galat < 2e-9)."""
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return num / den
    if p > p_high:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        return -num / den
    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    return num / den


def round4(value):
    """Bulatkan ke 4 desimal, setengah ke atas; bilangan bulat jadi int."""
    rounded = Decimal(repr(value)).quantize(
        Decimal("0.0001"), rounding=ROUND_HALF_UP)
    as_float = float(rounded)
    return int(as_float) if as_float.is_integer() else as_float


def log_time(hours_since_release):
    """ln(1 + jam sejak rilis); jam sebelum rilis dijepit supaya terdefinisi."""
    return math.log1p(max(hours_since_release, -0.9))


def weighted_fit(points):
    """Regresi linier berbobot (x, y, w) -> (kemiringan, galat-baku, sisa).

    Mengembalikan (None, None, 0.0) bila x tidak bervariasi.
    """
    total_w = sum(w for _, _, w in points)
    if total_w <= 0:
        return None, None, 0.0
    x_bar = sum(w * x for x, _, w in points) / total_w
    y_bar = sum(w * y for _, y, w in points) / total_w
    sxx = sum(w * (x - x_bar) ** 2 for x, _, w in points)
    sxy = sum(w * (x - x_bar) * (y - y_bar) for x, y, w in points)
    if sxx <= 1e-9:
        return None, None, 0.0
    slope = sxy / sxx
    # Sisa berbobot di sekitar garis; dipakai sebagai komponen spread.
    resid_sq = sum(
        w * (y - (y_bar + slope * (x - x_bar))) ** 2 for x, y, w in points
    )
    variance = resid_sq / total_w
    resid_sd = math.sqrt(variance)
    n = len(points)
    if n >= 3:
        # Galat baku kemiringan (pendekatan OLS berbobot).
        se = math.sqrt(variance / sxx)
    else:
        se = PRIOR_SLOPE_SD
    return slope, min(max(se, 0.08), 1.20), min(resid_sd, 1.0)


def forecast_trophy(points, tier, x_target, days):
    """points: [(x, rate_pct, weight)] terurut waktu. -> (median_pct, spread)."""
    if not points:
        median = TIER_FALLBACK_RATE.get(tier, 1.0)
        return median, 2.0

    distinct = len({round(rate, 2) for _, rate, _ in points})
    series = [(x, logit_of(rate), w) for x, rate, w in points]
    slope_ols, se_ols, resid_sd = weighted_fit(series)

    prior = PRIOR_SLOPE.get(tier, PRIOR_SLOPE_FALLBACK)
    # Makin banyak snapshot berbeda, makin besar suara data atas prior.
    w_data = min(max(distinct - 1, 0), 8)
    if slope_ols is None:
        slope = prior
        slope_sd = PRIOR_SLOPE_SD
    else:
        slope = (w_data * slope_ols + PRIOR_WEIGHT * prior) / (w_data + PRIOR_WEIGHT)
        slope_sd = (w_data * se_ols + PRIOR_WEIGHT * PRIOR_SLOPE_SD) / (
            w_data + PRIOR_WEIGHT
        )
    slope = min(max(slope, SLOPE_MIN), SLOPE_MAX)

    # Jangkar garis: bila kemiringan datang dari data, garis melewati titik
    # berat (centroid) data seperti regresi biasa; bila kemiringan datang
    # dari prior (history datar, tren tak terlihat), garis harus melewati
    # pembacaan TERAKHIR -- prior adalah pernyataan tentang pertumbuhan
    # mulai dari sekarang, bukan tentang masa lalu. Tanpa ini, history
    # datar di jam-jam pertama (centroid jauh sebelum checkpoint) akan
    # mengekstrapolasi pertumbuhan seolah-olah sudah berjalan sejak
    # tengah jendela, dan median melesat terlalu tinggi.
    total_w = sum(w for _, _, w in series)
    x_bar = sum(w * x for x, _, w in series) / total_w
    y_bar = sum(w * y for _, y, w in series) / total_w
    x_last, y_last = series[-1][0], series[-1][1]
    anchor_w = w_data + PRIOR_WEIGHT
    anchor_x = (w_data * x_bar + PRIOR_WEIGHT * x_last) / anchor_w
    anchor_y = (w_data * y_bar + PRIOR_WEIGHT * y_last) / anchor_w
    median_logit = anchor_y + slope * (x_target - anchor_x)
    median_logit = min(max(median_logit, LOGIT_LO), LOGIT_HI)

    x_last = series[-1][0]
    dx = max(x_target - x_last, 0.0)
    walk = WALK_SD_PER_SQRT_DAY * math.sqrt(days)
    shock = SHOCK_BASE * math.sqrt(1.0 + days)
    if distinct <= 1:
        flat = FLAT_PENALTY
    elif distinct == 2:
        flat = TWO_SNAPSHOT_PENALTY
    else:
        flat = 0.0
    spread = math.sqrt(
        walk * walk
        + (slope_sd * dx) ** 2
        + shock * shock
        + flat * flat
        + resid_sd * resid_sd
    )
    spread = min(max(spread, SPREAD_MIN), SPREAD_MAX)

    # Trophy yang 100% di setiap pembacaan hampir pasti terkunci di 100%:
    # median tetap di puncak, tapi spread dipangkas supaya tidak
    # menghukum keyakinan itu terlalu mahal.
    last_rate = points[-1][1]
    if last_rate >= HI and all(rate >= HI for _, rate, _ in points):
        spread = min(spread, 0.50)

    return sigmoid(median_logit) * 100.0, spread


def forecast_case(case):
    game = case.get("game") or {}
    release = parse_moment(game.get("release_date"))
    history = case.get("history") or []
    if release is None and history:
        release = parse_moment(history[0].get("slot")) or parse_moment(
            history[0].get("at")
        )
    if release is None:
        release = parse_moment(case.get("at")) or datetime(
            2026, 11, 19, tzinfo=timezone.utc
        )

    target = parse_moment(case.get("target"))
    checkpoint = parse_moment(case.get("at"))
    if target is None and checkpoint is not None:
        from datetime import timedelta

        horizon = case.get("horizon_minutes") or 0
        target = checkpoint + timedelta(minutes=horizon)
    if target is None:
        target = release
    target_hours = (target - release).total_seconds() / 3600.0
    x_target = log_time(target_hours)
    days = max((case.get("horizon_minutes") or 0) / 1440.0, 0.0)

    levels = case.get("levels") or LEVELS_DEFAULT
    levels = [float(t) for t in levels if 0.0 < float(t) < 1.0] or LEVELS_DEFAULT
    z_values = [inv_normal(t) for t in levels]

    # Waktu pembacaan terakhir (untuk bobot recency).
    read_times = []
    for read in history:
        moment = parse_moment(read.get("at")) or parse_moment(read.get("slot"))
        if moment is not None:
            read_times.append(moment)
    last_moment = max(read_times) if read_times else release

    trophies = case.get("trophies") or []
    rows = []
    for index, trophy in enumerate(trophies):
        points = []
        for read in history:
            rates = read.get("rates") or []
            if index >= len(rates):
                continue
            rate = rates[index]
            if rate is None:
                continue
            moment = parse_moment(read.get("at")) or parse_moment(read.get("slot"))
            if moment is None:
                continue
            hours = (moment - release).total_seconds() / 3600.0
            age = (last_moment - moment).total_seconds() / 3600.0
            weight = 0.5 ** (max(age, 0.0) / RECENCY_HALF_LIFE_HOURS)
            points.append((log_time(hours), float(rate), weight))
        points.sort(key=lambda p: p[0])
        median, spread = forecast_trophy(
            points, trophy.get("tier", "bronze"), x_target, days
        )
        median_logit = logit_of(median)
        quantiles = []
        for z in z_values:
            value = sigmoid(median_logit + z * spread) * 100.0
            quantiles.append(round4(clamp_pct(value)))
        # Pembulatan tidak boleh merusak urutan tak-menurun.
        for i in range(1, len(quantiles)):
            if quantiles[i] < quantiles[i - 1]:
                quantiles[i] = quantiles[i - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out_path = OUTPUT_DIR / f"{path.stem}.out"
        out_path.write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
