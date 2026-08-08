"""Pure-NumPy transmittance-boundary calibration.

The module has no Jittor dependency, so callers can validate the boundary
contract without initializing CUDA:
  1. output length == n exactly
  2. strictly increasing float32 — finite, positive, adjacent log-depth interval
     > 0 (checked AFTER the float32 conversion; a collapsed pair is re-spread
     with `nextafter` / log-space redistribution)
  3. calibration input is the QUERY distance q = max(dist - shadow_bias, znear)
     (the raw receiver distance is no longer used)
  4. low-end coverage: the first boundary is <= min(q), so NO actual receiver
     falls into the unconditional T=1 interval (`q < bounds[0]`)
  5. quantiles computed on the FULL q array (frequency weighted); np.unique is
     used only for collision detection / gap filling, never to compute quantiles
  6. tail keeps p99 / p99.5 / p99.9 / max(q)*1.02
  7. n < 4 / invalid bias / invalid znear / all-non-finite input -> explicit
     error or documented fallback (never a silent un-queryable grid)
"""
import numpy as np

_DEFAULT_BIAS = 0.02
_DEFAULT_ZNEAR = 0.01


def apply_query_distance(receiver_dists, shadow_bias=_DEFAULT_BIAS,
                         znear=_DEFAULT_ZNEAR):
    """q = max(dist - shadow_bias, znear). Validates the query parameters."""
    if not np.isfinite(shadow_bias) or shadow_bias < 0.0:
        raise ValueError(f"shadow_bias must be finite >= 0, got {shadow_bias}")
    if not np.isfinite(znear) or znear <= 0.0:
        raise ValueError(f"znear must be finite > 0, got {znear}")
    d = np.asarray(receiver_dists, dtype=np.float64).reshape(-1)
    d = d[np.isfinite(d)]
    return np.maximum(d - float(shadow_bias), float(znear))


def _strict_fill(cand, n):
    """Produce exactly `n` strictly-increasing positive boundaries that include
    every `cand` value, by repeatedly splitting the largest log-space gap
    (deterministic; guarantees adjacent log-depth intervals > 0)."""
    cand = np.sort(np.unique(np.asarray(cand, dtype=np.float64)))
    cand = cand[np.isfinite(cand) & (cand > 0)]
    if len(cand) == 0:
        return np.geomspace(_DEFAULT_ZNEAR, 10.0, n)
    if len(cand) == 1:
        v = float(cand[0])
        cand = np.array([v * 0.9, v * 1.1])
    while len(cand) < n:
        log_c = np.log(cand)
        gaps = np.diff(log_c)
        gi = int(np.argmax(gaps))
        mid = float(np.exp((log_c[gi] + log_c[gi + 1]) * 0.5))
        cand = np.sort(np.concatenate([cand, [mid]]))
    if len(cand) > n:
        idx = np.unique(np.round(np.linspace(0, len(cand) - 1, n)).astype(int))
        cand = np.sort(cand[idx])
        while len(cand) < n:                 # rounding may collapse a pair
            log_c = np.log(cand)
            gaps = np.diff(log_c)
            gi = int(np.argmax(gaps))
            mid = float(np.exp((log_c[gi] + log_c[gi + 1]) * 0.5))
            cand = np.sort(np.concatenate([cand, [mid]]))
    return cand[:n]


def _to_strict_float32(b):
    """Convert to float32 and re-assert the exact-N / finite / positive /
    strictly-increasing contract; re-spread a collapsed adjacent pair with the
    float32 `nextafter` step when the float64->float32 rounding folds it."""
    b = np.asarray(b, dtype=np.float32)
    if not (np.all(np.isfinite(b)) and np.all(b > 0)):
        raise ValueError("bounds became non-finite or non-positive after float32")
    for i in range(1, len(b)):
        if b[i] <= b[i - 1]:
            # float32 collapsed (or rounding nudged equal): spread by nextafter
            b[i] = np.nextafter(np.float32(b[i - 1]), np.float32(np.inf))
    if not np.all(np.diff(b) > 0):
        raise ValueError("bounds not strictly increasing after float32 fixup")
    return b


def calibrate_shadow_bounds(receiver_dists, n=16, return_meta=False,
                            shadow_bias=_DEFAULT_BIAS, znear=_DEFAULT_ZNEAR,
                            is_query_distance=False):
    """N3-A / N3-D (§25.5.1) receiver-distance-aware transmittance boundaries.

    Body: p1 + [p5..p95] quantiles over the full QUERY distances q, then an
    explicit tail p99 / p99.5 / p99.9 / max(q)*1.02. The FIRST boundary is
    min(q) (low-end guard) so no actual receiver falls into `q < bounds[0]`
    (which the P4.1 query maps to unconditional T=1).

    N3-D `is_query_distance=True`: the input is ALREADY q = max(dist-bias, znear)
    (e.g. from `collect_receiver_dists(..., as_q=True)`), so the q transform is
    NOT re-applied (applying it twice would double-shift the low end).

    Returns:
      bounds [n] float32  (backward-compatible when return_meta=False)
      (bounds, meta)      when return_meta=True
    """
    if n < 4:
        raise ValueError(f"n must be >= 4, got {n}")
    if is_query_distance:
        q = np.asarray(receiver_dists, dtype=np.float64).reshape(-1)
        q = q[np.isfinite(q)]
    else:
        q = apply_query_distance(receiver_dists, shadow_bias, znear)
    meta = {'fill_strategy': 'none', 'n_effective': 0, 'dmin': None, 'dmax': None,
            'unique_count': 0, 'q_below_first': 0, 'n': int(n),
            'shadow_bias': float(shadow_bias), 'znear': float(znear)}
    if len(q) == 0:
        b = np.geomspace(znear, 10.0, n)
        meta.update(fill_strategy='empty-default-geomspace', n_effective=n,
                    dmin=float(znear), dmax=10.0, unique_count=0)
        out = b.astype(np.float32)
        return (out, meta) if return_meta else out
    u = np.unique(q)
    dmin, dmax = float(q.min()), float(q.max())
    meta.update(dmin=dmin, dmax=dmax, unique_count=int(len(u)))
    if len(u) == 1:
        v = u[0]
        b = np.geomspace(max(v * 0.8, v * 1e-6), max(v * 1.25, v * 1.01), n)
        meta.update(fill_strategy='single-unique-expand', n_effective=n)
        out = _to_strict_float32(b)
        return (out, meta) if return_meta else out
    n_body = max(n - 6, 2)                      # low + p1 + body + 3 tail + max
    qs = np.linspace(5.0, 95.0, n_body)
    cand = np.concatenate([
        [float(q.min())],                       # low-end guard: first <= min q
        np.percentile(q, [1.0]),
        np.percentile(q, qs),
        np.percentile(q, [99.0, 99.5, 99.9]),
        [dmax * 1.02],
    ])
    cand = np.sort(np.unique(cand))
    cand = cand[np.isfinite(cand) & (cand > 0)]
    if cand[-1] < dmax * 1.02:
        cand[-1] = dmax * 1.02
    cand[0] = min(cand[0], float(q.min()))       # never above min q
    b = _strict_fill(cand, n)
    b32 = _to_strict_float32(b)
    # low-end coverage in the QUERY's dtype: the first float32 boundary must be
    # <= min(q) even when float64(min_q) rounds UP in float32 (0.98 -> 0.98000002
    # would wrongly put the min receiver into the unconditional T=1 interval).
    if float(b32[0]) > float(q.min()):
        b32[0] = np.nextafter(np.float32(float(q.min())), np.float32(-np.inf))
    meta.update(fill_strategy=('log-gap-fill' if len(u) < n else 'candidate-only'),
                n_effective=int(len(b32)),
                q_below_first=int((q < float(b32[0])).sum()))
    out = b32
    return (out, meta) if return_meta else out


def marker_meta(bounds, tail_k=4):
    """Explicit marker semantics + indices (guide §25.5.2: stop inferring the
    tail markers from array position). The first boundary is the low guard
    (<= min q); the last `tail_k` are the high tail (p99/p99.5/p99.9/max*1.02
    when produced by calibrate_shadow_bounds, or preserved markers under
    adaptive refinement)."""
    n = len(bounds)
    sem = ['p99', 'p99.5', 'p99.9', 'max*1.02']
    return {'first': 0, 'first_semantic': 'low-guard',
            'tail': list(range(n - tail_k, n)),
            'tail_semantics': sem[:tail_k], 'tail_k': int(tail_k)}


def derive_nested(bounds_master, n_derived, tail_k=4):
    """N3-B (§21) / N3-D (§25.5.2): derive an exact-n strictly-increasing subset
    of a master boundary list. Keeps the first/last boundaries and the explicit
    tail markers (via marker_meta); the remaining interior points are picked
    uniformly in log-depth.

    Returns (subset, indices_into_master, marker_meta)."""
    b = np.asarray(bounds_master, dtype=np.float64)
    mk = marker_meta(b, tail_k)
    if n_derived >= len(b):
        return _to_strict_float32(b), np.arange(len(b), dtype=np.int64), mk
    keep = [mk['first']] + mk['tail']
    interior = [i for i in range(len(b)) if i not in keep]
    need = max(n_derived - len(keep), 0)
    picks = (np.unique(np.round(np.linspace(0, len(interior) - 1, need)).astype(int))
             if need else np.asarray([], dtype=int))
    idx = sorted(set(keep) | {int(interior[i]) for i in picks})
    idx = np.asarray(idx, dtype=np.int64)
    sub = b[idx]
    # enforce strict float32
    return _to_strict_float32(sub), idx, mk


def compute_interval_error(q, w, T_lo, T_hi, bounds):
    """Per-interval weighted transmission uncertainty (guide §25.5.2):
    e_i = sum over receivers in [bounds[i], bounds[i+1]) of w * (T_lo - T_hi).

    `q` are the receiver query distances, `w` the per-receiver weight
    (unoccluded direct-light luminance, else atten*max(NoL,0)), `T_lo`/`T_hi`
    the per-receiver transmission bounds, `bounds` the current grid.

    Returns (err [n-1], counts [n-1], bin_idx [len(q)])."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    lo = np.asarray(T_lo, dtype=np.float64).reshape(-1)
    hi = np.asarray(T_hi, dtype=np.float64).reshape(-1)
    b = np.asarray(bounds, dtype=np.float64).reshape(-1)
    n = len(b)
    bin_idx = np.clip(np.searchsorted(b, q, side='right') - 1, 0, n - 2)
    gap = lo - hi
    err = np.zeros(n - 1)
    counts = np.zeros(n - 1, dtype=np.int64)
    for i in range(n - 1):
        m = bin_idx == i
        counts[i] = int(m.sum())
        if counts[i]:
            err[i] = float(np.sum(w[m] * gap[m]))
    return err, counts, bin_idx


def weighted_q_quantiles_in_interval(q, w, lo_b, hi_b, n_split):
    """Place `n_split` new boundaries inside (lo_b, hi_b) at WEIGHTED q
    quantiles (not arithmetic midpoints — guide §25.5.2). Falls back to
    log-space midpoints when fewer than 2 receivers lie in the interval."""
    m = (q >= lo_b) & (q < hi_b)
    qq, ww = q[m], w[m]
    if n_split < 1:
        return np.asarray([], dtype=np.float64)
    if len(qq) < 2:
        log_lo, log_hi = np.log(max(lo_b, 1e-12)), np.log(max(hi_b, 1e-12))
        return np.exp(np.linspace(log_lo, log_hi, n_split + 2)[1:-1])
    order = np.argsort(qq)
    qs, ws = qq[order], ww[order]
    ws = ws / max(float(ws.sum()), 1e-30)
    cdf = np.cumsum(ws)
    fracs = np.linspace(1.0 / (n_split + 1), 1.0 - 1.0 / (n_split + 1), n_split)
    new = np.interp(fracs, cdf, qs)
    # keep strictly inside the interval (avoid numeric drift onto bounds)
    new = np.clip(new, lo_b * 1.000001, hi_b * 0.999999)
    return new


def adaptive_refine(q, w, T_lo, T_hi, bounds_current, n_target, tail_k=4):
    """Error-driven nested refinement (N3-D §25.5.2). Allocates the added
    boundary budget to intervals by their weighted transmission uncertainty
    e_i = sum(w*(T_lo-T_hi)); inside a chosen interval the new boundaries sit at
    WEIGHTED q quantiles. The first boundary (low guard) and the last `tail_k`
    (high tail) are always preserved. Guarantees bounds_current is a strict
    subset of the result.

    Returns (bounds_new float32, meta) with parent hash / insertion indices /
    per-interval error budget / marker meta."""
    import hashlib
    b_cur = np.asarray(bounds_current, dtype=np.float64).reshape(-1)
    if len(b_cur) >= n_target:
        return _to_strict_float32(b_cur), {
            'note': f'n_target {n_target} <= current {len(b_cur)}',
            'bounds_hash': hashlib.md5(_to_strict_float32(b_cur).tobytes()).hexdigest()[:12]}
    mk = marker_meta(b_cur, tail_k)
    err, counts, _ = compute_interval_error(q, w, T_lo, T_hi, b_cur)
    n_add = n_target - len(b_cur)
    err_safe = np.maximum(err, 0.0)
    if float(err_safe.sum()) <= 0:
        err_safe = (counts.astype(np.float64) + 1.0)     # fallback: receiver count
    # allocate the added budget to intervals by error, then give the REMAINDER
    # (floor loss) to the highest fractional parts so the sum hits n_add exactly.
    alloc_f = err_safe / float(err_safe.sum()) * n_add
    floors = np.floor(alloc_f).astype(int)
    rem = n_add - int(floors.sum())
    for i in np.argsort(-(alloc_f - floors))[:rem]:
        floors[i] += 1
    new_pts = []
    insert_idx = []
    for i in range(len(b_cur) - 1):
        k = int(floors[i])
        if k < 1:
            continue
        pts = weighted_q_quantiles_in_interval(q, w, b_cur[i], b_cur[i + 1], k)
        for p in pts:
            new_pts.append(float(p))
            insert_idx.append(i + 1)                    # boundary inserted after idx i
    merged = sorted(set(b_cur.tolist() + new_pts))
    # top-up: if weighted-quantile collisions left us short of n_target, fill the
    # largest log-space gaps deterministically.
    while len(merged) < n_target:
        mm = np.asarray(merged, dtype=np.float64)
        log_m = np.log(mm)
        gi = int(np.argmax(np.diff(log_m)))
        mid = float(np.exp((log_m[gi] + log_m[gi + 1]) * 0.5))
        merged = sorted(set(merged + [mid]))
    if len(merged) > n_target:
        # overshoot (e.g. log-fallback midpoints collided): thin to exact n_target,
        # always re-adding the current grid + markers.
        keep_set = set(b_cur.tolist())
        others = [x for x in merged if x not in keep_set]
        keep_set.add(float(b_cur[mk['first']]))          # low guard always kept
        keep_set.update({float(b_cur[i]) for i in mk['tail']})  # high tail always kept
        n_keep = len(keep_set)
        pick_n = max(n_target - n_keep, 0)
        idx2 = (np.unique(np.round(np.linspace(0, len(others) - 1, pick_n)).astype(int))
                if pick_n and others else np.asarray([], dtype=int))
        merged = sorted(keep_set | {others[i] for i in idx2})
    merged = np.asarray(merged, dtype=np.float64)
    b32 = _to_strict_float32(merged)
    # low-end guard: first boundary must stay <= min(q) (never drift above it)
    if len(q) and float(b32[0]) > float(np.min(q)):
        b32[0] = np.nextafter(np.float32(float(np.min(q))), np.float32(-np.inf))
    subset_ok = bool(set(b_cur.tolist()) <= set(b32.tolist()))
    meta = {'parent_hash': hashlib.md5(b_cur.tobytes()).hexdigest()[:12],
            'n_parent': int(len(b_cur)), 'n_target': int(n_target),
            'tail_k': int(tail_k),
            'insert_indices': insert_idx,
            'per_interval_error': np.round(err, 6).tolist(),
            'per_interval_receivers': counts.tolist(),
            'markers': mk,
            'subset_of_parent': subset_ok,
            'bounds_hash': hashlib.md5(b32.tobytes()).hexdigest()[:12]}
    return b32, meta
