"""Small deterministic top-head fusion kernels; no files or hidden dataset state."""
import numpy as np

LAMBDAS = (.001, .01, .1, 1.)
ALPHAS = tuple(i / 20 for i in range(21))
G_COLUMNS = ['z_M', 'z_Q', 'rank_M_over20', 'rank_Q_over20', 'topic_cov', 'request_cov',
             'topic_request_jaccard', 'log1p_body_tokens', 'zM_gapM', 'zM_gapQ', 'zQ_gapM', 'zQ_gapQ']
C_COLUMNS = ['exclusion_hit', 'exclusion_cov', 'log1p_exclusion_hits', 'exclusion_density',
             'exclusion_cov_zM', 'exclusion_cov_zQ']


def zscore(values):
    values = np.asarray(values, dtype=np.float64)
    sigma = values.std(ddof=0)
    return (values - values.mean()) / sigma if sigma else np.zeros_like(values)


def doc_features(m, q, documents, topic_tokens, request_tokens, negatives):
    """documents=(token lists), in MiniLM top20 order; q/m include only these20."""
    assert len(m) == len(q) == len(documents) == 20
    zm, zq = zscore(m), zscore(q)
    # q rank tie must be supplied via canonical ordering by caller for equality.
    # Rank positions below are injected separately so text computation stays pure.
    gaps = [np.sort(z)[-1] - np.sort(z)[-2] for z in (zm, zq)]
    t, r, n = set(topic_tokens), set(request_tokens), set(negatives)
    result = []
    for i, toks in enumerate(documents):
        d, length = set(toks), len(toks)
        cov = lambda a: len(a & d) / len(a) if a else 0.
        hits = len(n & d)
        nc = cov(n)
        result.append([zm[i], zq[i], 0., 0., cov(t), cov(r),
            len((t | r) & d) / len(t | r | d) if t | r | d else 0., np.log1p(length),
            zm[i]*gaps[0], zm[i]*gaps[1], zq[i]*gaps[0], zq[i]*gaps[1],
            float(hits > 0), nc, np.log1p(hits), hits/max(length, 1), nc*zm[i], nc*zq[i]])
    result = np.asarray(result, dtype=np.float64)
    assert result.shape == (20, 18) and np.isfinite(result).all()
    return result


def top_positions(scores, canonical):
    """Head winner; canonical is per-query index list sorted by documentID."""
    return np.take_along_axis(canonical, np.argmax(np.take_along_axis(scores, canonical, axis=1),axis=1)[:,None],axis=1)[:,0]


def rank_positions(scores, canonical):
    ordered = np.take_along_axis(canonical, np.argsort(-np.take_along_axis(scores,canonical,axis=1),axis=1,kind='stable'),axis=1)
    return ordered


def correct_count(scores, canonical, positive, weights):
    return int(np.sum(weights * (top_positions(scores,canonical) == positive)))


def ridge_statistics(x, positive, negative, weights, dimensions):
    """One pair/query; weighted duplicates are exact empirical-count weights, no intercept."""
    assert dimensions in (12,18)
    valid = (positive >= 0) & (negative >= 0) & (weights > 0)
    indices = np.flatnonzero(valid)
    count = int(weights[valid].sum())
    if count == 0:
        raise ValueError('no qualified training pair')
    d = x[indices, positive[indices], :dimensions] - x[indices, negative[indices], :dimensions]
    wgt = weights[indices].astype(np.float64)
    scale = np.sqrt(np.sum(wgt[:,None]*d*d,axis=0)/count)
    scale[scale == 0] = 1.
    z = d/scale
    gram = z.T @ (wgt[:,None]*z)
    rhs = z.T @ wgt
    return scale, gram, rhs, count


def solve_statistics(stats, lam):
    scale, gram, rhs, count = stats
    assert lam in LAMBDAS
    coef = np.linalg.solve(gram + count*lam*np.eye(len(scale)),rhs)
    assert np.isfinite(coef).all()
    return scale, coef, count


def ridge_fit(x, positive, negative, weights, lam, dimensions):
    return solve_statistics(ridge_statistics(x,positive,negative,weights,dimensions),lam)


def ridge_scores(x, model):
    scale, coef, _ = model
    return (x[:,:,:len(coef)]/scale) @ coef


def select_alpha(x, positive, canonical, training_weights):
    best_count, best_alpha = -1, None
    for alpha in ALPHAS:
        value = correct_count(alpha*x[:,:,0]+(1-alpha)*x[:,:,1],canonical,positive,training_weights)
        if value > best_count:  # ascending alpha, strict comparison retains smaller alpha.
            best_count, best_alpha = value, alpha
    return best_alpha, best_count


def select_lambda(x, positive, negative, canonical, fold, outer, weights, dimensions):
    """Only training-side labels supplied. P1 validation uses every inner query, not pair subset."""
    validation = [f for f in range(5) if f != outer]
    counts = [0] * len(LAMBDAS)
    for held in validation:
        tw = weights * ((fold != outer) & (fold != held))
        vw = weights * (fold == held)
        stats = ridge_statistics(x,positive,negative,tw,dimensions)
        held_indices = np.flatnonzero(vw > 0)
        for j, lam in enumerate(LAMBDAS):
            model = solve_statistics(stats,lam)
            counts[j] += correct_count(ridge_scores(x[held_indices],model),canonical[held_indices],positive[held_indices],vw[held_indices])
    return LAMBDAS[int(np.argmax(counts))], counts
