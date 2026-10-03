"""Bounded diagnostic rewards; backend evidence must be supplied explicitly."""
import math


def score_feasibility(*, model_bytes, reference_bytes, gain=None,
                      backend_status='not_attempted', allow_proxy=False):
    """Environment errors are unscored; unknown conversion is never success.

    ``allow_proxy`` permits provisional local feedback without board conversion,
    and must not be used to select a final deployable model.
    """
    statuses = {'success', 'conversion_failed', 'runtime_failed',
                'environment_error', 'not_attempted'}
    if backend_status not in statuses:
        raise ValueError('Unknown backend evidence status')
    if backend_status == 'environment_error':
        return {'reward': None, 'deployable': False, 'reason': 'environment_error_excluded'}
    if backend_status in {'conversion_failed', 'runtime_failed'}:
        return {'reward': -1., 'deployable': False, 'reason': backend_status}
    if backend_status == 'not_attempted' and not allow_proxy:
        return {'reward': None, 'deployable': False, 'reason': 'backend_unverified'}
    if reference_bytes <= 0 or model_bytes <= 0:
        raise ValueError('Model byte counts must be positive')
    feasible = 5*model_bytes <= 3*reference_bytes
    if not feasible:
        distance = max(0., model_bytes/(0.6*reference_bytes)-1.)
        reward = -distance/(1.+distance)
        reason = 'size_budget_exceeded'
    else:
        if gain is None or not math.isfinite(gain) or gain < 0:
            raise ValueError('Finite nonnegative quality/speed gain required')
        reward = gain/(1.+gain)
        reason = 'feasible' if backend_status == 'success' else 'proxy_feasible_backend_unverified'
    return {'reward': reward,
            'deployable': backend_status == 'success' and feasible,
            'reason': reason}
