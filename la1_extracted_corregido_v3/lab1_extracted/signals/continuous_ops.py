from __future__ import annotations

import numpy as np

from signals.base_signals import get_signal_by_key
from signals.transformations import _build_continuous_from_argument, _safe_delta



def continuous_sum_operation(signal_key: str) -> dict:
    signal = get_signal_by_key(signal_key)
    if signal['domain'] != 'continuous':
        raise ValueError('La señal seleccionada no es continua.')

    t_x = np.array(signal['t'], dtype=float)
    x_t = np.array(signal['x'], dtype=float)

    # Primer término: v1(t) = x(t/3 + 2)
    t_v1, x_v1 = _build_continuous_from_argument(t_x, x_t, alpha=1/3, beta=2)

    # Segundo término: v2(t) = x(1 - t/4) = x(-t/4 + 1)
    t_v2, x_v2 = _build_continuous_from_argument(t_x, x_t, alpha=-1/4, beta=1)

    # En clase la suma final se hace llevando ambas señales al mismo eje temporal.
    delta = min(_safe_delta(t_v1, 0.01), _safe_delta(t_v2, 0.01))
    p1 = min(np.min(t_v1), np.min(t_v2))
    p2 = max(np.max(t_v1), np.max(t_v2))
    t = np.arange(p1, p2 + delta, delta)

    v1_t = np.interp(t, t_v1, x_v1, left=0.0, right=0.0)
    v2_t = np.interp(t, t_v2, x_v2, left=0.0, right=0.0)
    y_t = v1_t + v2_t

    return {
        'signal_key': signal_key,
        'domain': 'continuous',
        'final_expression': 'y(t) = x(t/3 + 2) + x(1 - t/4)',
        'note': 'Cada término se construyó por tramos y después ambos se sumaron sobre un mismo eje temporal.',
        'steps': [
            {
                'title': 'Paso 1 — Primer término',
                'expression': 'v1(t) = x(t/3 + 2)',
                'domain': 'continuous',
                't': t_v1.tolist(),
                'x': x_v1.tolist(),
            },
            {
                'title': 'Paso 2 — Segundo término',
                'expression': 'v2(t) = x(1 - t/4)',
                'domain': 'continuous',
                't': t_v2.tolist(),
                'x': x_v2.tolist(),
            },
            {
                'title': 'Paso 3 — Suma punto a punto',
                'expression': 'y(t) = v1(t) + v2(t)',
                'domain': 'continuous',
                't': t.tolist(),
                'x': y_t.tolist(),
            },
        ],
        'final_signal': {
            'title': 'Suma final de términos transformados',
            'domain': 'continuous',
            't': t.tolist(),
            'x': y_t.tolist(),
        }
    }
