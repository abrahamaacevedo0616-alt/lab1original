
from __future__ import annotations
from fractions import Fraction
import numpy as np
from signals.base_signals import Delta, continuous_support, evaluate_continuous_signal, get_signal_by_key

ALLOWED_A = [2, 3, 4, 5, 1 / 2, 1 / 3, 1 / 4, 1 / 5]
ALLOWED_T0 = [1, 2, 3, 4, 5, 6]
ALLOWED_M = [2, 3, 4, 1 / 2, 1 / 3, 1 / 4, 1 / 5]
ALLOWED_N0 = [1, 2, 3, 4, 5, 6]
INTERP_METHODS = ['zero', 'step', 'linear']


def _validate_sign(sign: str) -> None:
    if sign not in ('+', '-'):
        raise ValueError("El signo debe ser '+' o '-'.")


def _sign(sign: str) -> int:
    return 1 if sign == '+' else -1


def _to_list(a):
    return np.array(a, dtype=float).tolist()


# ============================================================
# TRANSFORMACIONES CONTINUAS TIPO NOTEBOOK
# ============================================================

def _continuous_by_argument(signal_key: str, alpha: float, beta: float):
    p = continuous_support(signal_key)
    nuevos_p = []
    for i in range(len(p)):
        nuevos_p.append((p[i] - beta) / alpha)

    if alpha < 0:
        nuevos_p = nuevos_p[::-1]

    t1 = np.arange(nuevos_p[0], nuevos_p[1], Delta)
    t2 = np.arange(nuevos_p[1], nuevos_p[2], Delta)
    t3 = np.arange(nuevos_p[2], nuevos_p[3], Delta)
    t4 = np.arange(nuevos_p[3], nuevos_p[4] + Delta, Delta)

    t = np.concatenate((t1, t2, t3, t4))
    x = evaluate_continuous_signal(signal_key, alpha * t + beta)
    return t, x, nuevos_p


def transform_continuous(signal_key: str, a_mag: float, a_sign: str, t0: int, shift_sign: str, method: str) -> dict:
    _validate_sign(a_sign)
    _validate_sign(shift_sign)

    if a_mag not in ALLOWED_A:
        raise ValueError('Valor de |a| no permitido.')
    if t0 not in ALLOWED_T0:
        raise ValueError('Valor de t0 no permitido.')
    if method not in ('method_1', 'method_2'):
        raise ValueError('Método no permitido.')

    signal = get_signal_by_key(signal_key)
    if signal['domain'] != 'continuous':
        raise ValueError('La señal seleccionada no es continua.')

    a = _sign(a_sign) * float(a_mag)
    b = _sign(shift_sign) * float(t0)

    # Método 1 del notebook: desplazamiento y luego escalamiento
    if method == 'method_1':
        p = continuous_support(signal_key)

        nuevos_p = []
        for i in range(len(p)):
            nuevos_p.append(p[i] - b)

        t1 = np.arange(nuevos_p[0], nuevos_p[1], Delta)
        t2 = np.arange(nuevos_p[1], nuevos_p[2], Delta)
        t3 = np.arange(nuevos_p[2], nuevos_p[3], Delta)
        t4 = np.arange(nuevos_p[3], nuevos_p[4] + Delta, Delta)

        t_step1 = np.concatenate((t1, t2, t3, t4))
        x_step1 = evaluate_continuous_signal(signal_key, t_step1 + b)

        for i in range(len(nuevos_p)):
            nuevos_p[i] = nuevos_p[i] * (1 / a)

        if a < 0:
            nuevos_p = nuevos_p[::-1]

        t1 = np.arange(nuevos_p[0], nuevos_p[1], Delta)
        t2 = np.arange(nuevos_p[1], nuevos_p[2], Delta)
        t3 = np.arange(nuevos_p[2], nuevos_p[3], Delta)
        t4 = np.arange(nuevos_p[3], nuevos_p[4] + Delta, Delta)

        t_final = np.concatenate((t1, t2, t3, t4))
        x_final = evaluate_continuous_signal(signal_key, a * t_final + b)

        steps = [
            {
                'title': 'Paso 1 — Desplazamiento',
                'expression': f'v(t) = x(t {"+" if b >= 0 else "-"} {abs(b):g})',
                'domain': 'continuous',
                't': _to_list(t_step1),
                'x': _to_list(x_step1),
            },
            {
                'title': 'Paso 2 — Escalamiento',
                'expression': f'y(t) = v({a:g}t)',
                'domain': 'continuous',
                't': _to_list(t_final),
                'x': _to_list(x_final),
            },
        ]

    else:
        p = continuous_support(signal_key)

        nuevos_p = []
        for i in range(len(p)):
            nuevos_p.append(p[i] * (1 / a))

        if a < 0:
            nuevos_p = nuevos_p[::-1]

        t1 = np.arange(nuevos_p[0], nuevos_p[1], Delta)
        t2 = np.arange(nuevos_p[1], nuevos_p[2], Delta)
        t3 = np.arange(nuevos_p[2], nuevos_p[3], Delta)
        t4 = np.arange(nuevos_p[3], nuevos_p[4] + Delta, Delta)

        t_step1 = np.concatenate((t1, t2, t3, t4))
        x_step1 = evaluate_continuous_signal(signal_key, a * t_step1)

        for i in range(len(nuevos_p)):
            nuevos_p[i] = nuevos_p[i] - (b / a)

        t1 = np.arange(nuevos_p[0], nuevos_p[1], Delta)
        t2 = np.arange(nuevos_p[1], nuevos_p[2], Delta)
        t3 = np.arange(nuevos_p[2], nuevos_p[3], Delta)
        t4 = np.arange(nuevos_p[3], nuevos_p[4] + Delta, Delta)

        t_final = np.concatenate((t1, t2, t3, t4))
        x_final = evaluate_continuous_signal(signal_key, a * t_final + b)

        steps = [
            {
                'title': 'Paso 1 — Escalamiento',
                'expression': f'v(t) = x({a:g}t)',
                'domain': 'continuous',
                't': _to_list(t_step1),
                'x': _to_list(x_step1),
            },
            {
                'title': 'Paso 2 — Desplazamiento',
                'expression': f'y(t) = v(t {"+" if b / a >= 0 else "-"} {abs(b / a):g})',
                'domain': 'continuous',
                't': _to_list(t_final),
                'x': _to_list(x_final),
            },
        ]

    return {
        'signal_key': signal_key,
        'domain': 'continuous',
        'method': method,
        'final_expression': f'y(t) = x({a:g}t {"+" if b >= 0 else "-"} {abs(b):g})',
        'steps': steps,
        'final_signal': {
            'title': 'Señal transformada',
            'domain': 'continuous',
            't': _to_list(t_final),
            'x': _to_list(x_final),
        },
        'note': 'Transformación continua escrita con puntos de quiebre, tramos y concatenación, como en el notebook de clase.',
    }

def interpolate_discrete(signal_key: str, L: int, mode: str) -> dict:
    signal = get_signal_by_key(signal_key)
    if signal['domain'] != 'discrete':
        raise ValueError('La señal seleccionada no es discreta.')
    if L not in [2, 3, 4, 5]:
        raise ValueError('El factor de interpolación permitido es 2, 3, 4 o 5.')
    if mode not in INTERP_METHODS:
        raise ValueError('Método de interpolación no permitido.')

    n = np.array(signal['n'], dtype=int)
    x_n = np.array(signal['x'], dtype=float)

    nI, x0, xEsc, xLin = _interpolacion_discreta_notebook(x_n, n, L)

    if mode == 'zero':
        x_final = x0
        nombre = 'por cero'
        note = f'Se insertan {L-1} muestras nulas entre muestras consecutivas.'
    elif mode == 'step':
        x_final = xEsc
        nombre = 'por escalón'
        note = f'Se insertan {L-1} muestras repitiendo el último valor.'
    else:
        x_final = xLin
        nombre = 'lineal'
        note = f'Se insertan {L-1} muestras usando rectas entre muestras consecutivas.'

    return {
        'domain': 'discrete',
        'signal_key': signal_key,
        'mode': mode,
        'factor': L,
        'final_expression': f'y[n] = x[n/{L}] ({nombre})',
        'note': note,
        'original_signal': {
            'title': 'Señal original',
            'domain': 'discrete',
            'n': n.tolist(),
            'x': x_n.tolist(),
            'n0': 0,
        },
        'final_signal': {
            'title': f'Interpolación {nombre}',
            'domain': 'discrete',
            'n': nI.tolist(),
            'x': x_final.tolist(),
            'n0': 0,
        },
        'steps': [
            {
                'title': 'Paso 1 — Señal original',
                'expression': 'x[n]',
                'domain': 'discrete',
                'n': n.tolist(),
                'x': x_n.tolist(),
            },
            {
                'title': f'Paso 2 — Interpolación {nombre}',
                'expression': f'y[n] = x[n/{L}]',
                'domain': 'discrete',
                'n': nI.tolist(),
                'x': x_final.tolist(),
            }
        ]
    }

