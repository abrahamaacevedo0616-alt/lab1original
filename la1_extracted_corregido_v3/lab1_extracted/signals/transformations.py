from __future__ import annotations

from fractions import Fraction

import numpy as np

from signals.base_signals import get_signal_by_key


ALLOWED_A = [2, 3, 4, 5, 1/2, 1/3, 1/4, 1/5]
ALLOWED_T0 = [1, 2, 3, 4, 5, 6]
ALLOWED_M = [2, 3, 4, 1/2, 1/3, 1/4, 1/5]
ALLOWED_N0 = [1, 2, 3, 4, 5, 6]
INTERP_METHODS = ['zero', 'step', 'linear']


def _validate_sign(sign: str) -> None:
    if sign not in ('+', '-'):
        raise ValueError("El signo debe ser '+' o '-'.")



def _sign(sign: str) -> int:
    if sign == '+':
        return 1
    return -1



def _split_continuous_segments(t_x: np.ndarray, x_t: np.ndarray, tol: float = 1e-6):
    """
    Separa la señal continua en tramos con pendiente constante,
    para dejar la transformación escrita como en los notebooks:
    puntos de quiebre -> intervalos -> ecuaciones de cada tramo.
    """
    if len(t_x) < 3:
        return [(t_x.copy(), x_t.copy())]

    pendientes = np.diff(x_t) / np.diff(t_x)
    cortes = [0]

    for i in range(1, len(pendientes)):
        if abs(pendientes[i] - pendientes[i - 1]) > tol:
            cortes.append(i)

    cortes.append(len(t_x) - 1)

    segmentos = []
    for i in range(len(cortes) - 1):
        ini = cortes[i]
        fin = cortes[i + 1] + 1
        segmentos.append((t_x[ini:fin].copy(), x_t[ini:fin].copy()))

    return segmentos





def _safe_delta(t: np.ndarray, default: float = 0.01) -> float:
    if len(t) < 2:
        return default
    dif = np.diff(np.sort(np.unique(np.round(t, 10))))
    dif = dif[dif > 1e-9]
    if len(dif) == 0:
        return default
    return float(np.min(dif))

def _extract_continuous_model(t_x: np.ndarray, x_t: np.ndarray):
    segmentos = _split_continuous_segments(t_x, x_t)
    modelo = []

    for tramo_t, tramo_x in segmentos:
        m = (tramo_x[-1] - tramo_x[0]) / (tramo_t[-1] - tramo_t[0]) if abs(tramo_t[-1] - tramo_t[0]) > 1e-12 else 0.0
        b = tramo_x[0] - m * tramo_t[0]
        modelo.append({
            'p_ini': float(tramo_t[0]),
            'p_fin': float(tramo_t[-1]),
            'm': float(m),
            'b': float(b),
        })

    delta_x = _safe_delta(t_x, 0.01)
    return modelo, delta_x



def _build_continuous_from_argument(t_x: np.ndarray, x_t: np.ndarray, alpha: float, beta: float):
    """
    Construye y(t)=x(alpha*t+beta) con una forma más parecida a clase:
    1) identificar puntos de quiebre de x(t)
    2) transformar cada intervalo del eje temporal
    3) escribir cada tramo y concatenar
    """
    modelo, delta_x = _extract_continuous_model(t_x, x_t)
    delta_t = delta_x / abs(alpha)

    partes_t = []
    partes_x = []

    for i, tramo in enumerate(modelo):
        p1 = tramo['p_ini']
        p2 = tramo['p_fin']
        m = tramo['m']
        b = tramo['b']

        q1 = (p1 - beta) / alpha
        q2 = (p2 - beta) / alpha

        ti = np.arange(min(q1, q2), max(q1, q2), delta_t)
        if i == len(modelo) - 1:
            ti = np.arange(min(q1, q2), max(q1, q2) + delta_t, delta_t)

        xi = m * (alpha * ti + beta) + b

        partes_t.append(ti)
        partes_x.append(xi)

    t = np.concatenate(partes_t)
    x = np.concatenate(partes_x)

    orden = np.argsort(t)
    return t[orden], x[orden]



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

    t_x = np.array(signal['t'], dtype=float)
    x_t = np.array(signal['x'], dtype=float)

    a = _sign(a_sign) * float(a_mag)
    b = _sign(shift_sign) * float(t0)

    if method == 'method_1':
        t_step1, x_step1 = _build_continuous_from_argument(t_x, x_t, alpha=1.0, beta=b)
        t_step2, x_step2 = _build_continuous_from_argument(t_x, x_t, alpha=a, beta=b)

        steps = [
            {
                'title': 'Paso 1 — Desplazamiento',
                'expression': f'v(t) = x(t {"+" if b >= 0 else "-"} {abs(b):g})',
                'domain': 'continuous',
                't': t_step1.tolist(),
                'x': x_step1.tolist(),
            },
            {
                'title': 'Paso 2 — Escalamiento',
                'expression': f'y(t) = v({a:g}t)',
                'domain': 'continuous',
                't': t_step2.tolist(),
                'x': x_step2.tolist(),
            }
        ]
    else:
        t_step1, x_step1 = _build_continuous_from_argument(t_x, x_t, alpha=a, beta=0.0)
        t_step2, x_step2 = _build_continuous_from_argument(t_x, x_t, alpha=a, beta=b)

        steps = [
            {
                'title': 'Paso 1 — Escalamiento',
                'expression': f'v(t) = x({a:g}t)',
                'domain': 'continuous',
                't': t_step1.tolist(),
                'x': x_step1.tolist(),
            },
            {
                'title': 'Paso 2 — Desplazamiento',
                'expression': f'y(t) = v(t {"+" if b/a >= 0 else "-"} {abs(b/a):g})',
                'domain': 'continuous',
                't': t_step2.tolist(),
                'x': x_step2.tolist(),
            }
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
            't': t_step2.tolist(),
            'x': x_step2.tolist(),
        },
        'note': 'La transformación continua se escribió por intervalos y tramos, siguiendo la lógica de puntos de quiebre y concatenación.',
    }



def _discrete_mapping(n: np.ndarray, x: np.ndarray) -> dict:
    return {int(k): float(v) for k, v in zip(n, x)}



def _build_interpolation(n_original: np.ndarray, x_original: np.ndarray, L: int, mode: str):
    n_new = np.arange(n_original[0] * L, n_original[-1] * L + 1)
    x_new = np.zeros(len(n_new), dtype=float)

    index_map = {int(k * L): float(v) for k, v in zip(n_original, x_original)}

    for i, k in enumerate(n_new):
        if k in index_map:
            x_new[i] = index_map[k]
        else:
            left_idx = int(np.floor((k - n_original[0] * L) / L))
            left_idx = max(0, min(left_idx, len(n_original) - 2))

            x_left = x_original[left_idx]
            x_right = x_original[left_idx + 1]
            r = (k - n_original[left_idx] * L) / L

            if mode == 'zero':
                x_new[i] = 0.0
            elif mode == 'step':
                x_new[i] = x_left
            else:
                x_new[i] = x_left + (x_right - x_left) * r

    return n_new, x_new



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

    nI, x_nI = _build_interpolation(n, x_n, L, mode)

    nombres = {
        'zero': 'por cero',
        'step': 'por escalón',
        'linear': 'lineal',
    }

    notas = {
        'zero': f'Se insertan {L-1} muestras nulas entre muestras consecutivas.',
        'step': f'Se insertan {L-1} muestras conservando el último valor conocido.',
        'linear': f'Se insertan {L-1} muestras calculadas por recta entre muestras consecutivas.',
    }

    return {
        'domain': 'discrete',
        'signal_key': signal_key,
        'mode': mode,
        'factor': L,
        'final_expression': f'y[n] = x[n/{L}] ({nombres[mode]})',
        'note': notas[mode],
        'original_signal': {
            'title': 'Señal original',
            'domain': 'discrete',
            'n': n.tolist(),
            'x': x_n.tolist(),
            'n0': 0,
        },
        'final_signal': {
            'title': f'Interpolación {nombres[mode]}',
            'domain': 'discrete',
            'n': nI.tolist(),
            'x': x_nI.tolist(),
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
                'title': f'Paso 2 — Interpolación {nombres[mode]}',
                'expression': f'y[n] = x[n/{L}]',
                'domain': 'discrete',
                'n': nI.tolist(),
                'x': x_nI.tolist(),
            }
        ]
    }



def _transform_discrete_direct(n_original: np.ndarray, x_original: np.ndarray, M: float, b: int, interp_mode: str = 'zero'):
    mapping = _discrete_mapping(n_original, x_original)

    if abs(M) >= 1:
        M_int = int(M)
        n_candidates = np.arange(-80, 81)
        n_valid = []
        x_valid = []

        for n in n_candidates:
            arg = M_int * n + b
            if arg in mapping:
                n_valid.append(int(n))
                x_valid.append(mapping[arg])

        if len(n_valid) == 0:
            return np.array([0]), np.array([0.0]), 'No hubo muestras coincidentes dentro del soporte original.'

        return (
            np.array(n_valid, dtype=int),
            np.array(x_valid, dtype=float),
            'Transformación discreta usando coincidencia exacta de índices (diezmado/compresión discreta).'
        )

    frac = Fraction(abs(M)).limit_denominator()
    L = frac.denominator
    nI, x_nI = _build_interpolation(n_original, x_original, L, interp_mode)

    if M < 0:
        nI = -nI[::-1]
        x_nI = x_nI[::-1]

    n_shift = nI - b
    order = np.argsort(n_shift)

    modo = {'zero': 'cero', 'step': 'escalón', 'linear': 'lineal'}[interp_mode]
    return n_shift[order], x_nI[order], f'Interpolación {modo} con factor L={L} para |M|<1.'



def transform_discrete(signal_key: str, m_mag: float, m_sign: str, n0: int, shift_sign: str, method: str, interp_mode: str = 'zero') -> dict:
    _validate_sign(m_sign)
    _validate_sign(shift_sign)

    if m_mag not in ALLOWED_M:
        raise ValueError('Valor de |M| no permitido.')
    if n0 not in ALLOWED_N0:
        raise ValueError('Valor de n0 no permitido.')
    if method not in ('method_1', 'method_2'):
        raise ValueError('Método no permitido.')

    signal = get_signal_by_key(signal_key)
    if signal['domain'] != 'discrete':
        raise ValueError('La señal seleccionada no es discreta.')

    n = np.array(signal['n'], dtype=int)
    x_n = np.array(signal['x'], dtype=float)

    M = _sign(m_sign) * float(m_mag)
    b = _sign(shift_sign) * int(n0)

    n_final, x_final, note = _transform_discrete_direct(n, x_n, M, b, interp_mode=interp_mode)
    n_shift, x_shift, _ = _transform_discrete_direct(n, x_n, 1.0, b, interp_mode=interp_mode)

    steps = [
        {
            'title': 'Paso 1 — Desplazamiento discreto',
            'expression': f'v[n] = x[n {"+" if b >= 0 else "-"} {abs(b)}]',
            'domain': 'discrete',
            'n': n_shift.tolist(),
            'x': x_shift.tolist(),
        },
        {
            'title': 'Paso 2 — Escalamiento discreto',
            'expression': f'y[n] = v[{M:g}n]',
            'domain': 'discrete',
            'n': n_final.tolist(),
            'x': x_final.tolist(),
        }
    ]

    return {
        'signal_key': signal_key,
        'domain': 'discrete',
        'method': method,
        'final_expression': f'y[n] = x[{M:g}n {"+" if b >= 0 else "-"} {abs(b)}]',
        'final_signal': {
            'title': 'Secuencia transformada',
            'domain': 'discrete',
            'n': n_final.tolist(),
            'x': x_final.tolist(),
            'n0': 0,
        },
        'steps': steps,
        'note': note,
    }
