from __future__ import annotations

import numpy as np


# -----------------------------------------------------------------------------
# Utilidades simples para devolver la información en el formato que usa la app
# -----------------------------------------------------------------------------
def _continuous_dict(key: str, title: str, description: str, t: np.ndarray, x: np.ndarray) -> dict:
    return {
        'key': key,
        'title': title,
        'description': description,
        'domain': 'continuous',
        't': t.tolist(),
        'x': x.tolist(),
    }


def _discrete_dict(key: str, title: str, description: str, n: np.ndarray, x: np.ndarray, n0: int = 0) -> dict:
    return {
        'key': key,
        'title': title,
        'description': description,
        'domain': 'discrete',
        'n': n.tolist(),
        'x': x.tolist(),
        'n0': int(n0),
    }


# -----------------------------------------------------------------------------
# SEÑALES CONTINUAS
# Estructura escrita al estilo de los notebooks del drive:
# puntos de quiebre -> intervalos -> tramos -> concatenación
# -----------------------------------------------------------------------------
def build_continuous_signal_1(delta: float = 0.01) -> dict:
    p1 = -2
    p2 = -1
    p3 = 1
    p4 = 3
    p5 = 4

    t1 = np.arange(p1, p2, delta)
    t2 = np.arange(p2, p3, delta)
    t3 = np.arange(p3, p4, delta)
    t4 = np.arange(p4, p5 + delta, delta)

    x1 = 2 * t1 + 4
    x2 = 2 * np.ones(len(t2))
    x3 = 3 * np.ones(len(t3))
    x4 = -3 * t4 + 12

    t = np.concatenate((t1, t2, t3, t4))
    x_t = np.concatenate((x1, x2, x3, x4))

    return _continuous_dict(
        key='continuous_1',
        title='Señal continua 1',
        description='Recta ascendente, nivel 2, nivel 3 y recta descendente.',
        t=t,
        x=x_t,
    )



def build_continuous_signal_2(delta: float = 0.01) -> dict:
    p1 = -3
    p2 = -2
    p3 = 0
    p4 = 2
    p5 = 3

    t1 = np.arange(p1, p2, delta)
    t2 = np.arange(p2, p3, delta)
    t3 = np.arange(p3, p4, delta)
    t4 = np.arange(p4, p5 + delta, delta)

    x1 = t1 + 3
    x2 = 0.5 * t2 + 3
    x3 = -t3 + 3
    x4 = np.ones(len(t4))

    t = np.concatenate((t1, t2, t3, t4))
    x_t = np.concatenate((x1, x2, x3, x4))

    return _continuous_dict(
        key='continuous_2',
        title='Señal continua 2',
        description='Rampa ascendente, salto en t=-2, pico en t=0, descenso lineal, meseta en 1 y caída a 0 en t=3.',
        t=t,
        x=x_t,
    )


# -----------------------------------------------------------------------------
# SEÑALES DISCRETAS
# Estructura escrita al estilo de los notebooks discretos del drive:
# eje n + vector de muestras + asignación por tramos o por lista explícita
# -----------------------------------------------------------------------------
def build_discrete_signal_1() -> dict:
    x_n = np.array([
        0, 0, 0, 0, 0, -4, 0, 3, 5, 2, -3, -1, 3, 6, 8, 3, -1, 0, 0, 0, 0, 0
    ], dtype=float)

    indice_origen = 5
    n = np.arange(-indice_origen, len(x_n) - indice_origen)

    return _discrete_dict(
        key='discrete_1',
        title='Secuencia discreta 1',
        description='Secuencia finita con origen en la muestra de valor -4.',
        n=n,
        x=x_n,
        n0=0,
    )



def build_discrete_signal_2() -> dict:
    n = np.arange(-10, 11)
    x_n = np.zeros(len(n), dtype=float)

    for i, k in enumerate(n):
        if -10 <= k <= -6:
            x_n[i] = 0
        elif -5 <= k <= 0:
            x_n[i] = (3 / 4) ** k
        elif 1 <= k <= 5:
            x_n[i] = (7 / 4) ** k
        elif 6 <= k <= 10:
            x_n[i] = 0

    return _discrete_dict(
        key='discrete_2',
        title='Secuencia discreta 2',
        description='Secuencia definida por tramos exponenciales y ceros laterales.',
        n=n,
        x=x_n,
        n0=0,
    )



def get_signal_catalog() -> dict:
    return {
        'continuous': [
            build_continuous_signal_1(),
            build_continuous_signal_2(),
        ],
        'discrete': [
            build_discrete_signal_1(),
            build_discrete_signal_2(),
        ]
    }



def get_signal_by_key(key: str) -> dict:
    signals = [
        build_continuous_signal_1(),
        build_continuous_signal_2(),
        build_discrete_signal_1(),
        build_discrete_signal_2(),
    ]

    for signal in signals:
        if signal['key'] == key:
            return signal

    raise KeyError(f'No existe una señal registrada con la llave: {key}')
