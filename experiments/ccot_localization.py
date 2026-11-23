"""Band-limited, one-dimensional score localization without filter training."""
import cmath
import hashlib
import json
from math import cos, tau
from pathlib import Path


def coefficient(samples, frequency):
    size = len(samples)
    return sum(value * cmath.exp(-1j * tau * frequency * j / size)
               for j, value in enumerate(samples)) / size


def score(modes, position, derivative=0):
    # The real input supplies conjugate negative modes; the examples have zero DC.
    return 2 * sum((value * (1j * tau * k) ** derivative
                    * cmath.exp(1j * tau * k * position)).real
                   for k, value in modes.items())


def locate(center):
    modes = {}
    for size, k, amplitude in [(5, 1, 1.0), (9, 2, 0.25)]:
        samples = [amplitude * cos(tau * k * (j / size - center)) for j in range(size)]
        modes[k] = coefficient(samples, k)
    grid_peak = max((j / 9 for j in range(9)), key=lambda t: score(modes, t))
    refined = grid_peak
    for _ in range(5):
        refined -= score(modes, refined, 1) / score(modes, refined, 2)
    return grid_peak, refined % 1


def measure():
    records = []
    for center in (0.31, 0.62, 0.98):
        grid, refined = locate(center)
        records.append({"center": center, "grid_peak": grid, "refined_peak": refined,
                        "grid_error": abs((grid - center + 0.5) % 1 - 0.5),
                        "refined_error": abs((refined - center + 0.5) % 1 - 0.5)})
    # cos(4*2πt) and cos(2πt) agree at j/5, but disagree at t=1/2.
    alias = max(abs(cos(tau * 4 * j / 5) - cos(tau * j / 5)) for j in range(5))
    return {"peaks": records, "alias_sample_error": alias,
            "alias_half_period_gap": abs(cos(tau * 4 / 2) - cos(tau / 2))}


if __name__ == "__main__":
    print(json.dumps({"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "results": measure()}, indent=2))
