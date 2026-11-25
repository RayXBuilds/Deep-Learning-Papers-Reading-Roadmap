"""Gradients of temperature-scaled distillation for a fixed three-class input."""
from hashlib import sha256
import json
from pathlib import Path

from run import softmax


def scaled_gradient(student, teacher, temperature):
    """Derivative of T² CE(p_teacher, q_student) with respect to student logits."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    q = softmax([x / temperature for x in student])
    p = softmax([x / temperature for x in teacher])
    return [temperature * (a - b) for a, b in zip(q, p, strict=True)]


def measure():
    student = [2, 0, 0]
    teacher = [4, 1, -2]
    rows = []
    for temperature in (1, 4, 16, 128):
        gradient = scaled_gradient(student, teacher, temperature)
        limit = [-5 / 9, -2 / 9, 7 / 9]
        error = max(abs(x - y) for x, y in zip(gradient, limit))
        rows.append({"T": temperature, "scaled_gradient": gradient, "max_error_to_limit": error})
    return {"student": student, "teacher": teacher, "sweep": rows,
            "swapped_teacher_T4": scaled_gradient(student, [4, -2, 1], 4)}


if __name__ == "__main__":
    root = Path(__file__).parent
    hashes = {name: sha256((root / name).read_bytes()).hexdigest()
              for name in ("distillation_temperature.py", "run.py")}
    print(json.dumps({"source_sha256": hashes, "results": measure()}, indent=2))
