from src.data.synthetic import generate_problems, generate_verified_cot


def test_reference_gravity_generator_and_solver_roundtrip():
    rows = list(generate_problems({"physics_gravity": 3}, seed=42))
    assert len(rows) == 3
    assert all(row["id"].startswith("syn-physics_gravity-") for row in rows)
    solved, report = generate_verified_cot(rows)
    assert len(solved) == 3
    assert report["states"].get("correct") == 3
    assert all(row["cot_correct"] and row["cot"] for row in solved)


def test_problem_generation_is_deterministic():
    first = list(generate_problems({"physics_gravity": 3}, seed=42))
    second = list(generate_problems({"physics_gravity": 3}, seed=42))
    assert first == second
    assert len({row["id"] for row in first}) == len(first)
