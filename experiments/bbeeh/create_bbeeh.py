from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import random
from datasets import Dataset
from pathlib import Path


# ============================
# Basic data structures
# ============================


@dataclass(frozen=True)
class BoolExpr:
    text: str
    value: bool


class ExprGenerator:
    """Interface for anything that can generate a BoolExpr."""

    def generate(self, rng: random.Random) -> BoolExpr:  # pragma: no cover - interface only
        raise NotImplementedError


# ============================
# Leaf generators
# ============================


@dataclass(frozen=True)
class ThresholdGen(ExprGenerator):
    """Generate an expression like: a op b cmp t

    We keep ops integer-safe: +, -, * for robustness and readability.
    """

    min_value: int = 1
    max_value: int = 100

    def generate(self, rng: random.Random) -> BoolExpr:
        a = rng.randint(self.min_value, self.max_value)
        b = rng.randint(self.min_value, self.max_value)
        op = rng.choice(["+", "-", "*"])  # avoid division for robustness

        if op == "+":
            result = a + b
        elif op == "-":
            result = a - b
        else:  # "*"
            result = a * b

        # Choose a threshold near the result
        spread = max(1, abs(result) // 2)
        lo, hi = result - spread, result + spread
        threshold = rng.randint(lo, hi)

        cmp_op = rng.choice([">", "<"])  # keep simple, readable
        expr = f"{a} {op} {b} {cmp_op} {threshold}"
        if cmp_op == ">":
            val = result > threshold
        else:
            val = result < threshold
        return BoolExpr(expr, val)


@dataclass(frozen=True)
class ComparisonGen(ExprGenerator):
    """Generate (a op b) cmp (c op d) with integer ops (+,-,*)."""

    min_value: int = 1
    max_value: int = 50

    def _apply(self, x: int, op: str, y: int) -> int:
        if op == "+":
            return x + y
        if op == "-":
            return x - y
        return x * y

    def generate(self, rng: random.Random) -> BoolExpr:
        a, b, c, d = (
            rng.randint(self.min_value, self.max_value),
            rng.randint(self.min_value, self.max_value),
            rng.randint(self.min_value, self.max_value),
            rng.randint(self.min_value, self.max_value),
        )
        op_l = rng.choice(["+", "-", "*"])
        op_r = rng.choice(["+", "-", "*"])
        left_val = self._apply(a, op_l, b)
        right_val = self._apply(c, op_r, d)

        cmp_op = rng.choice(["==", "!=", ">", "<", ">=", "<="])
        left_expr = f"{a} {op_l} {b}"
        right_expr = f"{c} {op_r} {d}"
        expr = f"({left_expr}) {cmp_op} ({right_expr})"

        if cmp_op == "==":
            val = left_val == right_val
        elif cmp_op == "!=":
            val = left_val != right_val
        elif cmp_op == ">":
            val = left_val > right_val
        elif cmp_op == "<":
            val = left_val < right_val
        elif cmp_op == ">=":
            val = left_val >= right_val
        else:
            val = left_val <= right_val

        return BoolExpr(expr, val)


@dataclass(frozen=True)
class ModuloGen(ExprGenerator):
    """Generate num % divisor == remainder."""

    min_value: int = 1
    max_value: int = 100
    max_divisor: int = 10

    def generate(self, rng: random.Random) -> BoolExpr:
        num = rng.randint(self.min_value, self.max_value)
        divisor = rng.randint(2, max(2, self.max_divisor))
        remainder = rng.randint(0, divisor - 1)
        expr = f"{num} % {divisor} == {remainder}"
        return BoolExpr(expr, (num % divisor) == remainder)


@dataclass(frozen=True)
class PrimeGen(ExprGenerator):
    """Generate is_prime(n)."""

    min_value: int = 2
    max_value: int = 100

    def _is_prime(self, n: int) -> bool:
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        i = 3
        while i * i <= n:
            if n % i == 0:
                return False
            i += 2
        return True

    def generate(self, rng: random.Random) -> BoolExpr:
        n = rng.randint(self.min_value, self.max_value)
        return BoolExpr(f"is_prime({n})", self._is_prime(n))


@dataclass(frozen=True)
class RangeGen(ExprGenerator):
    """Generate a <= x <= b."""

    min_value: int = 1
    max_value: int = 100

    def generate(self, rng: random.Random) -> BoolExpr:
        # Ensure we can always choose a < b
        if self.max_value <= self.min_value:
            a = self.min_value
            b = self.max_value
        else:
            a = rng.randint(self.min_value, self.max_value - 1)
            b = rng.randint(a + 1, self.max_value)
        x = rng.randint(self.min_value, self.max_value)
        expr = f"{a} <= {x} <= {b}"
        return BoolExpr(expr, a <= x <= b)


# ============================
# Combinator (internal node)
# ============================


@dataclass(frozen=True)
class CombinatorSpec:
    """Specification for a logical combinator.

    method: "AND" | "OR" | None (None = choose randomly per instance)
    not_ratio: chance to wrap NOT(...) for each child input
    """

    method: Optional[str] = None
    not_ratio: float = 0.0


class CombinatorNode(ExprGenerator):
    """Combines N child generators with AND/OR and optional NOT per child."""

    def __init__(
        self,
        children: List[ExprGenerator],
        spec: CombinatorSpec,
    ) -> None:
        self.children = children
        if spec.method not in (None, "AND", "OR"):
            raise ValueError("CombinatorSpec.method must be 'AND', 'OR', or None")
        if not (0.0 <= spec.not_ratio <= 1.0):
            raise ValueError("CombinatorSpec.not_ratio must be in [0, 1]")
        self.spec = spec

    def generate(self, rng: random.Random) -> BoolExpr:
        method = self.spec.method or rng.choice(["AND", "OR"])  # choose per instance if None

        rendered: List[str] = []
        values: List[bool] = []
        for child in self.children:
            bexpr = child.generate(rng)
            use_not = rng.random() < self.spec.not_ratio
            text = f"NOT({bexpr.text})" if use_not else bexpr.text
            val = (not bexpr.value) if use_not else bexpr.value
            rendered.append(text)
            values.append(val)

        joiner = f" {method} "
        full_text = "(" + joiner.join(rendered) + ")"
        if method == "AND":
            full_val = all(values)
        else:
            full_val = any(values)
        return BoolExpr(full_text, full_val)


# ============================
# Difficulty / generation configs
# ============================


@dataclass(frozen=True)
class DifficultyConfig:
    name: str
    depth_min: int
    depth_max: int
    args_min: int
    args_max: int
    leaves: List[ExprGenerator]
    combinators: List[CombinatorSpec]


class BBEEHGenerator:
    """Tree-based boolean expression generator (deterministic via seed)."""

    def __init__(self, root: DifficultyConfig, inner: Optional[DifficultyConfig] = None, seed: Optional[int] = None):
        self.root = root
        self.inner = inner or root
        self._validate_config(self.root)
        self._validate_config(self.inner)
        self.rng = random.Random(seed)

    @staticmethod
    def _validate_config(cfg: DifficultyConfig) -> None:
        if cfg.depth_min < 0 or cfg.depth_max < 0 or cfg.depth_min > cfg.depth_max:
            raise ValueError("Invalid depth range in DifficultyConfig")
        if cfg.args_min < 2 or cfg.args_max < 2 or cfg.args_min > cfg.args_max:
            raise ValueError("Invalid args range in DifficultyConfig (need at least 2 children per combinator)")
        if not cfg.leaves:
            raise ValueError("DifficultyConfig must include at least one leaf generator")
        if not cfg.combinators:
            raise ValueError("DifficultyConfig must include at least one combinator spec")

    # ---- internal: tree construction ----
    def _build_node(self, depth: int, is_root: bool) -> ExprGenerator:
        cfg = self.root if is_root else self.inner
        if depth <= 0:
            return self.rng.choice(cfg.leaves)

        n_args = self.rng.randint(cfg.args_min, cfg.args_max)
        spec = self.rng.choice(cfg.combinators)
        children = [self._build_node(depth - 1, False) for _ in range(n_args)]
        return CombinatorNode(children, spec)

    # ---- public API ----
    def generate_one(self, target_depth: int) -> BoolExpr:
        node = self._build_node(target_depth, True)
        return node.generate(self.rng)

    def generate_many(self, count: int, depth_range: tuple[int, int]) -> List[BoolExpr]:
        lo, hi = depth_range
        return [self.generate_one(self.rng.randint(lo, hi)) for _ in range(count)]


# ============================
# Dataset building utilities
# ============================


def make_splits(
    total: int, train_ratio: float, val_ratio: float, rng: random.Random
) -> tuple[List[int], List[int], List[int]]:
    if total <= 0:
        raise ValueError("Total number of samples must be positive")
    if not (0.0 < train_ratio < 1.0 and 0.0 <= val_ratio < 1.0 and train_ratio + val_ratio < 1.0):
        raise ValueError("Invalid split ratios. Ensure 0<train<1, 0<=val<1 and train+val<1.")
    idxs = list(range(total))
    rng.shuffle(idxs)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train : n_train + n_val]
    return train_idx, val_idx, idxs[n_train + n_val :]


def build_hf_dataset(samples: List[dict]) -> Dataset:
    # Rely on automatic feature inference; simple flat structure keeps it robust
    return Dataset.from_list(samples)


def generate_dataset(
    generator: BBEEHGenerator,
    num_samples: int,
    train_ratio: float,
    val_ratio: float,
    seed: Optional[int],
) -> Dataset:
    rng = random.Random(seed)

    # Decide depth per sample based on root config range
    depth_lo, depth_hi = generator.root.depth_min, generator.root.depth_max

    # Generate all expressions first (deterministically)
    exprs = generator.generate_many(num_samples, (depth_lo, depth_hi))

    # Compute splits
    train_idx, val_idx, _ = make_splits(num_samples, train_ratio, val_ratio, rng)
    train_set, val_set = set(train_idx), set(val_idx)

    # Convert to sample rows with split column
    rows = []
    for i, expr in enumerate(exprs):
        if i in train_set:
            split = "train"
        elif i in val_set:
            split = "val"
        else:
            split = "test"
        rows.append(
            {
                "id": i,
                "problem": expr.text,
                "answer": str(expr.value).lower(),  # "true" | "false"
                "value": expr.value,  # boolean field for convenience
                "difficulty": generator.root.name,
                "split": split,
            }
        )

    return build_hf_dataset(rows)


# ============================
# Default example configuration (edit as desired)
# ============================


def example_configs() -> tuple[DifficultyConfig, DifficultyConfig, DifficultyConfig]:
    # Leaf generators
    threshold = ThresholdGen(min_value=1, max_value=50)
    comparison = ComparisonGen(min_value=1, max_value=30)
    modulo = ModuloGen(min_value=1, max_value=50, max_divisor=8)
    prime = PrimeGen(min_value=2, max_value=50)
    range_gen = RangeGen(min_value=1, max_value=50)

    # Combinator specs
    simple = CombinatorSpec(method=None, not_ratio=0.15)  # pick AND/OR randomly per node
    complex_ = CombinatorSpec(method=None, not_ratio=0.35)

    easy = DifficultyConfig(
        name="easy",
        depth_min=1,
        depth_max=2,
        args_min=2,
        args_max=4,
        leaves=[threshold, comparison, modulo],
        combinators=[simple],
    )
    medium = DifficultyConfig(
        name="medium",
        depth_min=2,
        depth_max=3,
        args_min=2,
        args_max=4,
        leaves=[threshold, comparison, modulo, prime, range_gen],
        combinators=[simple, complex_],
    )
    hard = DifficultyConfig(
        name="hard",
        depth_min=2,
        depth_max=3,
        args_min=4,
        args_max=6,
        leaves=[threshold, comparison, modulo, prime, range_gen],
        combinators=[complex_, simple],
    )
    return easy, medium, hard


# ============================
# Main (edit args by hand)
# ============================


def main() -> None:
    # Choose configs
    easy, medium, hard = example_configs()

    # Example: hard root, hard-like inner but with fewer args
    inner = DifficultyConfig(
        name="hard-inner",
        depth_min=hard.depth_min,
        depth_max=hard.depth_max,
        args_min=max(2, hard.args_min // 2),
        args_max=max(2, hard.args_max // 2),
        leaves=hard.leaves,
        combinators=hard.combinators,
    )

    gen = BBEEHGenerator(root=hard, inner=inner, seed=42)

    # Dataset config (edit freely)
    num_samples = 10000
    train_ratio = 0.7
    val_ratio = 0.2
    output_dir = (Path(__file__).parent / "data").as_posix()  # saves next to this file
    save_seed = 42

    ds = generate_dataset(gen, num_samples, train_ratio, val_ratio, seed=save_seed)

    # Quick sanity prints
    print("Dataset size:", len(ds))
    # Split sizes
    split_counts = {"train": 0, "val": 0, "test": 0}
    for s in ds["split"]:
        split_counts[s] = split_counts.get(s, 0) + 1
    print("Split sizes:", split_counts)

    # Balanced? quick histogram
    hist = {"true": 0, "false": 0}
    for a in ds["answer"]:
        hist[a] += 1
    print("Answer distribution (all):", hist)

    out_path = Path(output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(out_path.as_posix())
    print("Saved dataset to:", out_path.as_posix())


if __name__ == "__main__":
    main()
