"""Check that a test-expectation change has the shape maintainers accept.

transformers keeps hardware-specific expected values in ``Expectations({key:
value})`` blocks, keyed ``(device, compute_capability)``. The way maintainers
re-record a value that drifted on the CI runner is narrow, and the evidence is
in their merges: on 2026-10-11, of the serge PRs opened in September and October,
3 of 3 value updates that touched only the runner's ``("cuda", 8)`` entry merged
cleanly (#48515, #49068, #49076). The one that overwrote the broad ``("cuda",
None)`` key merged with a wrong value (#48580). The rest of the open queue
overwrote ``(None, None)``, rewrote XPU values with no XPU evidence, or edited a
plain literal in place.

The rule, as this module enforces it:

* only the CI runner's key may change or be added: ``("cuda", 8)``, or
  ``("cuda", (8, 6))`` where the block keys by ``(major, minor)``
  (:data:`RUNNER_KEYS`; the runners are NVIDIA A10G, compute capability 8.6);
* every other key keeps its value. The one move allowed is *demotion*: an
  existing value may move to ``(None, None)`` unchanged, which is how #48198 did
  it;
* a new ``Expectations`` block (converting a plain literal) may hold only
  ``(None, None)`` and a runner key;
* in an expectation-only patch, nothing outside an ``Expectations`` block may
  change except the glue a conversion needs (the import, the
  ``get_expectation()`` call). That catches loosened tolerances and plain
  literals overwritten in place. One exemption: an assertion on the *shape or
  length of the model's inputs* (``inputs.input_ids.shape == (2, 422)``). The
  tokenizer and image processor run before the model and do not depend on the
  GPU, so a runner key is the wrong home for that value; #49507 (pegasus) was
  sent back three times for not putting a tokenizer length under ``("cuda",
  8)``. The exemption needs an input name *and* a ``.shape``/``len(``/
  ``.size(`` on the line, so an output count (``len(results["scores"])``,
  #47698) is still held to the rule.

The runner-key value itself is checked against what it replaces (the old
runner value, or the ``(None, None)`` value when the key is new, including a
plain literal just converted into a block, #49243):

* numbers may move by at most :data:`MAX_RUNNER_DRIFT` of their scale;
* the *count* of numbers may not change: three detections where there were
  five (#47698) is a different result, not drift;
* text may not gain a control character or a stutter
  (:func:`reviewbot.expectation_guard.degenerate_text`; #49244, #47983).

This only reads the two versions of a test file. It cannot tell whether a new
value is *right* (see :mod:`reviewbot.expectation_guard` for degenerate values);
it checks that the change is shaped so a reviewer can see what the new hardware
changed and the old value still holds where it held.
"""

from __future__ import annotations

import ast
import difflib
import re
from dataclasses import dataclass

from .expectation_guard import degenerate_text

__all__ = ["RUNNER_KEYS", "expectation_key_problems", "RULE_TEXT"]

RUNNER_KEYS = frozenset({("cuda", 8), ("cuda", (8, 6))})
GENERIC_KEY = (None, None)

RULE_TEXT = (
    'The CI runner is an NVIDIA A10G: its key is `("cuda", 8)`, or '
    '`("cuda", (8, 6))` where the block already keys by (major, minor). Change or '
    "add ONLY that key. Every other key keeps its value: never overwrite "
    '`(None, None)`, `("cuda", None)`, another CUDA generation, `"xpu"` or '
    '`"rocm"`. You may move an existing value, unchanged, to `(None, None)`. If '
    "the expected value is a plain literal, convert it to `Expectations({...})` "
    "with the old value under `(None, None)` and the new one under the runner "
    "key. Do not change tolerances or anything else outside the `Expectations` "
    "block."
)

# A runner-key value may move by this fraction of the block's largest
# magnitude. Accepted drift is far below it: #48515 moved 0.4% of its scale,
# #49068 (bf16 logits near zero) 3.6%. The rejected pvt groups moved 11% and 45%.
MAX_RUNNER_DRIFT = 0.05

_NUMBER = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+(?:[eE][-+]?\d+)")
_TOLERANCE = re.compile(r"\b(atol|rtol)\s*=\s*([0-9.]+(?:[eE][-+]?\d+)?)")
# Changed lines outside a block that read as an expected value or a check: an
# assertion, an EXPECTED_*/expected_* name, a tolerance, or a bare literal line
# (a multi-line tensor or string continuation). Anything else is test code, and
# a test-code fix is not this module's business (#47697 renamed a config key).
_EXPECTED_LINE = re.compile(
    r"expect|assert|\batol\b|\brtol\b|^\s*[\[\]\"'\d-]", re.IGNORECASE
)
# An assertion on the shape or length of what goes INTO the model (see the
# module docstring, #49507). Both halves must match, and nothing output-side may
# appear on the line.
_INPUT_NAME = re.compile(
    r"\b(?:inputs|input_ids|attention_mask|token_type_ids|pixel_values"
    r"|input_features|input_values|encoding|encoded)\b"
)
_SIZE_OF = re.compile(r"\.shape\b|\blen\(|\.size\(")
_OUTPUT_NAME = re.compile(
    r"output|logits|generat|result|pred|decoded|score", re.IGNORECASE
)
# Lines a literal → Expectations conversion has to touch outside the block.
_GLUE = re.compile(
    r"get_expectation\(|^\s*Expectations,?\s*$|import\b.*\bExpectations\b"
)


@dataclass
class _Block:
    scope: tuple[str, int]  # (enclosing function, ordinal within it)
    entries: dict  # key tuple -> normalized value source
    first: int  # 1-based line span, inclusive
    last: int


def _key(node: ast.expr):
    try:
        key = ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None
    return key if isinstance(key, tuple) and len(key) == 2 else None


def _is_expectations(call: ast.Call) -> bool:
    func = call.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    return (
        name == "Expectations"
        and bool(call.args)
        and isinstance(call.args[0], ast.Dict)
    )


def _blocks(src: str) -> list[_Block] | None:
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    out: list[_Block] = []
    counts: dict[str, int] = {}

    def visit(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(child, child.name)
                continue
            if isinstance(child, ast.Call) and _is_expectations(child):
                d = child.args[0]
                entries = {}
                for k, v in zip(d.keys, d.values):
                    key = _key(k) if k is not None else None
                    if key is not None:
                        entries[key] = ast.unparse(v)
                ordinal = counts.get(scope, 0)
                counts[scope] = ordinal + 1
                out.append(
                    _Block(
                        (scope, ordinal),
                        entries,
                        child.lineno,
                        child.end_lineno or child.lineno,
                    )
                )
            visit(child, scope)

    visit(tree, "<module>")
    return out


def _numbers(value_src: str) -> list[float]:
    return [float(x) for x in _NUMBER.findall(value_src)]


def _strings(value_src: str) -> list[str] | None:
    """Every string in a literal value; ``None`` when it is not a literal."""
    try:
        tree = ast.parse(value_src, mode="eval")
    except SyntaxError:
        return None
    return [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]


def _drift(old_src: str, new_src: str) -> float | None:
    """Largest change between two same-shaped numeric values, as a fraction of
    the old value's largest magnitude; ``None`` for text or a shape change."""
    old, new = _numbers(old_src), _numbers(new_src)
    if not old or len(old) != len(new):
        return None
    scale = max(abs(x) for x in old) or 1.0
    return max(abs(a - b) for a, b in zip(old, new)) / scale


def _runner_value_problems(where: str, key, before: str, after: str) -> list[str]:
    """What is wrong with a runner-key value given the value it replaces."""
    old_text, new_text = _strings(before), _strings(after)
    if old_text is None or new_text is None:
        return []
    if not old_text and not new_text:
        old_n, new_n = _numbers(before), _numbers(after)
        if old_n and new_n and len(old_n) != len(new_n):
            return [
                (
                    f"{where}: the {_fmt(key)} value has {len(new_n)} numbers where "
                    f"it had {len(old_n)}; a different count is a different result, "
                    "not hardware drift."
                )
            ]
        drift = _drift(before, after)
        if drift is not None and drift > MAX_RUNNER_DRIFT:
            return [
                (
                    f"{where}: the {_fmt(key)} values move by {drift:.0%} of the "
                    f"block's scale; hardware drift is about 1%, so this looks like "
                    "a regression, not a new baseline."
                )
            ]
        return []
    out = []
    for text in new_text:
        why = degenerate_text(text)
        if why and not any(degenerate_text(t) == why for t in old_text):
            out.append(
                f"{where}: the new {_fmt(key)} text {why}; that is broken "
                "output, not a new baseline."
            )
            break
    return out


def _loosened(old_lines: list[str], new_lines: list[str]) -> list[str]:
    def tolerances(lines):
        out: dict[str, float] = {}
        for ln in lines:
            for name, value in _TOLERANCE.findall(ln):
                try:
                    out[name] = max(out.get(name, 0.0), float(value))
                except ValueError:
                    pass
        return out

    old, new = tolerances(old_lines), tolerances(new_lines)
    return [
        f"{n} {old[n]:g} -> {new[n]:g}"
        for n in sorted(new)
        if n in old and new[n] > old[n]
    ]


def _fmt(key) -> str:
    return repr(key).replace("'", '"')


def _is_input_size(line: str) -> bool:
    return bool(
        _INPUT_NAME.search(line)
        and _SIZE_OF.search(line)
        and not _OUTPUT_NAME.search(line)
    )


def _within(line: int, blocks: list[_Block]) -> bool:
    return any(b.first <= line <= b.last for b in blocks)


def expectation_key_problems(
    path: str, old_src: str, new_src: str, *, expectation_only: bool = False
) -> list[str]:
    """What is wrong with how one test file's expectations were changed, as
    one sentence per problem; ``[]`` when the change has the accepted shape.

    ``expectation_only`` turns on the "nothing outside an Expectations block"
    rule, which only makes sense for a patch that changes literals alone: a real
    fix may well edit test code."""
    old_blocks, new_blocks = _blocks(old_src), _blocks(new_src)
    if old_blocks is None or new_blocks is None:
        return []
    problems: list[str] = []
    old_by_scope = {b.scope: b for b in old_blocks}

    for new in new_blocks:
        old = old_by_scope.get(new.scope)
        where = f"`{path}` (line {new.first})"
        if old is None:
            extra = [
                k for k in new.entries if k != GENERIC_KEY and k not in RUNNER_KEYS
            ]
            if extra:
                problems.append(
                    f"{where}: a new Expectations block may only hold `(None, None)` "
                    f"and the runner key, not {', '.join(_fmt(k) for k in extra)}."
                )
            before = new.entries.get(GENERIC_KEY)
            for key in RUNNER_KEYS & new.entries.keys():
                if before is not None:
                    problems += _runner_value_problems(
                        where, key, before, new.entries[key]
                    )
            continue
        old_values = set(old.entries.values())
        for key, value in old.entries.items():
            if key in RUNNER_KEYS or new.entries.get(key) == value:
                continue
            demoted = (
                key not in new.entries
                and GENERIC_KEY not in old.entries
                and new.entries.get(GENERIC_KEY) == value
            )
            if demoted:
                continue
            verb = "changed" if key in new.entries else "removed"
            problems.append(
                f"{where}: {verb} the {_fmt(key)} value, which is not the CI runner's key."
            )
        for key in RUNNER_KEYS & new.entries.keys():
            before = old.entries.get(key) or new.entries.get(GENERIC_KEY)
            if before and before != new.entries[key]:
                problems += _runner_value_problems(where, key, before, new.entries[key])
        for key, value in new.entries.items():
            if key in old.entries or key in RUNNER_KEYS:
                continue
            if key == GENERIC_KEY and value in old_values:
                continue  # a demotion; checked from the old side above
            what = (
                "a `(None, None)` value that is not an existing one"
                if key == GENERIC_KEY
                else f"a {_fmt(key)} entry"
            )
            problems.append(f"{where}: added {what}; only the runner key may be added.")

    old_lines, new_lines = old_src.splitlines(), new_src.splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            loosened = _loosened(old_lines[i1:i2], new_lines[j1:j2])
            if loosened:
                problems.append(
                    f"`{path}` (line {j1 + 1}): loosened a tolerance "
                    f"({', '.join(loosened)})."
                )
    if expectation_only:
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            in_block = any(_within(i + 1, old_blocks) for i in range(i1, i2)) or any(
                _within(j + 1, new_blocks) for j in range(j1, j2)
            )
            changed = [
                ln
                for ln in new_lines[j1:j2] + old_lines[i1:i2]
                if ln.strip() and not ln.lstrip().startswith("#")
            ]
            glue = changed and all(
                _GLUE.search(ln) for ln in new_lines[j1:j2] if ln.strip()
            )
            if in_block or glue or not changed:
                continue
            if _loosened(old_lines[i1:i2], new_lines[j1:j2]):
                continue  # already reported above
            flagged = [
                ln
                for ln in changed
                if _EXPECTED_LINE.search(ln) and not _is_input_size(ln)
            ]
            if not flagged:
                continue
            sample = flagged[0].strip()
            problems.append(
                f"`{path}` (line {j1 + 1}): changed `{sample[:80]}` outside any "
                "Expectations block."
            )
    return problems
