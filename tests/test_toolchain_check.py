from reviewbot.toolchain_check import find_mismatches

# Shaped like importlib.metadata.requires("transformers") on main, 2026-11-18.
REQUIRES = [
    "huggingface-hub<3.0,>=1.31.0",
    "tokenizers<0.24.0,>=0.23.1",
    "numpy>=1.17",
    "transformers-mlinter==0.1.6; extra == 'quality'",
    "ruff==0.14.10; extra == 'quality'",
    "torch>=2.4; extra == 'torch'",
    "tensorflow; python_version < '3.0'",
]


def _versions(**v):
    table = {k.replace("_", "-"): val for k, val in v.items()}
    return lambda name: table.get(name.lower().replace("_", "-"))


GOOD = dict(
    huggingface_hub="2.4.0",
    tokenizers="0.23.3",
    numpy="2.3.5",
    transformers_mlinter="0.1.6",
    ruff="0.14.10",
)


def test_consistent_image_passes():
    assert (
        find_mismatches(REQUIRES, extras=["quality"], version_of=_versions(**GOOD))
        == []
    )


def test_2026_11_16_mlinter_drift_is_caught():
    # The image built 2026-10-30 held mlinter 0.1.5 against main's ==0.1.6.
    found = find_mismatches(
        REQUIRES,
        extras=["quality"],
        version_of=_versions(**{**GOOD, "transformers_mlinter": "0.1.5"}),
    )
    assert [(m.name, m.installed) for m in found] == [("transformers-mlinter", "0.1.5")]
    assert "0.1.6" in found[0].describe()


def test_core_pin_drift_is_caught_without_extras():
    found = find_mismatches(
        REQUIRES, version_of=_versions(**{**GOOD, "huggingface_hub": "1.20.0"})
    )
    assert [m.name for m in found] == ["huggingface-hub"]


def test_extra_not_requested_is_ignored():
    # torch is not in the quality image, and the gate does not need it.
    assert (
        find_mismatches(REQUIRES, extras=["quality"], version_of=_versions(**GOOD))
        == []
    )


def test_missing_quality_dependency_is_a_mismatch():
    v = {k: val for k, val in GOOD.items() if k != "ruff"}
    found = find_mismatches(REQUIRES, extras=["quality"], version_of=_versions(**v))
    assert [(m.name, m.installed) for m in found] == [("ruff", None)]


def test_prerelease_satisfies_lower_bound():
    found = find_mismatches(
        ["tokenizers>=0.23.1"], version_of=_versions(tokenizers="0.24.0rc1")
    )
    assert found == []
