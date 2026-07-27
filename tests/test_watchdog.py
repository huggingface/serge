from reviewbot.watchdog import RepeatedIntentWatchdog, WatchdogEarlyExit


def test_repeated_intent_triggers_after_min_tool_calls():
    watchdog = RepeatedIntentWatchdog(
        repeated_intent_limit=2,
        min_tool_calls=2,
        post_intent_tool_calls=0,
        post_intent_reasoning_chars=0,
    )

    assert watchdog.observe("reasoning", "the fix should be to simplify") is None
    assert watchdog.observe("tool", "read_file({})") is None
    assert watchdog.observe("tool", "grep({})") is None

    trigger = watchdog.observe("reasoning", "the fix should be to simplify")

    assert trigger is not None
    assert trigger.occurrences == 2
    assert trigger.tool_calls == 2
    assert trigger.pattern == r"\bthe fix (?:is|should be) to\b"


def test_post_intent_tool_calls_trigger():
    watchdog = RepeatedIntentWatchdog(
        repeated_intent_limit=3,
        min_tool_calls=1,
        post_intent_tool_calls=2,
        post_intent_reasoning_chars=0,
    )

    assert watchdog.observe("tool", "read_file({})") is None
    assert watchdog.observe("reasoning", "the fix is to add an override") is None
    assert watchdog.observe("tool", "grep({})") is None
    trigger = watchdog.observe("tool", "read_file({})")

    assert trigger is not None
    assert trigger.reason == "Decisive intent was followed by more non-edit tool calls"


def test_no_patch_decision_triggers_on_followup_tool():
    watchdog = RepeatedIntentWatchdog(
        repeated_intent_limit=3,
        min_tool_calls=1,
        post_intent_tool_calls=1,
        post_intent_reasoning_chars=0,
    )

    assert watchdog.observe("tool", "read_file({})") is None
    assert (
        watchdog.observe(
            "reasoning",
            "I believe the slowness is expected framework overhead. No patch is needed.",
        )
        is None
    )
    trigger = watchdog.observe("tool", "grep({})")

    assert trigger is not None
    assert trigger.reason == "Decisive intent was followed by more non-edit tool calls"
    assert trigger.pattern == r"\b(?:no patch|no change) (?:is needed|needed)\b"


def test_no_patch_decision_triggers_on_more_reasoning():
    watchdog = RepeatedIntentWatchdog(
        repeated_intent_limit=3,
        min_tool_calls=1,
        post_intent_tool_calls=0,
        post_intent_reasoning_chars=10,
    )

    assert watchdog.observe("tool", "read_file({})") is None
    assert (
        watchdog.observe(
            "reasoning",
            "Final decision: no patch. The test is healthy compile overhead.",
        )
        is None
    )
    trigger = watchdog.observe("reasoning", "x" * 11)

    assert trigger is not None
    assert trigger.pattern == r"\bfinal decision:\s*(?:no patch|no change)\b"


def test_post_intent_reasoning_uses_total_chars_not_rolling_buffer():
    watchdog = RepeatedIntentWatchdog(
        repeated_intent_limit=3,
        min_tool_calls=1,
        post_intent_tool_calls=0,
        post_intent_reasoning_chars=10,
    )

    assert watchdog.observe("tool", "read_file({})") is None
    assert watchdog.observe("reasoning", "the fix is to add an override") is None
    trigger = watchdog.observe("reasoning", "x" * 11)

    assert trigger is not None
    assert trigger.reason == (
        "Decisive intent was followed by extended reasoning without an edit/final result"
    )


def test_edit_tool_suppresses_repeated_intent_watchdog():
    watchdog = RepeatedIntentWatchdog(
        repeated_intent_limit=2,
        min_tool_calls=1,
        post_intent_tool_calls=1,
        post_intent_reasoning_chars=1,
    )

    assert watchdog.observe("tool", "read_file({})") is None
    assert watchdog.observe("reasoning", "the fix is to add an override") is None
    assert watchdog.observe("tool", "apply_patch({})") is None
    assert watchdog.observe("reasoning", "the fix is to add an override") is None
    assert watchdog.trigger is None


def test_local_early_exit_uses_base_exception():
    watchdog = RepeatedIntentWatchdog(
        repeated_intent_limit=1,
        min_tool_calls=0,
        post_intent_tool_calls=0,
        post_intent_reasoning_chars=0,
    )
    trigger = watchdog.observe("reasoning", "the fix is to add an override")
    assert trigger is not None

    caught_by_exception = False
    try:
        try:
            raise WatchdogEarlyExit(trigger)
        except Exception:
            caught_by_exception = True
    except WatchdogEarlyExit as exc:
        assert not caught_by_exception
        assert exc.payload["reason"] == trigger.reason
