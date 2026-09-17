"""
explain.py — Plain-language translation layer for the compressor anomaly pipeline.

Takes the structured output of CompressorAnomalyPipeline.run() (severity, top_features,
flag_rate) and turns it into a worker-readable explanation. This is the layer that sits
between the ML pipeline and the dashboard / notification system.

Design choice: rule-based templates, not an LLM call. This keeps it fast, free,
deterministic, and fully explainable — every sentence traces back to a specific
number, with no risk of a language model inventing details. If the team wants more
natural/varied phrasing later, this module's output can be used as a structured
"briefing" that gets passed to an LLM prompt as grounding context, rather than
letting the LLM generate explanations from scratch.
"""

# Human-readable names and plain descriptions for each engineered feature.
# Extend this dict as more features get added to the pipeline.
FEATURE_DESCRIPTIONS = {
    'Oil_temperature_roll_mean': {
        'label': 'oil temperature',
        'high': 'running hotter than normal',
        'low': 'running cooler than normal',
    },
    'Oil_temperature_roll_trend': {
        'label': 'oil temperature trend',
        'high': 'climbing steadily',
        'low': 'dropping unexpectedly',
    },
    'H1_roll_std': {
        'label': 'pressure cycling pattern',
        'high': 'more irregular than usual',
        'low': 'flatter than usual — the compressor may not be cycling on and off normally',
    },
    'H1_cycle_transitions': {
        'label': 'on/off cycling behavior',
        'high': 'cycling more frequently than usual',
        'low': 'barely cycling — the compressor appears to be running continuously instead of switching off periodically',
    },
    'TP2_roll_std': {
        'label': 'primary pressure sensor variability',
        'high': 'more variable than usual',
        'low': 'flatter than usual',
    },
    'TP2_roll_range': {
        'label': 'primary pressure swing',
        'high': 'swinging more widely than usual',
        'low': 'barely changing, which is unusual for normal operation',
    },
    'TP3_roll_std': {
        'label': 'secondary pressure sensor variability',
        'high': 'more variable than usual',
        'low': 'flatter than usual',
    },
    'Reservoirs_roll_std': {
        'label': 'reservoir pressure variability',
        'high': 'more variable than usual',
        'low': 'flatter than usual',
    },
    'Motor_current_roll_std': {
        'label': 'motor current variability',
        'high': 'more variable than usual',
        'low': 'unusually steady, which can indicate continuous running instead of normal cycling',
    },
    'Motor_current_roll_range': {
        'label': 'motor current swing',
        'high': 'swinging more widely than usual',
        'low': 'barely changing, unusual for normal operation',
    },
}

SEVERITY_INTRO = {
    'ESCALATE': "URGENT: This compressor is showing strong signs of a developing problem.",
    'INVESTIGATE': "This compressor is showing signs that are worth checking.",
    'MONITOR': "This compressor is showing mild, early signs of unusual behavior. No action needed yet, but keep an eye on it.",
}

SEVERITY_ACTION = {
    'ESCALATE': "Recommended action: dispatch a maintenance check as soon as possible.",
    'INVESTIGATE': "Recommended action: schedule an inspection in the near term.",
    'MONITOR': "Recommended action: no immediate action — continue routine monitoring.",
}


def _describe_feature(feature_name, z_score, actual_value=None, normal_mean=None):
    """
    Turn one (feature, z-score) pair into a plain sentence fragment.

    IMPORTANT: our pipeline's explain() method (in pipeline.py) currently computes
    ABSOLUTE z-scores (always positive), so the sign of z_score alone cannot tell us
    whether the feature is unusually HIGH or unusually LOW. To describe direction
    correctly, pass actual_value and normal_mean for the feature -- if both are given,
    direction is derived from (actual_value - normal_mean) instead of z_score's sign.
    Without them, this falls back to a direction-neutral phrasing so it never states
    an incorrect direction (e.g. claiming H1_roll_std is "high" when it's actually
    collapsed toward zero, which is what real failures look like).
    """
    info = FEATURE_DESCRIPTIONS.get(feature_name)
    if info is None:
        return f"{feature_name.replace('_', ' ')} is {z_score:.1f} standard deviations from normal"

    if actual_value is not None and normal_mean is not None:
        direction = 'high' if actual_value >= normal_mean else 'low'
        description = info[direction]
        return f"{info['label']} is {description} ({z_score:.1f} standard deviations from normal)"

    return f"{info['label']} is unusual, {z_score:.1f} standard deviations from its normal range"


def explain_alert_plain(alert):
    """
    Convert one alert dict (as produced by CompressorAnomalyPipeline.run(), one row)
    into a plain-language report a field worker can read directly.

    Parameters
    ----------
    alert : dict or pandas.Series
        Must contain 'severity' (str), 'top_features' (list of (name, z_score) tuples),
        and 'flag_rate' (float, 0.5-1.0).

    Returns
    -------
    str — a short, plain-language explanation.
    """
    severity = alert['severity']
    top_features = alert['top_features']
    flag_rate = alert['flag_rate']

    intro = SEVERITY_INTRO.get(severity, "This compressor is showing unusual behavior.")
    action = SEVERITY_ACTION.get(severity, "Recommended action: review manually.")

    # top_features can be either [(name, z_score), ...] (direction-neutral fallback)
    # or [(name, z_score, actual_value, normal_mean), ...] (correct direction).
    feature_sentences = []
    for item in top_features:
        if len(item) == 4:
            name, z, actual_value, normal_mean = item
            feature_sentences.append(_describe_feature(name, z, actual_value, normal_mean))
        else:
            name, z = item
            feature_sentences.append(_describe_feature(name, z))
    if len(feature_sentences) == 1:
        reasons = feature_sentences[0]
    elif len(feature_sentences) == 2:
        reasons = f"{feature_sentences[0]}, and {feature_sentences[1]}"
    else:
        reasons = ", ".join(feature_sentences[:-1]) + f", and {feature_sentences[-1]}"

    persistence_note = (
        f"This pattern has been consistent for at least the last "
        f"{int(flag_rate * 10)} of the last 10 minutes."
        if flag_rate >= 0.5 else ""
    )

    report = f"{intro}\n\nWhat we're seeing: {reasons}.\n\n{persistence_note}\n\n{action}".strip()
    return report


def explain_alerts_batch(alerts_df):
    """
    Apply explain_alert_plain() to every row of an alerts dataframe
    (as returned by CompressorAnomalyPipeline.run()).

    Returns the same dataframe with an added 'plain_language' column.
    """
    alerts_df = alerts_df.copy()
    alerts_df['plain_language'] = alerts_df.apply(
        lambda row: explain_alert_plain(row), axis=1
    )
    return alerts_df


if __name__ == "__main__":
    # Quick manual test with a fabricated example alert
    example_alert = {
        'severity': 'ESCALATE',
        'top_features': [
            ('Oil_temperature_roll_mean', 3.29),
            ('TP2_roll_std', 1.55),
            ('H1_roll_std', 1.43),
        ],
        'flag_rate': 1.0,
    }
    print(explain_alert_plain(example_alert))
