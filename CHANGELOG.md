# Changelog

## 0.1.0

- `numeric_grounding` scorer: flags numbers in model output that are not in the input, tool results or metadata.
- Handles thousands separators, percentages, `k` suffixes, full-width digits and rounding.
- `extract_numbers` and `ungrounded_numbers` helpers for use outside Inspect.
