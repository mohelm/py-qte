from collections.abc import Iterable


def _make_header(
    content: dict[str, str],
    separator: str,
    new_line_char: str,
    additional_padding: int = 2,
) -> Iterable[str]:
    max_length_keys = max(len(k) for k in content)
    for k, v in content.items():
        padding = max_length_keys - len(k) + additional_padding
        yield f"{k}{separator * padding}: {v}{new_line_char}"
