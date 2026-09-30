"""Custom quartodoc renderer for the qte docs.

Quartodoc renders an empty ``EnumName()`` constructor signature for string
enums, which is misleading: enum members are accessed as attributes and are
never constructed that way. This renderer replaces that signature with the
actual class definition, so the members and their values appear right below the
enum name.
"""

from quartodoc import layout
from quartodoc.renderers.md_renderer import MdRenderer

_orig_signature = MdRenderer.signature


def _target(obj: object) -> object:
    return getattr(obj, "final_target", obj)


def _is_enum(obj: object) -> bool:
    bases = getattr(_target(obj), "bases", None) or []
    return any("Enum" in getattr(base, "name", str(base)) for base in bases)


def _enum_signature(obj: object) -> str:
    target = _target(obj)
    bases = getattr(target, "bases", None) or []
    base = getattr(bases[0], "name", "Enum") if bases else "Enum"
    name = getattr(target, "name", "?")
    lines = [f"class {name}({base}):"]
    for member_name, member in getattr(target, "members", {}).items():
        value = getattr(member, "value", None)
        lines.append(f"    {member_name} = {value}" if value is not None else f"    {member_name}")
    body = "\n".join(lines)
    return f"```python\n{body}\n```"


def _signature(self: MdRenderer, el: object, source: object | None = None) -> str:
    if isinstance(el, layout.DocClass) and _is_enum(el.obj):
        return _enum_signature(el.obj)
    if source is None:
        return _orig_signature(self, el)
    return _orig_signature(self, el, source)


MdRenderer.signature = _signature

# The builder instantiates `Renderer` from this module; reuse the patched class.
Renderer = MdRenderer
