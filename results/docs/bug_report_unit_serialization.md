# Bug Report: pulse.Variable Unit Serialization Loses Scale Factor

## Summary

Serializing `pulse.Variable` material parameters to JSON and reconstructing them
produces a material that is **1000x softer** than intended, because the kPa-to-Pa
conversion factor is silently dropped.

## Reproduction

```python
import pulse

# Original parameter (from pulse defaults)
params = pulse.HolzapfelOgden.transversely_isotropic_parameters()
v = params["a"]
# v.value = 2.28, v.original_unit = "kilopascal", v.factor = 1000
# v.to_base_units() = 2280.0  (correct: 2.28 kPa = 2280 Pa)

# Naive serialization (the bug)
saved = {"value": float(v.value), "unit": str(v.unit)}
# saved = {"value": 2.28, "unit": "kilogram / meter / second ** 2"}

# Reconstruction
v2 = pulse.Variable(saved["value"], saved["unit"])
# v2.value = 2.28, v2.original_unit = "kilogram / meter / second ** 2", v2.factor = 1
# v2.to_base_units() = 2.28  (WRONG: 1000x too small)
```

## Root Cause

`pulse.Variable` has two unit representations:

| Attribute        | Original              | After round-trip           |
|------------------|-----------------------|----------------------------|
| `.value`         | `2.28`                | `2.28`                     |
| `.original_unit` | `kilopascal`          | `kilogram / meter / second ** 2` |
| `.unit`          | `kilogram / meter / second ** 2` | `kilogram / meter / second ** 2` |
| `.factor`        | `1000.0`              | `1`                        |
| `.to_base_units()` | `2280.0`            | `2.28`                     |

`.unit` returns the SI decomposition, which is `kg/(m*s^2)` = Pa for both kPa and Pa.
When you reconstruct `Variable(2.28, "kilogram / meter / second ** 2")`, it interprets
the value as 2.28 Pa (not 2.28 kPa), because the unit string IS already base SI.

The trap: `.value` and `.unit` look identical before and after serialization, so the
bug is invisible unless you check `.factor` or `.to_base_units()`.

## Impact on Our Simulations

- Material passive stiffness 1000x too low in postprocessed metrics
- Active stress unaffected (depends on Ta, not material params)
- Energy conservation check broken: internal work != external work (29% gap)
- Passive work: 7.7e-3 J (correct) vs 7.7e-6 J (with bug)

## Fix Applied

In `complete_cycle.py`, changed the serialization from:
```python
# BUG: str(v.unit) -> "kilogram / meter / second ** 2" (loses kilo prefix)
{"value": float(v.value), "unit": str(v.unit)}
```
to:
```python
# FIX: str(v.original_unit) -> "kilopascal" (preserves scale factor)
{"value": float(v.value), "unit": str(v.original_unit)}
```

## Suggested Improvements for fenicsx-pulse

### 1. Make `Variable` round-trip safe

The core issue is that `Variable(v.value, str(v.unit))` is not an identity operation.
This violates the principle of least surprise. Options:

**Option A** — Make `str(v.unit)` return the original unit (simplest):
```python
@property
def unit(self):
    return self._original_unit  # "kilopascal" not "kilogram / meter / second ** 2"
```
Add a separate `.base_unit` property for the SI decomposition if needed.

**Option B** — Make `Variable.__init__` detect SI-decomposed strings and apply no factor:
This is fragile and not recommended.

**Option C** — Add serialization helpers:
```python
class Variable:
    def to_dict(self):
        return {"value": self.value, "unit": str(self.original_unit)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["value"], d["unit"])
```
This makes the correct pattern obvious and avoids users having to know about
`.original_unit` vs `.unit`.

### 2. Add a warning or guard

At minimum, document that `str(v.unit)` is NOT safe for serialization. Ideally,
raise a warning or override `__repr__`/`__str__` to show the original unit:

```python
>>> v = pulse.Variable(2.28, "kPa")
>>> repr(v)
"Variable(2.28, 'kilopascal')"  # not "Variable(2.28, 'kilogram / meter / second ** 2')"
```

### 3. Consider storing base-unit values internally

An alternative design: always convert to base SI on construction and store the
converted value. Then `.value` would return Pa directly:

```python
v = Variable(2.28, "kPa")
v.value  # -> 2280.0 (Pa)
v.unit   # -> "pascal"
```

This eliminates the factor ambiguity entirely, though it changes the API.

### Recommendation

Option C (serialization helpers) is the least disruptive and most practical.
Option A is cleaner long-term. Either way, the current `.unit` property returning
a different string than what was passed to the constructor is a footgun.


$$
r(X, Y) = \frac{\sum_i (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_i (X_i - \bar{X})^2 \cdot \sum_i (Y_i - \bar{Y})^2}}
$$