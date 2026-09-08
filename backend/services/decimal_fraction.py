"""
Decimal to Fraction Conversion Engine

Implements the exact 63 fraction mappings from Unilog's Decimal_Fraction.xlsx
(from 1/64 to 63/64).

Rule Book Standard:
  Manufacturers publish decimals; industrial trade buyers search fractions.
  - 0.5 -> 1/2
  - 50.25 in -> 50-1/4 in
  - 0.125 -> 1/8
  - 0.0625 -> 1/16
"""
import re
import math

# Complete 64th decimal-to-fraction table from Decimal_Fraction.xlsx
DECIMAL_FRACTION_TABLE = [
    (0.015625, "1/64"), (0.03125, "1/32"), (0.046875, "3/64"), (0.0625, "1/16"),
    (0.078125, "5/64"), (0.09375, "3/32"), (0.109375, "7/64"), (0.125, "1/8"),
    (0.140625, "9/64"), (0.15625, "5/32"), (0.171875, "11/64"), (0.1875, "3/16"),
    (0.203125, "13/64"), (0.21875, "7/32"), (0.234375, "15/64"), (0.25, "1/4"),
    (0.265625, "17/64"), (0.28125, "9/32"), (0.296875, "19/64"), (0.3125, "5/16"),
    (0.328125, "21/64"), (0.34375, "11/32"), (0.359375, "23/64"), (0.375, "3/8"),
    (0.390625, "25/64"), (0.40625, "13/32"), (0.421875, "27/64"), (0.4375, "7/16"),
    (0.453125, "29/64"), (0.46875, "15/32"), (0.484375, "31/64"), (0.5, "1/2"),
    (0.515625, "33/64"), (0.53125, "17/32"), (0.546875, "35/64"), (0.5625, "9/16"),
    (0.578125, "37/64"), (0.59375, "19/32"), (0.609375, "39/64"), (0.625, "5/8"),
    (0.640625, "41/64"), (0.65625, "21/32"), (0.671875, "43/64"), (0.6875, "11/16"),
    (0.703125, "45/64"), (0.71875, "23/32"), (0.734375, "47/64"), (0.75, "3/4"),
    (0.765625, "49/64"), (0.78125, "25/32"), (0.796875, "51/64"), (0.8125, "13/16"),
    (0.828125, "53/64"), (0.84375, "27/32"), (0.859375, "55/64"), (0.875, "7/8"),
    (0.890625, "57/64"), (0.90625, "29/32"), (0.921875, "59/64"), (0.9375, "15/16"),
    (0.953125, "61/64"), (0.96875, "31/32"), (0.984375, "63/64"),
]


def decimal_to_fraction(val: float, tolerance: float = 0.008) -> str | None:
    """
    Looks up decimal part in the 64ths table.
    Returns fraction string like '1/4', '50-1/4', or None if no match.
    """
    if val <= 0:
        return None

    int_part = int(math.floor(val))
    dec_part = val - int_part

    # Exact or near integer
    if dec_part < tolerance:
        return str(int_part) if int_part > 0 else None
    if (1.0 - dec_part) < tolerance:
        return str(int_part + 1)

    # Find closest match in table
    best_match = None
    best_diff = tolerance
    for dec_val, frac_str in DECIMAL_FRACTION_TABLE:
        diff = abs(dec_part - dec_val)
        if diff < best_diff:
            best_diff = diff
            best_match = frac_str

    if best_match:
        if int_part > 0:
            return f"{int_part}-{best_match}"
        return best_match

    return None


def convert_measurement_string(val_str: str, uom: str | None = None) -> tuple[str, bool]:
    """
    Given an attribute value string (e.g. '50.25', '0.5', '24.125'),
    converts any imperial / decimal representation into Unilog trade fraction format.
    Returns (converted_string, did_convert).
    """
    if not val_str:
        return val_str, False

    clean_str = str(val_str).strip()

    # Only convert if UOM is inch / in or unspecified dimension
    is_inch = uom and any(u in uom.lower() for u in ["in", "inch", '"'])

    # Try direct numeric match
    try:
        fval = float(clean_str)
        frac = decimal_to_fraction(fval)
        if frac:
            return frac, True
    except ValueError:
        pass

    # Regex search for decimal numbers inside strings like '50.25 in' or '1/2 x 18'
    def replace_dec(m):
        try:
            fv = float(m.group(0))
            fr = decimal_to_fraction(fv)
            return fr if fr else m.group(0)
        except Exception:
            return m.group(0)

    converted = re.sub(r'\b\d+\.\d+\b', replace_dec, clean_str)
    return converted, (converted != clean_str)
