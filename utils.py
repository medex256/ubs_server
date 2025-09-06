import re
from typing import Optional, Tuple


ROMAN_VALID_RE = re.compile(r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def roman_to_int(s: str) -> Optional[int]:
    if not isinstance(s, str) or not s:
        return None
    u = s.strip().upper()
    if not ROMAN_VALID_RE.match(u):
        return None
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    i = 0
    while i < len(u):
        v = values[u[i]]
        if i + 1 < len(u) and values[u[i + 1]] > v:
            total += values[u[i + 1]] - v
            i += 2
        else:
            total += v
            i += 1
    return total


def parse_english_number(s: str) -> Optional[int]:
    if not isinstance(s, str) or not s.strip():
        return None
    text = s.strip().lower()
    text = text.replace('-', ' ')
    tokens = [t for t in text.split() if t and t != 'and']
    if not tokens:
        return None

    units = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'a': 1
    }
    teens = {
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19
    }
    tens = {
        'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
    }
    scales = {
        'hundred': 100,
        'thousand': 1_000,
        'million': 1_000_000,
        'billion': 1_000_000_000,
        'trillion': 1_000_000_000_000
    }

    total = 0
    current = 0
    for t in tokens:
        if t in units:
            current += units[t]
        elif t in teens:
            current += teens[t]
        elif t in tens:
            current += tens[t]
        elif t == 'hundred':
            current = (current if current != 0 else 1) * 100
        elif t in ('thousand', 'million', 'billion', 'trillion'):
            total += (current if current != 0 else 1) * scales[t]
            current = 0
        else:
            return None
    return total + current


def _normalize_german(s: str) -> str:
    t = s.lower().strip()
    t = t.replace('ß', 'ss')
    t = t.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')
    t = t.replace('-', '').replace(' ', '')
    return t


def parse_german_number(s: str) -> Optional[int]:
    if not isinstance(s, str) or not s.strip():
        return None
    t = _normalize_german(s)

    units = {
        'null': 0, 'ein': 1, 'eins': 1, 'eine': 1, 'zwei': 2, 'drei': 3, 'vier': 4, 'fuenf': 5, 'sechs': 6, 'sieben': 7, 'acht': 8, 'neun': 9
    }
    teens = {
        'zehn': 10, 'elf': 11, 'zwoelf': 12, 'dreizehn': 13, 'vierzehn': 14, 'fuenfzehn': 15, 'sechzehn': 16, 'siebzehn': 17, 'achtzehn': 18, 'neunzehn': 19
    }
    tens = {
        'zwanzig': 20, 'dreissig': 30, 'vierzig': 40, 'fuenfzig': 50, 'sechzig': 60, 'siebzig': 70, 'achtzig': 80, 'neunzig': 90
    }

    def parse_under_hundred(seg: str) -> Optional[int]:
        if seg == '':
            return 0
        if seg in teens:
            return teens[seg]
        if seg in tens:
            return tens[seg]
        if 'und' in seg:
            pos = seg.rfind('und')
            left = seg[:pos]
            right = seg[pos + 3:]
            unit_val = units.get(left, None)
            ten_val = tens.get(right, None)
            if unit_val is None or ten_val is None:
                return None
            return ten_val + unit_val
        if seg in units:
            return units[seg]
        return None

    def parse_under_thousand(seg: str) -> Optional[int]:
        if seg == '':
            return 0
        if 'hundert' in seg:
            pos = seg.find('hundert')
            left = seg[:pos]
            right = seg[pos + 7:]
            left_val: Optional[int]
            if left == '' or left == 'ein':
                left_val = 1
            else:
                left_val = units.get(left)
            if left_val is None:
                return None
            rest = parse_under_hundred(right)
            if rest is None:
                return None
            return left_val * 100 + rest
        else:
            return parse_under_hundred(seg)

    def parse_with_thousand(seg: str) -> Optional[int]:
        if 'tausend' in seg:
            pos = seg.find('tausend')
            left = seg[:pos]
            right = seg[pos + 7:]
            if left == '':
                left_val = 1
            else:
                left_val = parse_under_thousand(left)
            if left_val is None:
                return None
            right_val = parse_under_thousand(right)
            if right_val is None:
                return None
            return left_val * 1000 + right_val
        else:
            return parse_under_thousand(seg)

    return parse_with_thousand(t)


CH_DIGITS = {
    '零': 0, '〇': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '两': 2, '兩': 2
}
CH_SMALL_UNITS = {'十': 10, '百': 100, '千': 1000}
CH_BIG_UNITS = {
    '万': 10000, '萬': 10000, '亿': 100000000, '億': 100000000, '兆': 1000000000000
}


def chinese_to_int(s: str) -> Optional[int]:
    if not isinstance(s, str) or not s.strip():
        return None
    t = s.strip()
    if not any(ch in CH_DIGITS or ch in CH_SMALL_UNITS or ch in CH_BIG_UNITS for ch in t):
        return None
    total = 0
    section = 0
    number = 0
    for ch in t:
        if ch in CH_DIGITS:
            number = CH_DIGITS[ch]
        elif ch in CH_SMALL_UNITS:
            unit = CH_SMALL_UNITS[ch]
            if number == 0:
                number = 1
            section += number * unit
            number = 0
        elif ch in CH_BIG_UNITS:
            big = CH_BIG_UNITS[ch]
            part = section + number
            if part == 0:
                part = 1
            total += part * big
            section = 0
            number = 0
        else:
            return None
    return total + section + number


def classify_representation(s: str) -> Tuple[str, Optional[int]]:
    if not isinstance(s, str):
        return ("ARABIC", None)
    raw = s.strip()
    if raw == '':
        return ("ARABIC", None)

    if re.fullmatch(r"\d+", raw):
        try:
            return ("ARABIC", int(raw))
        except Exception:
            return ("ARABIC", None)

    r = roman_to_int(raw)
    if r is not None:
        return ("ROMAN", r)

    if any(ch in CH_DIGITS or ch in CH_SMALL_UNITS or ch in CH_BIG_UNITS for ch in raw):
        val = chinese_to_int(raw)
        if val is not None:
            if any(ch in ('萬', '億') for ch in raw):
                rep = "TRAD_CH"
            elif any(ch in ('万', '亿') for ch in raw):
                rep = "SIMP_CH"
            else:
                rep = "TRAD_CH"
            return (rep, val)

    e = parse_english_number(raw)
    if e is not None:
        return ("ENGLISH", e)

    g = parse_german_number(raw)
    if g is not None:
        return ("GERMAN", g)

    try:
        return ("ARABIC", int(raw))
    except Exception:
        return ("ARABIC", None)


