"""Read emails as inert data. No links, attachments, scripts or network calls."""
import hashlib
import codecs
import copy
import re
import tarfile
import unicodedata
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import PurePosixPath

if not __debug__:
    raise RuntimeError("Run without -O: reproducibility and data guards must remain enabled")

MAX_BYTES = 2_000_000
MAX_CHARS = 20_000
PARSER_VERSION = "2.0.0-mime-repair"
HEADER_LINE = re.compile(r"(?im)^(?:x-[\w-]+|received|return-path|delivered-to|message-id|authentication-results|dkim-signature|from|to|cc|bcc|date|mime-version|content-type|content-transfer-encoding):[^\n]*(?:\n[ \t]+[^\n]*)*")
URL = re.compile(r"(?i)(?:https?://|www\.)[^\s<>\"']+")
ADDRESS = re.compile(r"[\w.+%-]+@[\w.-]+\.[A-Za-z]{2,}")


class VisibleHTML(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in ("script", "style", "noscript"):
            self.hidden += 1
        if not self.hidden:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag.lower() in ("script", "style", "noscript") and self.hidden:
            self.hidden -= 1
        if not self.hidden:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def visible_html(value):
    p = VisibleHTML()
    p.feed(value[:1_000_000])
    return " ".join(p.parts)


def clean_text(value, html=False):
    value = str(value or "")[:1_000_000]
    value = HEADER_LINE.sub(" ", value)
    if html or re.search(r"(?i)<(?:html|body|div|p|table|br|a)\b", value):
        value = visible_html(value)
    value = unicodedata.normalize("NFKC", value)
    value = URL.sub(" URLTOKEN ", value)
    value = ADDRESS.sub(" EMAILTOKEN ", value)
    return re.sub(r"\s+", " ", value).strip()[:MAX_CHARS]


def feature_text(subject, body, html=False):
    return (clean_text(subject)[:600] + "\n" + clean_text(body, html))[:MAX_CHARS]


def decode_part(part):
    payload = part.get_payload(decode=True)
    if payload is None:
        payload = str(part.get_payload() or "").encode("utf-8", errors="replace")
    declared = part.get_content_charset()
    for charset in [declared, "utf-8", "windows-1252"]:
        if not charset:
            continue
        try:
            return payload.decode(charset)
        except (UnicodeError, LookupError):
            pass
    return payload.decode("utf-8", errors="replace")


def parse_email(raw):
    """Render eligible body parts without using labels, paths or model scores.

    Shared clean_text/feature_text/char_preprocess retain baseline semantics.
    The only content changes are MIME selection and a narrowly recognized
    malformed Content-Type repair. Classifier-generated tags are deliberately
    left under the unchanged feature policy to preserve training compatibility.
    """
    if len(raw) > MAX_BYTES:
        raise ValueError("Email exceeds 2 MB limit")
    msg = BytesParser(policy=policy.default).parsebytes(raw)
    stats = {"plain_parts": 0, "html_parts": 0, "mime_repairs": 0,
             "empty_plain_html_fallbacks": 0, "charset_fallback_parts": 0,
             "unsupported_parts": 0}
    visited = []
    selected_codecs = set()
    repair_pattern = re.compile(
        r"^\s*(text/(?:plain|html))\s+content-transfer-encoding\s*:\s*"
        r"(7bit|8bit|binary|base64|quoted-printable)(?:\\r\\n|\s)*$",
        re.IGNORECASE,
    )

    def excluded(part):
        return (part.get_content_disposition() == "attachment"
                or part.get_filename() is not None
                or part.get_content_maintype() == "message")

    def decode_with_diagnostics(part):
        payload = part.get_payload(decode=True)
        if payload is None:
            payload = str(part.get_payload() or "").encode("utf-8", errors="replace")
        declared = part.get_content_charset()
        for index, charset in enumerate([declared, "utf-8", "windows-1252"]):
            if not charset:
                continue
            try:
                text = payload.decode(charset)
                selected_codecs.add(codecs.lookup(charset).name)
                stats["charset_fallback_parts"] += int(index != 0)
                return text
            except (UnicodeError, LookupError):
                pass
        stats["charset_fallback_parts"] += 1
        selected_codecs.add("utf-8-replace")
        return payload.decode("utf-8", errors="replace")

    def render(part):
        if excluded(part):
            return "", "excluded"
        visited.append(part)
        kind = part.get_content_type()
        if part.is_multipart():
            children = part.get_payload()
            if not children:
                return "", "empty"
            if kind == "multipart/related":
                # RFC related resources do not become body features.
                start = part.get_param("start")
                if start:
                    matching = [c for c in children if str(c.get("Content-ID", "")).strip() == str(start).strip()]
                    child = matching[0] if matching else children[0]
                else:
                    child = children[0] if children else None
                return render(child) if child is not None else ("", "empty")
            rendered = [render(child) for child in children]
            if kind == "multipart/alternative":
                plain = [text for text, mode in rendered if mode == "plain" and text]
                if plain:
                    return plain[0], "plain"
                useful = [(text, mode) for text, mode in rendered if text]
                if useful:
                    chosen = useful[-1]
                    if chosen[1] == "html" and any(mode == "plain" for _, mode in rendered):
                        stats["empty_plain_html_fallbacks"] += 1
                    return chosen
                return "", "empty"
            # Mixed/signed bodies retain their eligible text children in order.
            useful = [(text, mode) for text, mode in rendered if text]
            if len(useful) == 1:
                return useful[0]
            return " ".join(text for text, _ in useful), "mixed"
        decoding_part = part
        if kind not in ("text/plain", "text/html"):
            # Exact known malformed syntax only. No raw-message or multipart
            # fallback, and attachment exclusion has already happened.
            raw_type = next((value for name, value in part.raw_items()
                             if name.lower() == "content-type"), "")
            match = repair_pattern.fullmatch(raw_type)
            if not match:
                stats["unsupported_parts"] += 1
                return "", "unsupported"
            kind, transfer = match.groups()
            kind = kind.lower()
            decoding_part = copy.deepcopy(part)
            decoding_part.replace_header("Content-Type", kind)
            if decoding_part.get("Content-Transfer-Encoding") is None:
                decoding_part["Content-Transfer-Encoding"] = transfer.lower()
            stats["mime_repairs"] += 1
        mode = "plain" if kind == "text/plain" else "html"
        stats[mode + "_parts"] += 1
        text = decode_with_diagnostics(decoding_part)
        return clean_text(text, html=(mode == "html")), mode

    body, selected_mode = render(msg)
    subject = clean_text(str(msg.get("Subject", "")))[:600]
    payload_defects = [type(d).__name__ for p in visited for d in p.defects]
    header_defects = []
    for p in visited:
        for name in ("Subject", "Content-Type", "Content-Transfer-Encoding", "Content-Disposition", "MIME-Version"):
            try:
                header_defects.extend(type(d).__name__ for d in getattr(p.get(name), "defects", ()))
            except Exception as exc:
                # Header error text can include personal data: retain type only.
                header_defects.append("HeaderAccessException:" + type(exc).__name__)
    return {
        "text": (subject + "\n" + body)[:MAX_CHARS],
        "subject": subject,
        **stats,
        "parser_version": PARSER_VERSION,
        "selected_body_mode": selected_mode,
        "charset_codecs_used": sorted(selected_codecs),
        "root_parse_defects": len(msg.defects),
        "payload_defects": payload_defects,
        "header_defects": header_defects,
        "parse_defects": len(payload_defects) + len(header_defects),
        "body_missing": not bool(body),
        "body_empty_after_normalization": not bool(body),
    }


def char_preprocess(value):
    """Bound character vocabulary construction cost before fitting a fold."""
    return value[:4000].lower()


def read_test_archive(path, with_labels=False):
    rows = []
    total = 0
    with tarfile.open(path, "r|xz") as archive:
        for member in archive:
            p = PurePosixPath(member.name)
            if p.is_absolute() or ".." in p.parts or member.issym() or member.islnk():
                raise ValueError("Unsafe archive member")
            if member.isdir():
                continue
            if not member.isfile() or len(p.parts) != 2 or p.parts[0] not in ("ham", "spam") or p.suffix != ".eml":
                raise ValueError("Unexpected test member")
            total += member.size
            if total > 100_000_000 or member.size > MAX_BYTES or len(rows) >= 5000:
                raise ValueError("Archive exceeds limits")
            raw = archive.extractfile(member).read()
            row = parse_email(raw)
            row.update(id=member.name, raw_sha256=hashlib.sha256(raw).hexdigest())
            if with_labels:
                row["label"] = int(p.parts[0] == "spam")
            rows.append(row)
    return sorted(rows, key=lambda x: x["id"])


def canonical(text):
    value = unicodedata.normalize("NFKC", text).lower()
    value = re.sub(r"\d+", " numbertoken ", value)
    return " ".join(re.findall(r"\w+", value))


def subject_group(subject):
    s = re.sub(r"(?i)^(?:(?:re|fw|fwd)\s*:\s*)+", "", subject)
    s = canonical(s)
    return s if len(s) >= 12 else ""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
