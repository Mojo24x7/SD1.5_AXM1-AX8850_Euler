#!/usr/bin/env python3
import os
import re
import json
import argparse
from functools import lru_cache
from typing import List, Dict, Any, Tuple

# We use the third-party "regex" module if available (matches CLIP/OpenAI tokenization better)
try:
    import regex as reg  # pip install regex
except Exception:
    reg = None

# Optional text cleanup (like transformers CLIPTokenizer does)
try:
    import ftfy  # pip install ftfy
except Exception:
    ftfy = None


@lru_cache(maxsize=1)
def _bytes_to_unicode():
    """
    Returns mapping for byte-level BPE (OpenAI / CLIP).
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def _get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for ch in word[1:]:
        pairs.add((prev_char, ch))
        prev_char = ch
    return pairs


def _load_bpe_merges(merges_path: str):
    merges = {}
    with open(merges_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    # First line is a header in OpenAI merges.txt
    for i, line in enumerate(lines[1:]):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        merges[tuple(parts)] = i
    return merges


def _bpe(token: str, bpe_ranks: dict):
    word = tuple(token)
    pairs = _get_pairs(word)

    if not pairs:
        return token

    while True:
        bigram = min(pairs, key=lambda p: bpe_ranks.get(p, 10**10))
        if bigram not in bpe_ranks:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except ValueError:
                new_word.extend(word[i:])
                break

            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                new_word.append(first + second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        word = tuple(new_word)
        if len(word) == 1:
            break
        pairs = _get_pairs(word)

    return " ".join(word)


def _basic_clean(text: str) -> str:
    text = text or ""
    text = text.strip()
    if ftfy is not None:
        text = ftfy.fix_text(text)
    # transformers CLIPTokenizer lowercases by default for SD1.5 tokenizer
    return text.lower()


def _clip_regex_findall(text: str):
    # OpenAI CLIP tokenization pattern; requires "regex" for \p{L}\p{N}.
    pat = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    return reg.findall(pat, text)


def _fallback_split(text: str):
    # If regex module isn't installed, use a simpler (less exact) fallback.
    return re.findall(r"\S+|\s+", text)


@lru_cache(maxsize=8)
def _load_vocab_and_merges(tokenizer_dir: str):
    vocab_path = os.path.join(tokenizer_dir, "vocab.json")
    merges_path = os.path.join(tokenizer_dir, "merges.txt")

    if not (os.path.isfile(vocab_path) and os.path.isfile(merges_path)):
        raise FileNotFoundError(f"Missing vocab.json/merges.txt in {tokenizer_dir}")

    with open(vocab_path, "r", encoding="utf-8") as f:
        encoder = json.load(f)

    bpe_ranks = _load_bpe_merges(merges_path)
    byte_encoder = _bytes_to_unicode()
    return encoder, bpe_ranks, byte_encoder


def clip_bpe_pieces(tokenizer_dir: str, text: str) -> List[str]:
    """
    Return CLIP BPE pieces (WITHOUT BOS/EOS).
    """
    _, bpe_ranks, byte_encoder = _load_vocab_and_merges(tokenizer_dir)

    text = _basic_clean(text)

    if reg is not None:
        tokens = _clip_regex_findall(text)
    else:
        tokens = _fallback_split(text)

    pieces: List[str] = []
    for tok in tokens:
        tok_bytes = tok.encode("utf-8")
        tok_trans = "".join(byte_encoder[b] for b in tok_bytes)
        bpe_out = _bpe(tok_trans, bpe_ranks).split(" ")
        for bpe_tok in bpe_out:
            pieces.append(bpe_tok)
    return pieces


def count_clip_bpe_tokens(tokenizer_dir: str, text: str) -> int:
    """
    Total tokens INCLUDING BOS/EOS inside the 77 limit.
    """
    pieces = clip_bpe_pieces(tokenizer_dir, text)
    return 1 + len(pieces) + 1  # BOS + pieces + EOS


def _split_words_keep_punct(s: str) -> List[str]:
    """
    UI-friendly "words": keeps punctuation as part of token so it’s visible.
    """
    s = s.strip()
    if not s:
        return []
    # split by whitespace but keep punctuation glued to word
    return re.findall(r"\S+", s)


def analyze_word_cost(tokenizer_dir: str, text: str) -> Dict[str, Any]:
    """
    Approximate per-word token cost by incremental tokenization:
      cost(word_i) = tokens(prefix_with_word_i) - tokens(prefix_without_word_i)
    BOS/EOS included in totals; per-word costs sum (roughly) to total-2.
    """
    words = _split_words_keep_punct(text)
    rows = []
    prefix = ""

    # tokens for empty prefix includes BOS+EOS = 2
    prev_total = count_clip_bpe_tokens(tokenizer_dir, prefix)

    for w in words:
        new_prefix = (prefix + " " + w).strip()
        cur_total = count_clip_bpe_tokens(tokenizer_dir, new_prefix)
        delta = cur_total - prev_total
        rows.append({"word": w, "cost": int(delta), "cum": int(cur_total)})
        prefix = new_prefix
        prev_total = cur_total

    total = count_clip_bpe_tokens(tokenizer_dir, text)
    return {"ok": True, "text": text, "total_tokens": total, "words": rows}


def trim_to_max_len(tokenizer_dir: str, text: str, max_len: int) -> Dict[str, Any]:
    """
    Deterministic "CLIP-safe trim" without an LLM:
    - Prefer trimming by comma-separated phrases (keeps meaning better)
    - Otherwise trim by words from the end
    - Ensures total tokens (BOS+...+EOS) <= max_len
    """
    text0 = (text or "").strip()
    if not text0:
        return {"ok": True, "trimmed": "", "changed": False, "dropped": [], "tokens": 2, "max": max_len}

    cur = text0
    cur_tokens = count_clip_bpe_tokens(tokenizer_dir, cur)
    if cur_tokens <= max_len:
        return {"ok": True, "trimmed": cur, "changed": False, "dropped": [], "tokens": cur_tokens, "max": max_len}

    # Strategy A: phrase trim by commas
    parts = [p.strip() for p in re.split(r"\s*,\s*", cur) if p.strip()]
    kept: List[str] = []
    dropped: List[str] = []

    if len(parts) >= 2:
        for p in parts:
            candidate = ", ".join(kept + [p]).strip()
            if not candidate:
                continue
            if count_clip_bpe_tokens(tokenizer_dir, candidate) <= max_len:
                kept.append(p)
            else:
                dropped.append(p)

        if kept:
            out = ", ".join(kept).strip()
            out_tokens = count_clip_bpe_tokens(tokenizer_dir, out)
            if out_tokens <= max_len:
                return {
                    "ok": True, "trimmed": out, "changed": True,
                    "dropped": dropped, "tokens": out_tokens, "max": max_len
                }

    # Strategy B: word trim from the end
    words = _split_words_keep_punct(cur)
    kept_words: List[str] = []
    for w in words:
        candidate = (" ".join(kept_words + [w])).strip()
        if count_clip_bpe_tokens(tokenizer_dir, candidate) <= max_len:
            kept_words.append(w)
        else:
            break

    out = " ".join(kept_words).strip()
    # If even first word overflows (unlikely), hard-empty
    if not out:
        out = ""
    out_tokens = count_clip_bpe_tokens(tokenizer_dir, out)

    dropped_words = words[len(kept_words):]
    return {
        "ok": True, "trimmed": out, "changed": True,
        "dropped": dropped_words, "tokens": out_tokens, "max": max_len
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_dir", required=True)

    ap.add_argument("--mode", default="count", choices=["count", "analyze", "trim"])
    ap.add_argument("--prompt", default="")
    ap.add_argument("--negative", default="")
    ap.add_argument("--text", default="")

    ap.add_argument("--max_len", type=int, default=77)
    ap.add_argument("--sum", action="store_true", help="Also return prompt+negative sum (count mode)")

    args = ap.parse_args()
    tok_dir = os.path.expanduser(args.tokenizer_dir)
    max_len = int(args.max_len)

    if args.mode == "count":
        p = count_clip_bpe_tokens(tok_dir, args.prompt)
        n = count_clip_bpe_tokens(tok_dir, args.negative)
        out = {
            "ok": True,
            "prompt": {"tokens": p, "max": max_len, "over": p > max_len},
            "negative": {"tokens": n, "max": max_len, "over": n > max_len},
        }
        if args.sum:
            out["total"] = {"tokens": p + n}
        print(json.dumps(out))
        return

    if args.mode == "analyze":
        text = args.text if args.text != "" else args.prompt
        out = analyze_word_cost(tok_dir, text)
        out["max"] = max_len
        out["over"] = out.get("total_tokens", 0) > max_len
        print(json.dumps(out))
        return

    if args.mode == "trim":
        text = args.text if args.text != "" else args.prompt
        out = trim_to_max_len(tok_dir, text, max_len)
        print(json.dumps(out))
        return


if __name__ == "__main__":
    main()

