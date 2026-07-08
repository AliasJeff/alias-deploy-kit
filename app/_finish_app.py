#!/usr/bin/env python3
"""Writes the final app.py directly."""
import ast

app_src = """\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import re
import base64
import time
from copy import deepcopy

import requests
import streamlit as st

VLLM_API = "http://127.0.0.1:8000"
PLACEHOLDER_IMG = "https://placehold.co/{w}x{h}/e2e8f0/94a3b8?text=Image"


def check_server():
    try:
        r = requests.get(f"{VLLM_API}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def generate_ui_json(instruction, max_new_tokens=4096,
                     temperature=0.1, top_p=0.9, repetition_penalty=1.1):
    payload = {
        "instruction": instruction,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }
    t0 = time.time()
    resp = requests.post(f"{VLLM_API}/generate", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["output"], time.time() - t0


def replace_invalid_images(html):
    BAD = re.compile(
        r"placeholder|via[.]placeholder|picsum|lorem"
        r"|example[.]com|localhost|127[.]0[.]0[.]1"
        r"|[.][.]/|^/images?/|^/img/|^/assets/",
        re.IGNORECASE,
    )

    def _fix(m):
        tag = m.group(0)
        sm = re.search(r'src="([^"]*)"', tag)
        if not sm:
            return tag
        src = sm.group(1)
        if src and not BAD.search(src):
            return tag
        wm = re.search(r"width=[\"']?(\\d+)", tag)
        hm = re.search(r"height=[\"']?(\\d+)", tag)
        w = wm.group(1) if wm else "120"
        h = hm.group(1) if hm else "120"
        new_src = PLACEHOLDER_IMG.format(w=w, h=h)
        return re.sub(r'src="[^"]*"', 'src="' + new_src + '"', tag)

    return re.sub(r"<img[^>]*>", _fix, html)


def merge_tokens(original, additions):
    tokens = original.split()
    seen = set(tokens)
    s = list(tokens)
    for a in additions:
        if a not in seen:
            s.append(a)
            seen.add(a)
    return " ".join(s)


def compute_beautify_additions(tokens_set):
    add = []
    if "min-h-screen" in tokens_set and "flex" in tokens_set:
        add += ["md:mx-auto", "md:max-w-4xl"]
    if "grid-cols-2" in tokens_set:
        add += ["md:grid-cols-3"]
    if "rounded-lg" in tokens_set and "p-4" in tokens_set:
        add += ["max-w-sm", "w-full", "mx-auto"]
    if any(t.startswith("text-") for t in tokens_set):
        add += ["leading-relaxed"]
    return list(dict.fromkeys(add))


def get_ratio_tag(w, h):
    ratio = w / h
    if 0.9 <= ratio <= 1.4:
        return "ratio-4-3"
    if ratio > 1.4:
        return "ratio-16-9"
    return "ratio-mobile"


def transform_node(node, ratio_tag=None, apply_beautify=True):
    if not isinstance(node, dict):
        return node
    node = deepcopy(node)
    if "className" in node and isinstance(node["className"], str):
        orig = node["className"]
        ts = set(orig.split())
        fc = orig
        if apply_beautify:
            fc = merge_tokens(fc, compute_beautify_additions(ts))
        if ratio_tag:
            fc = merge_tokens(fc, [ratio_tag])
        node["className"] = fc
    if "children" in node and isinstance(node["children"], list):
        node["children"] = [
            transform_node(c, ratio_tag, apply_beautify) for c in node["children"]
        ]
    return node


def json_to_html(node):
    if not isinstance(node, dict):
        return ""
    tag = node.get("name", "div")
    cn = node.get("className", "")
    params = node.get("params", {})
    children = node.get("children", [])
    parts = ['class="' + cn + '"'] if cn else []
    inner = ""
    for k, v in params.items():
        if k == "textContent":
            inner = str(v)
        else:
            parts.append(k + '="' + str(v) + '"')
    attrs = " ".join(parts)
    child_html = "".join(json_to_html(c) for c in children)
    return "<" + tag + " " + attrs + ">" + inner + child_html + "</" + tag + ">"


def wrap_in_template(body, width, height, show_frame):
    fc = (
        "box-shadow:0 25px 50px -12px rgba(0,0,0,0.25);border-radius:20px;"
        if show_frame else ""
    )
    css = (
        "body{background:#f3f4f6;display:flex;justify-content:center;"
        "padding:20px 0;margin:0;}"
        ".phone-canvas{width:" + str(width) + "px;height:" + str(height) + "px;"
        "background:white;overflow-y:auto;position:relative;" + fc + "}"
    )
    return (
        "<!DOCTYPE html>\\n"
        '<html lang="zh-CN"><head>\\n'
        '  <meta charset="UTF-8">\\n'
        '  <script src="https://cdn.tailwindcss.com"></script>\\n'
        "  <style>" + css + "</style>\\n"
        "</head><body>\\n"
        '  <div class="phone-canvas">' + body + "</div>\\n"
        "</body></html>"
    )


def rebuild_html(raw_output, width, height, show_frame, apply_beautify):
    raw_json = json.loads(raw_output.strip())
    ratio_tag = get_ratio_tag(width, height)
    adapted = transform_node(raw_json, ratio_tag=ratio_tag, apply_beautify=apply_beautify)
    body_inner = json_to_html(adapted)
    html = wrap_in_template(body_inner, width, height, show_frame)
    html = replace_invalid_images(html)
    return html, adapted, ratio_tag
"""

# Write and verify part 1
with open('/root/autodl-tmp/app.py', 'w') as f:
    f.write(app_src)
try:
    ast.parse(app_src)
    print('Part1 syntax OK, chars:', len(app_src))
except SyntaxError as e:
    print('SyntaxError part1:', e)
