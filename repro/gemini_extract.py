"""Run the MathVista answer-extraction requests (from llm_extract.py prepare) through the Gemini API.

Same call as the official extractor (references/sources/mathvista_official_r4/models/gpt.py): one text-only
user message = the official extraction prompt, temperature 0, max output 1024 tokens; the model's text is the
extraction. Standard library only. Resumable: ids already in --out are skipped.

  export GEMINI_API_KEY=...            # Google AI Studio key
  python3 repro/gemini_extract.py REQUESTS.jsonl --out RESPONSES.jsonl --model <AI Studio model id>

Then (in the Claude session): python3 repro/llm_extract.py merge ... --responses RESPONSES.jsonl
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

URL = 'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'


def response_text(resp):
    """Concatenate the answer text parts of the first candidate (thought parts, if any, are skipped)."""
    cands = resp.get('candidates') or []
    if not cands:
        return None, (resp.get('promptFeedback') or {}).get('blockReason', 'no_candidates')
    parts = (cands[0].get('content') or {}).get('parts') or []
    text = ''.join(p.get('text', '') for p in parts if not p.get('thought'))
    return text.strip(), cands[0].get('finishReason')


def call(model, key, prompt, retries=6):
    body = json.dumps({'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
                       'generationConfig': {'temperature': 0, 'maxOutputTokens': 1024}}).encode()
    req = urllib.request.Request(URL.format(model=model), data=body, method='POST',
                                 headers={'Content-Type': 'application/json', 'x-goog-api-key': key})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return response_text(json.loads(r.read()))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and i < retries - 1:
                time.sleep(min(60, 2 ** i * 2))
                continue
            raise SystemExit(f'HTTP {e.code}: {e.read().decode()[:500]}')
        except urllib.error.URLError:
            if i < retries - 1:
                time.sleep(min(60, 2 ** i * 2))
                continue
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('requests')
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', required=True, help='Gemini model id as listed in Google AI Studio')
    ap.add_argument('--sleep', type=float, default=0.0, help='seconds between calls (free-tier rate limits)')
    a = ap.parse_args()
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        sys.exit('set GEMINI_API_KEY')
    reqs = [json.loads(l) for l in open(a.requests) if l.strip()]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)['id'] for l in open(a.out) if l.strip()}
    todo = [r for r in reqs if r['id'] not in done]
    print(f'{len(reqs)} requests, {len(done)} done, {len(todo)} to run', flush=True)
    with open(a.out, 'a') as f:
        for n, r in enumerate(todo, 1):
            text, finish = call(a.model, key, r['prompt'])
            f.write(json.dumps({'id': r['id'], 'extraction': text or '', 'finish': finish, 'model': a.model},
                               ensure_ascii=False) + '\n')
            f.flush()
            print(f'[{n}/{len(todo)}] {r["id"]} -> {text!r} ({finish})', flush=True)
            if a.sleep:
                time.sleep(a.sleep)


if __name__ == '__main__':
    main()
