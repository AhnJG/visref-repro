"""Common execution framework for diagnostic runs (extracted from diag/sel_think.py).

Provides process_group, run_loop, read_done, and select_pids.
"""
import json
import resource
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'diag'))

from closed_multistep import ChunkError, error_row, read_rows  # noqa: E402


def read_done(out_path, skip_path=''):
    """Return set of pids that completed with status 'ok' from out_path, plus any pids in skip_path."""
    done = {str(r['pid']) for r in read_rows(out_path) if r.get('status') == 'ok'}
    if skip_path:
        p = Path(skip_path)
        if p.exists() and p.stat().st_size:
            done |= {str(x) for x in json.loads(p.read_text())}
    return done


def select_pids(doc, set_name, limit=0, shard='0/1', done=()):
    """The set's order, first `limit`, the shard's share by index (i % n), without the done ones."""
    if set_name not in ('main', 'trial'):
        raise SystemExit(f"--set {set_name} unknown: must be main or trial")
    ids = [str(p) for p in doc[set_name]]
    ids = ids[:limit] if limit else ids
    si, sn = [int(x) for x in shard.split('/')]
    if not (0 <= si < sn):
        raise SystemExit(f"--shard {shard} invalid")
    sharded = [p for i, p in enumerate(ids) if i % sn == si]
    done_set = {str(p) for p in done}
    return [p for p in sharded if p not in done_set]


def process_group(pids, make_item, eng, tok, sp, cfg, generate, version, set_name, run_group, row_out, timing=None):
    """Rows (ok or error) for pids, in order; when the group raises, every question is re-run alone. A ChunkError stops the run."""
    timing = {} if timing is None else timing
    rows, items = {}, []
    t = time.time()
    for pid in pids:
        t0 = time.time()
        try:
            it = make_item(pid)
            it['encode_s'] = round(time.time() - t0, 2)
            items.append(it)
        except ChunkError:
            raise
        except Exception as e:
            rows[pid] = error_row(pid, 'load', f'{type(e).__name__}: {e}', version, set_name)
    timing['encode'] = time.time() - t
    try:
        run_group(items, eng, tok, sp, cfg, generate, timing)
    except ChunkError:
        raise
    except Exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        for it in items:
            try:
                run_group([it], eng, tok, sp, cfg, generate)
            except ChunkError:
                raise
            except Exception as e:
                rows[it['pid']] = error_row(it['pid'], 'generate', f'{type(e).__name__}: {e}', version, set_name)
    for it in items:
        if it['pid'] not in rows:
            rows[it['pid']] = row_out(it, cfg, version, set_name)
    return [rows[p] for p in pids]


def run_loop(todo, out, make_item, eng, tok, sp, cfg, generate, version, set_name, run_group, row_out,
             time_limit=0, t0=None, skip=''):
    """Run todo items in groups of cfg['group'], appending rows to out. Resumes by skipping ok rows."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = read_done(out, skip)
    todo = [p for p in todo if str(p) not in done]
    t0 = time.time() if t0 is None else t0
    peak_all, n_err = 0.0, 0
    with out.open('a') as f:
        for gi in range(0, len(todo), cfg['group']):
            if time_limit and time.time() - t0 > time_limit:
                print(json.dumps({'event': 'time_limit', 'left': len(todo) - gi}), flush=True)
                break
            tg = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            tm = {}
            out_rows = process_group(todo[gi:gi + cfg['group']], make_item, eng, tok, sp, cfg, generate, version, set_name,
                                     run_group, row_out, tm)
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
                if r['status'] != 'ok':
                    n_err += 1
                    print(json.dumps({'event': 'error', 'pid': r['pid'], 'error': r.get('error', '')}), flush=True)
            f.flush()
            peak = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else 0
            peak_all = max(peak_all, peak)
            ev = {'event': 'group', 'n': len(out_rows), 'n_error': sum(r['status'] != 'ok' for r in out_rows)}
            for k in sorted(tm):
                ev[f'{k}_s'] = round(tm[k], 1)
            ev.update({'group_s': round(time.time() - tg, 1), 'elapsed': round(time.time() - t0),
                       'left': len(todo) - gi - len(out_rows), 'peak_alloc_gb': peak,
                       'max_rss_gb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)})
            print(json.dumps(ev), flush=True)
    return peak_all, n_err
