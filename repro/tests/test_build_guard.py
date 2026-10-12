"""Builders must not overwrite an existing run record (2026-10-12): published sessions are historical records."""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'kaggle'))
import build  # noqa: E402


def test_new_run_dir_creates_fresh_folder(tmp_path):
    out = build.new_run_dir(tmp_path, 'rerun_s1')
    assert out == tmp_path / 'runs' / 'rerun_s1' and out.is_dir()


@pytest.mark.parametrize('marker', ['run_config.json', 'notebook.ipynb', 'output'])
def test_new_run_dir_refuses_existing_record(tmp_path, marker):
    run = tmp_path / 'runs' / 'b_main_s1'
    run.mkdir(parents=True)
    (run / marker).mkdir() if marker == 'output' else (run / marker).write_text('{}')
    with pytest.raises(SystemExit):
        build.new_run_dir(tmp_path, 'b_main_s1')
    assert build.new_run_dir(tmp_path, 'b_main_s1', overwrite=True) == run


def test_existing_published_runs_are_protected():
    runs = pathlib.Path(__file__).resolve().parent.parent / 'kaggle' / 'runs'
    for name in ('b_main_s1', 'step1_branch_pilot', 'step1_branch_main'):
        if (runs / name / 'run_config.json').exists():
            with pytest.raises(SystemExit):
                build.new_run_dir(runs.parent, name)
