#!/usr/bin/env python3
"""Assemble the Hugging Face Space in deploy/huggingface: copy in the serving code and the saved model, then check it.

    python scripts/build_hf_space.py
    hf upload <user>/<space> deploy/huggingface . --repo-type space --delete "**/__pycache__/**"

The Space needs only what scores an applicant: the Gradio interface, inference, the preprocessing and feature code
(so a request's features are built exactly as in training), and the model files. No data, notebooks or training code.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / 'deploy' / 'huggingface'

FILES = [
    'src/__init__.py',
    'src/app/__init__.py', 'src/app/ui.py', 'src/app/example_request.json',
    'src/serving/__init__.py', 'src/serving/inference.py',
    'src/data/__init__.py', 'src/data/load_data.py', 'src/data/preprocess.py',
    'src/features/__init__.py', 'src/features/build_features.py', 'src/features/bureau.py', 'src/features/previous_application.py',
    'src/features/installments.py', 'src/features/pos_cash.py', 'src/features/credit_card.py',
]
MODEL_FILES = ['model.txt', 'metadata.json', 'input_schema.json', 'typical_applicant.json', 'holdout_evaluation.json']

CHECK = '''
import json, sys
sys.path.insert(0, sys.argv[1])
from src.app.ui import build_ui, score_form, score_json
from src.serving.inference import load_model
model = load_model()                                   # refuses a model.txt that does not match its checksum
summary, reasons = score_form(45, 6, 'F', 'Higher education', 'Cash loans', 180000, 600000, 27000, 500000, 0.5, 0.6, 0.5)
example = open(sys.argv[1] + '/src/app/example_request.json', encoding='utf-8').read()
summary_json, _, response = score_json(example)
build_ui()
print(json.dumps({'features': len(model.features), 'form': summary.splitlines()[0], 'example': response['probability']}))
'''


def main():
    target = SPACE / 'src'
    if target.exists():
        shutil.rmtree(target)
    for f in FILES:
        (SPACE / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, SPACE / f)
    (target / 'serving' / 'model').mkdir(parents=True)
    for f in MODEL_FILES:
        shutil.copy2(ROOT / 'src' / 'serving' / 'model' / f, target / 'serving' / 'model' / f)

    # the copy is byte-identical, and the Space's code runs from the Space folder alone
    meta = json.loads((target / 'serving' / 'model' / 'metadata.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((target / 'serving' / 'model' / 'model.txt').read_bytes()).hexdigest() == meta['files']['model_sha256']
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}        # leave no __pycache__ in the Space
    result = subprocess.run([sys.executable, '-c', CHECK, str(SPACE)], capture_output=True, text=True, cwd=SPACE, env=env)
    for cache in SPACE.rglob('__pycache__'):
        shutil.rmtree(cache)
    if result.returncode != 0:
        raise SystemExit(f'the Space does not run on its own:\n{result.stderr[-3000:]}')
    check = json.loads(result.stdout.strip().splitlines()[-1])

    files = sorted(p.relative_to(SPACE).as_posix() for p in SPACE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    size = sum((SPACE / f).stat().st_size for f in files) / 1e6
    print(f'{SPACE.relative_to(ROOT).as_posix()}: {len(files)} files, {size:.1f} MB')
    print(f"checked: model loads ({check['features']} features), form -> {check['form']}, example request -> {check['example']}")


if __name__ == '__main__':
    main()
