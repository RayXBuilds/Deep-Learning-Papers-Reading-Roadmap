#!/usr/bin/env python3
"""Check Markdown file links, note structure, and recorded experiment provenance.

Local anchor fragments and remote link availability are deliberately not checked.
Git-tracked sparse files count as existing so legacy PDFs need not be downloaded.
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def main():
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    known = {ROOT / p for p in tracked if p}
    for p in list(known):
        known.update(p.parents)
    markdown = sorted(p for p in ROOT.rglob('*.md') if '.git' not in p.parts)
    errors, link_count = [], 0
    for path in markdown:
        content = path.read_text(encoding='utf-8')
        # Fenced examples are not rendered links. Repo links use inline Markdown.
        visible = re.sub(r'```.*?```', '', content, flags=re.S)
        for raw in re.findall(r'!?\[[^\]]*\]\(([^\n]+?)\)', visible):
            target = raw.strip().strip('<>')
            parts = urlsplit(target)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            link_count += 1
            resolved = (path.parent / unquote(parts.path)).resolve()
            if not resolved.is_relative_to(ROOT):
                errors.append(f'{path.relative_to(ROOT)}: link escapes repository: {target}')
            elif not resolved.exists() and resolved not in known:
                errors.append(f'{path.relative_to(ROOT)}: missing link target: {target}')
    required = ['## 问题', '## 方法与关键推导', '## 原论文证据',
                '## 与其他论文的关系', '## 可验证实验', '## 局限与思考']
    notes = list((ROOT / 'notes').glob('*.md'))
    for note in notes:
        text = note.read_text(encoding='utf-8')
        for section in required:
            if section not in text:
                errors.append(f'{note.name}: missing section {section}')
        if 'https://arxiv.org/' not in text:
            errors.append(f'{note.name}: missing primary paper source')
    source_hash = hashlib.sha256((ROOT / 'experiments/run.py').read_bytes()).hexdigest()
    reports = sorted((ROOT / 'experiments/results').glob('baseline-seed-*.json'))
    if not reports:
        errors.append('No recorded baseline experiment reports')
    for path in reports:
        report = json.loads(path.read_text(encoding='utf-8'))
        if report.get('source_sha256') != source_hash:
            errors.append(f'{path.name}: source changed; rerun and explain the new result')
        expected = {'residual', 'normalization', 'adam', 'attention', 'diffusion'}
        results = report.get('results', {})
        if set(results) != expected:
            errors.append(f'{path.name}: missing or unexpected experiment names')
        if report.get('passed') is not True:
            errors.append(f'{path.name}: baseline contains a failed experiment')
        for name, result in results.items():
            checks = result.get('checks', {})
            if not checks or not all(v is True for v in checks.values()) or result.get('passed') is not True:
                errors.append(f'{path.name}: inconsistent checks for {name}')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print(f'OK: {len(markdown)} Markdown files, {link_count} local links, '
          f'{len(notes)} structured notes, {len(reports)} matching baseline reports')
    print('Scope: file targets only; remote URLs and anchor fragments are not checked.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
