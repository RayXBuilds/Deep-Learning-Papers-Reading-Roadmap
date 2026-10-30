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
    supplemental = ROOT / 'experiments/results/additive-attention.json'
    if not supplemental.exists():
        errors.append('Missing additive attention experiment report')
    else:
        report = json.loads(supplemental.read_text(encoding='utf-8'))
        script = ROOT / 'experiments/additive_attention.py'
        digest = hashlib.sha256(script.read_bytes()).hexdigest()
        if report.get('source_sha256') != digest:
            errors.append('additive-attention.json: source hash mismatch')
        checks = report.get('checks', {})
        if not checks or not all(v is True for v in checks.values()) or report.get('passed') is not True:
            errors.append('additive-attention.json: missing or failed mechanism checks')
    vae_path = ROOT / 'experiments/results/vae-objectives.json'
    if not vae_path.exists():
        errors.append('Missing VAE objective experiment report')
    else:
        report = json.loads(vae_path.read_text(encoding='utf-8'))
        digest = hashlib.sha256((ROOT / 'experiments/vae_objectives.py').read_bytes()).hexdigest()
        if report.get('source_sha256') != digest:
            errors.append('vae-objectives.json: source hash mismatch')
        expected_checks = {
            'elbo_gap_equals_posterior_kl', 'exact_posterior_makes_bound_tight',
            'mc_reconstruction_all_seeds', 'mc_gradients_all_seeds',
            'skip_normalized_density_preserved', 'skip_coupled_sample_preserved',
            'skip_mse_ratio_is_b_squared', 'missing_jacobian_negative_control',
            'fixed_residual_variance_worsens_nll',
        }
        checks = report.get('checks', {})
        if (set(checks) != expected_checks or not all(v is True for v in checks.values())
                or report.get('passed') is not True):
            errors.append('vae-objectives.json: missing or failed mechanism checks')
        if [r.get('seed') for r in report.get('mc_runs', [])] != [7, 42, 2026]:
            errors.append('vae-objectives.json: missing required sampling seeds')
    adamw_path = ROOT / 'experiments/results/adamw-decay.json'
    if not adamw_path.exists():
        errors.append('Missing AdamW decay experiment report')
    else:
        report = json.loads(adamw_path.read_text(encoding='utf-8'))
        digest = hashlib.sha256((ROOT / 'experiments/adamw_decay.py').read_bytes()).hexdigest()
        if report.get('source_sha256') != digest:
            errors.append('adamw-decay.json: source hash mismatch')
        expected_checks = {
            'l2_first_step_matches_hand_calculation', 'adamw_first_step_matches_uniform_shrinkage',
            'l2_and_adamw_are_different', 'adamw_does_not_put_decay_into_moments',
            'scheduled_decay_matches_product', 'plain_sgd_equivalence',
            'all_fixed_budget_trajectories_finite',
        }
        checks = report.get('checks', {})
        if (set(checks) != expected_checks or not all(v is True for v in checks.values())
                or report.get('passed') is not True):
            errors.append('adamw-decay.json: missing or failed mechanism checks')
        trajectories = report.get('trajectories', {})
        if set(trajectories) != {'adam', 'adam_l2', 'adamw'} or any(
                [row.get('step') for row in rows] != list(range(201))
                for rows in trajectories.values()):
            errors.append('adamw-decay.json: missing fixed-budget trajectory steps')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print(f'OK: {len(markdown)} Markdown files, {link_count} local links, '
          f'{len(notes)} structured notes, {len(reports)} baseline reports + 1 additive report + 1 VAE report + 1 AdamW report')
    print('Scope: file targets only; remote URLs and anchor fragments are not checked.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
