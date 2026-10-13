"""Produce phase status ledgers without changing the supplied audit package."""
import csv
import hashlib
import json
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/audit_v2'
PACKAGE = OUT / 'package/PokeAgent3_Final_Audit'
SHA = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
ENV = ('CARGO_HOME=/home/minidc/Desktop/PokeAgent3.0-A/.tools/cargo '
       'RUSTUP_HOME=/home/minidc/Desktop/PokeAgent3.0-A/.tools/rustup '
       'CARGO_TARGET_DIR=/home/minidc/Desktop/PokeAgent3.0-V2/.audit-target ')
CARGO = '/home/minidc/Desktop/PokeAgent3.0-A/.tools/cargo/bin/cargo'
COMMAND = ENV + CARGO + ' test --manifest-path ' + str(ROOT / 'engine/Cargo.toml') + ' --locked --lib public_type_tests'


def ledger(name, key, updates):
    with open(PACKAGE / name, encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames + ['local_status', 'source_sha', 'exact_command', 'evidence', 'remaining_gate']
        rows = list(reader)
    with open(OUT / name, 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            status, command, evidence, remaining = updates.get(row[key], (
                'BLOCKED', '', '', 'Not executed for V2; no PASS claimed.'))
            row.update(local_status=status, source_sha=SHA, exact_command=command,
                       evidence=evidence, remaining_gate=remaining)
            writer.writerow(row)


primitive = ('REPRODUCED', COMMAND, 'P1_PRIMITIVES_AFTER.log',
             'Primitive fix passes; full move-level/public oracle and V2 policy validation pending.')
ledger('02_DEFECT_REGISTER.csv', 'issue_id', {
    'KNOW-01': primitive,
    'KNOW-02': primitive,
    'KNOW-03': ('REPRODUCED', COMMAND, 'KNOW03_BEFORE.log;P1_PRIMITIVES_AFTER.log',
                'Controlled hidden-state paired witness fixed; natural Imprison move-level oracle still required.'),
    'KNOW-04': ('REPRODUCED', COMMAND, 'P1_PRIMITIVES_AFTER.log',
                'View no longer exports stale bench disabled; V2 known/unknown restriction encoding pending.'),
    'DATA-01': ('FIXED', 'python scripts/v2_source_inventory.py --output docs/audit_v2/P0_INVENTORY.json',
                'P0_INVENTORY.json', 'Exact local 1220-loader identity resolved; new V2 rank receipts/profile not implemented.'),
    'OPS-03': ('FIXED', 'python scripts/v2_source_inventory.py --output docs/audit_v2/P0_INVENTORY.json',
               'P0_INVENTORY.json', 'Same-disk archive only; independent Evaluation League remains live.'),
    'ENG-01': ('BLOCKED', ENV + CARGO + ' run --manifest-path engine/Cargo.toml --locked --example coverage_report -- engine/data /home/minidc/Desktop/PokeAgent3.0-A/engine/data/a10-training-teams.json',
               'EXACT_POOL_COVERAGE.log', '0 direct pool blockers; 11 global dynamic closure blockers. Exact-pool dynamic closure and full public oracle certification unresolved.'),
})
ledger('09_TEST_MATRIX.csv', 'test_id', {
    'T-OPS-03': ('FIXED', 'python scripts/v2_source_inventory.py --output docs/audit_v2/P0_INVENTORY.json', 'P0_INVENTORY.json', 'Archived immutable checkpoint/policy hashes verified.'),
    'T-DATA-01': ('BLOCKED', 'python scripts/v2_source_inventory.py --output docs/audit_v2/P0_INVENTORY.json', 'P0_INVENTORY.json', 'Two independent native loaders verified 1220; V2 DDP not executed.'),
    'T-KNOW-01': primitive, 'T-KNOW-02': primitive, 'T-KNOW-03': primitive, 'T-KNOW-04': primitive,
    'T-ENG-01': ('BLOCKED', '', 'EXACT_POOL_COVERAGE.log;P1_ORACLE_RELEASE.log', 'Stored corpus parity is not full exact-pool dynamic coverage certification.'),
})
artifacts = {}
for path in sorted(OUT.glob('*')):
    if path.is_file() and path.name != 'EVIDENCE_MANIFEST.json':
        artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
(OUT / 'EVIDENCE_MANIFEST.json').write_text(json.dumps({
    'tested_runtime_source_sha': SHA,
    'production_started': False,
    'phase_status': {'P0': 'inventory/archive complete', 'P1': 'partial; release blocked', 'P2-P5': 'not started'},
    'artifacts_sha256': artifacts,
}, indent=2) + '\n')
