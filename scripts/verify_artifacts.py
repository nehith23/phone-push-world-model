"""Verify frozen inputs, raw recordings and reported precision aggregates."""
from pathlib import Path
import hashlib
import json
import statistics

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for name in ['frozen_evaluation_protocol.json','precision_protocol.json','precision_frozen_protocol.json']:
        protocol=json.loads((ROOT/'data'/name).read_text())
        for filename,expected in protocol['code_sha256'].items():
            assert digest(ROOT/'scripts'/filename)==expected, f'Frozen code changed: {filename}'
        assert digest(ROOT/'results'/'stabilized'/'plus_pilot'/'small_mlp.pt')==protocol['checkpoint_sha256']
    manifest=json.loads((ROOT/'data'/'raw_manifest.json').read_text())
    for record in manifest['recordings']:
        path=ROOT/'data'/'raw'/record['file']
        assert path.stat().st_size==record['bytes'] and digest(path)==record['sha256'],record['file']
    folder=ROOT/'results'/'precision'/'fresh'
    data=json.loads((folder/'summary.json').read_text())
    for policy in ['geometric','learned']:
        rows=[json.loads((folder/f'{s["id"]}_{policy}.json').read_text()) for s in data['protocol']['evaluation']]
        for scenario,row in zip(data['protocol']['evaluation'],rows):
            assert row['scenario']==scenario and row['checkpoint_sha256']==data['protocol']['checkpoint_sha256']
            assert row['success']==(row['final_error_m']<=.005)
            assert row['tool_contact_steps']>0 and row['robot_body_contact_steps']==0
        summary=data['summary'][policy]
        assert summary['episodes']==len(rows)==24
        assert summary['successes']==sum(r['success'] for r in rows)
        assert abs(summary['mean_final_error_mm']-statistics.mean(r['final_error_m']*1000 for r in rows))<1e-10
        assert abs(summary['median_final_error_mm']-statistics.median(r['final_error_m']*1000 for r in rows))<1e-10
        assert abs(summary['mean_pushes']-statistics.mean(r['pushes'] for r in rows))<1e-10
    print('PASS: frozen code/model hashes, five raw recordings, 48 trial records and precision summary aggregates.')


if __name__=='__main__': main()
