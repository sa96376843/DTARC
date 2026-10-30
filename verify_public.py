"""Verify unchanged source and aggregate files without optional dependencies."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
records=json.loads((root/'SOURCE_MANIFEST.json').read_text(encoding='utf8'))
errors=[]
for item in records:
    p=root/item['path']
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
        errors.append(item['path'])
if errors: raise SystemExit('Missing or changed: '+', '.join(errors))
print(f'PASS: {len(records)} source/aggregate files match SHA-256. This is not a training reproduction test.')
