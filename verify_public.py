from pathlib import Path
import json,hashlib
r=Path(__file__).resolve().parent
items=json.loads((r/'SOURCE_MANIFEST.json').read_text(encoding='utf8'))
bad=[x['path'] for x in items if not (r/x['path']).is_file() or hashlib.sha256((r/x['path']).read_bytes()).hexdigest()!=x['sha256']]
if bad:raise SystemExit('Missing or changed: '+', '.join(bad))
print(f'PASS: {len(items)} public files match SHA-256. This check is not a training reproduction.')
