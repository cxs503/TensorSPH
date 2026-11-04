import json, os, re, urllib.request

user = os.environ.get('TENSOR_AGENT_USER')
pw = os.environ.get('TENSOR_AGENT_PASS')
base = os.environ.get('TENSOR_API_BASE', 'http://127.0.0.1:8001/api/v1')
req = urllib.request.Request(base + '/auth/login',
    data=json.dumps({'username': user, 'password': pw}).encode(),
    headers={'Content-Type': 'application/json'})
tok = json.load(urllib.request.urlopen(req, timeout=10))['access_token']
req = urllib.request.Request(base + '/tensor/reports?limit=25',
    headers={'Authorization': 'Bearer ' + tok})
d = json.load(urllib.request.urlopen(req, timeout=15))
items = d if isinstance(d, list) else d.get('items') or d.get('reports') or []

want = {'3653', '3639', '3675', '3648'}
for r in items:
    if str(r.get('mission_id')) not in want:
        continue
    md = r.get('markdown') or ''
    print('=' * 20, 'm' + str(r.get('mission_id')), (r.get('title') or '')[:70])
    heads = re.findall(r'(?m)^#{1,3} .*$', md)
    print('HEADINGS:', heads)
    print(md[-2600:])
    print()
