from pathlib import Path
import requests,zipfile
p = Path(__file__).resolve().parent / 'data'
p.mkdir(parents=True, exist_ok=True)
target = p / 'stairs.zip'
r=requests.get('https://data.mendeley.com/public-api/zip/6kffmjt7g2/download/1',stream=True,timeout=(20,120)); r.raise_for_status()
print('Downloading',r.headers.get('Content-Length'),flush=True)
n=0
with target.with_suffix('.part').open('wb') as f:
 for chunk in r.iter_content(2**20):
  n+=len(chunk)
  if n>5*2**30: raise RuntimeError('archive exceeds 5 GiB budget')
  f.write(chunk)
  if n//(50*2**20)!=(n-len(chunk))//(50*2**20): print(n,flush=True)
target.with_suffix('.part').rename(target)
with zipfile.ZipFile(target) as z:
 print('Archive',len(z.infolist()),'files',flush=True)
 for item in z.infolist():
  dest=(p/'raw'/item.filename).resolve()
  if not dest.is_relative_to((p/'raw').resolve()): raise RuntimeError('unsafe path')
 z.extractall(p/'raw')
print('DONE',flush=True)
