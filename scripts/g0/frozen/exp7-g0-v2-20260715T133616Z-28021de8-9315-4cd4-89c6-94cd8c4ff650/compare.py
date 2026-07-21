#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json

def file_sha256(path):
    h=hashlib.sha256();
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def read_jsonl(path):
    raw=open(path,'rb').read(); rec=[]; seen=set()
    for i,line in enumerate(raw.splitlines(),1):
        if not line: raise RuntimeError(f'empty line {i}')
        r=json.loads(line); rid=r.get('record_id')
        if not isinstance(rid,str) or rid in seen: raise RuntimeError('missing/duplicate record_id')
        seen.add(rid); rec.append(r)
    return rec,raw
def read_json(path):
    raw=open(path,'rb').read(); return json.loads(raw),raw
def compare(source_pre,source_post,paper_pre,paper_post,bundle):
    pre,pre_raw=read_jsonl(source_pre); post,post_raw=read_jsonl(source_post); pp,ppr=read_json(paper_pre); po,por=read_json(paper_post)
    failures=[]
    if pre_raw!=post_raw: failures.append('source_manifest_bytes_differ')
    if pp!=po: failures.append('paper_manifest_differs')
    for lbl,p in [('pre',pp),('post',po)]:
        if p.get('bundle_lock_sha256')!=bundle: failures.append(lbl+'_bundle_mismatch')
    return {'status':'PASS' if not failures else 'FAIL','failures':failures,'source_record_count':len(pre),'source_pre_sha256':hashlib.sha256(pre_raw).hexdigest(),'source_post_sha256':hashlib.sha256(post_raw).hexdigest(),'paper_pre_sha256':hashlib.sha256(ppr).hexdigest(),'paper_post_sha256':hashlib.sha256(por).hexdigest(),'paper_sha256':pp.get('sha256'),'bundle_lock_sha256':bundle}
