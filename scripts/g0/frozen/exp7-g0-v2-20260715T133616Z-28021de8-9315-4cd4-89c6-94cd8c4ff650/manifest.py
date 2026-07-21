#!/usr/bin/env python3
from __future__ import annotations
import base64, hashlib, json, os, stat
from pathlib import PurePosixPath
CHUNK=8*1024*1024
SCHEMA='experiments7-source-pre-record/v2'
PAPER_SCHEMA='experiments7-paper-pre/v2'
def canonical_json(v): return (json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
def sha256_bytes(b:bytes)->str: return hashlib.sha256(b).hexdigest()
def kind(m:int)->str:
    return 'regular' if stat.S_ISREG(m) else 'directory' if stat.S_ISDIR(m) else 'symlink' if stat.S_ISLNK(m) else 'fifo' if stat.S_ISFIFO(m) else 'socket' if stat.S_ISSOCK(m) else 'character_device' if stat.S_ISCHR(m) else 'block_device' if stat.S_ISBLK(m) else 'unknown'
def meta(st): return {'type':kind(st.st_mode),'size':st.st_size,'mode':stat.S_IMODE(st.st_mode),'uid':st.st_uid,'gid':st.st_gid,'nlink':st.st_nlink,'mtime_ns':st.st_mtime_ns,'ctime_ns':st.st_ctime_ns,'dev':st.st_dev,'inode':st.st_ino}
def stable(st):
    m=meta(st); return tuple(m[k] for k in sorted(m))
def read_hash(fd):
    h=hashlib.sha256(); n=0
    while True:
        b=os.read(fd,CHUNK)
        if not b: break
        h.update(b); n+=len(b)
    return h.hexdigest(),n
def record_id(root,path): return hashlib.sha256((root+'\0'+path).encode()).hexdigest()
def rec(root,path,st):
    return {'schema':SCHEMA,'record_id':record_id(root,path),'root':root,'path':path, **meta(st)}
def _scan(root_id, dfd, rel, records):
    before=os.fstat(dfd)
    names=sorted(os.listdir(dfd), key=os.fsencode)
    for name in names:
        child=name if not rel else rel+'/'+name
        first=os.stat(name,dir_fd=dfd,follow_symlinks=False)
        r=rec(root_id,child,first); t=r['type']
        if t=='regular':
            fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=dfd)
            try:
                opened=os.fstat(fd)
                if stable(first)!=stable(opened): raise RuntimeError(f'regular substitution {root_id}:{child}')
                dg,total=read_hash(fd); final=os.fstat(fd)
                if stable(opened)!=stable(final) or total!=final.st_size: raise RuntimeError(f'regular changed {root_id}:{child}')
                r['sha256']=dg
            finally: os.close(fd)
        elif t=='directory':
            fd=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=dfd)
            try:
                opened=os.fstat(fd)
                if stable(first)!=stable(opened): raise RuntimeError(f'dir substitution {root_id}:{child}')
                _scan(root_id,fd,child,records)
                if stable(opened)!=stable(os.fstat(fd)): raise RuntimeError(f'dir changed {root_id}:{child}')
            finally: os.close(fd)
        elif t=='symlink':
            r['link_target_b64']=base64.b64encode(os.fsencode(os.readlink(name,dir_fd=dfd))).decode('ascii')
            if stable(first)!=stable(os.stat(name,dir_fd=dfd,follow_symlinks=False)): raise RuntimeError(f'symlink changed {root_id}:{child}')
        records.append(r)
    if stable(before)!=stable(os.fstat(dfd)): raise RuntimeError(f'directory changed {root_id}:{rel}')
def build_source_records(config):
    if not os.environ.get('EXPERIMENTS7_ENVELOPE_ID'): raise RuntimeError('missing envelope id')
    records=[]; seen=set()
    for root in sorted(config['protected_sources'], key=lambda r:r['root']):
        rid=root['root']; path=root['path']
        if rid in seen: raise RuntimeError('duplicate root')
        seen.add(rid); fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
        try:
            st=os.fstat(fd); b=root['binding']
            if (st.st_dev,st.st_ino)!=(b['dev'],b['inode']): raise RuntimeError(f'root binding mismatch {rid}')
            records.append(rec(rid,'',st)); _scan(rid,fd,'',records)
            if stable(st)!=stable(os.fstat(fd)): raise RuntimeError(f'root changed {rid}')
        finally: os.close(fd)
    records.sort(key=lambda r:(r['root'].encode(), r['path'].encode()))
    ids=[r['record_id'] for r in records]
    if len(ids)!=len(set(ids)): raise RuntimeError('duplicate source record id')
    return records
def _write_all(fd,data):
    view=memoryview(data)
    while view:
        n=os.write(fd,view)
        if n<=0: raise OSError('short write')
        view=view[n:]
def atomic_publish(path,chunks):
    path=os.path.abspath(path); directory=os.path.dirname(path); base=os.path.basename(path)
    dfd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC); tmp=f'.{base}.tmp.{os.getpid()}'
    fd=-1
    try:
        fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o644,dir_fd=dfd)
        h=hashlib.sha256(); size=0
        for c in chunks:
            _write_all(fd,c); h.update(c); size+=len(c)
        os.fsync(fd); tst=os.fstat(fd); os.close(fd); fd=-1
        os.link(tmp,base,src_dir_fd=dfd,dst_dir_fd=dfd,follow_symlinks=False); os.unlink(tmp,dir_fd=dfd); os.fsync(dfd)
        fst=os.stat(base,dir_fd=dfd,follow_symlinks=False)
        if (fst.st_dev,fst.st_ino)!=(tst.st_dev,tst.st_ino): raise RuntimeError('publication identity mismatch')
        return {'path':path,'sha256':h.hexdigest(),'size':size,'dev':fst.st_dev,'inode':fst.st_ino}
    except Exception:
        if fd>=0: os.close(fd)
        try: os.unlink(tmp,dir_fd=dfd)
        except OSError: pass
        raise
    finally: os.close(dfd)
def publish_source_manifest(config, output):
    recs=build_source_records(config); res=atomic_publish(output,(canonical_json(r) for r in recs));
    res.update({'record_count':len(recs),'regular_count':sum(1 for r in recs if r['type']=='regular'),'directory_count':sum(1 for r in recs if r['type']=='directory'),'symlink_count':sum(1 for r in recs if r['type']=='symlink')}); return res
def namespace_entries(dfd):
    out=[]
    for name in sorted(os.listdir(dfd), key=os.fsencode):
        st=os.stat(name,dir_fd=dfd,follow_symlinks=False)
        out.append({'name_b64':base64.b64encode(os.fsencode(name)).decode('ascii'), **meta(st)})
    return out
def pdf_pages(path):
    try:
        from pypdf import PdfReader
        return len(PdfReader(path).pages)
    except Exception as e:
        raise RuntimeError(f'pdf page count failed: {e!r}')
def build_paper_record(config,bundle_hash):
    if not os.environ.get('EXPERIMENTS7_ENVELOPE_ID'): raise RuntimeError('missing envelope id')
    p=config['paper']; parent=p['parent']; base=os.path.basename(p['path'])
    pfd=os.open(parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        pst=os.fstat(pfd)
        if (pst.st_dev,pst.st_ino)!=(p['parent_binding']['dev'],p['parent_binding']['inode']): raise RuntimeError('paper parent binding mismatch')
        ns_before=namespace_entries(pfd)
        first=os.stat(base,dir_fd=pfd,follow_symlinks=False)
        if not stat.S_ISREG(first.st_mode): raise RuntimeError('paper not regular')
        if (first.st_dev,first.st_ino)!=(p['binding']['dev'],p['binding']['inode']): raise RuntimeError('paper binding mismatch')
        fd=os.open(base,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=pfd)
        try:
            opened=os.fstat(fd)
            if stable(first)!=stable(opened): raise RuntimeError('paper substitution')
            dg,total=read_hash(fd); final=os.fstat(fd)
            if stable(opened)!=stable(final) or total!=final.st_size: raise RuntimeError('paper changed')
        finally: os.close(fd)
        pages=pdf_pages(p['path'])
        ns_after=namespace_entries(pfd)
        if ns_before!=ns_after or stable(pst)!=stable(os.fstat(pfd)): raise RuntimeError('paper namespace changed')
        rec={'schema':PAPER_SCHEMA,'path':p['path'],'ownership':'preserved_foreign_readonly','sha256':dg,'size':total,'pages':pages,'metadata':meta(final),'paper_directory':{'path':parent,'metadata':meta(pst),'entries':ns_after},'bundle_lock_sha256':bundle_hash,'envelope_id':os.environ['EXPERIMENTS7_ENVELOPE_ID'],'expected_identity':{'sha256':config['expected_paper']['sha256'],'size':config['expected_paper']['size'],'pages':config['expected_paper']['pages']}}
        if (dg,total,pages)!=(config['expected_paper']['sha256'],config['expected_paper']['size'],config['expected_paper']['pages']): raise RuntimeError('exact PDF identity mismatch')
        return rec
    finally: os.close(pfd)
def publish_paper_manifest(config,bundle_hash,output):
    r=build_paper_record(config,bundle_hash); res=atomic_publish(output,[canonical_json(r)]); res.update({'paper_sha256':r['sha256'],'paper_size':r['size'],'pages':r['pages']}); return res
