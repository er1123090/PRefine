#!/usr/bin/env python3
from __future__ import annotations
import argparse, errno, hashlib, json, os, platform, stat, sys
from datetime import datetime, timezone
import manifest, provider
HERE=os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE_LOCK=os.path.join(HERE,'bundle-lock.json')
def now(): return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
def canonical(v): return (json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
def hash_file(p):
    h=hashlib.sha256(); n=0
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b); n+=len(b)
    return h.hexdigest(),n
def load_json(p):
    with open(p,'rb') as f: return json.load(f)
def current_umask():
    u=os.umask(0); os.umask(u); return u
def verify_bundle(path):
    b=load_json(path)
    if b.get('schema')!='experiments7-g0v2-bundle-lock/v1': raise RuntimeError('bad bundle schema')
    for name,e in b['components'].items():
        dg,sz=hash_file(e['path'])
        if (dg,sz)!=(e['sha256'],e['size']): raise RuntimeError(f'frozen component mismatch {name}')
    dg,sz=hash_file(path)
    expected_payload=b.get('self_payload_sha256_before_self_fields')
    if not isinstance(expected_payload,str): raise RuntimeError('bundle self payload marker missing')
    return b,dg
def verify_runtime(r):
    exe=os.path.realpath(sys.executable); dg,sz=hash_file(exe)
    if exe!=r['python']['realpath'] or dg!=r['python']['sha256'] or sz!=r['python']['size']: raise RuntimeError('python runtime drift')
    if platform.python_version()!=r['python']['version'] or platform.release()!=r['kernel'] or platform.platform()!=r['platform']: raise RuntimeError('platform drift')
    if current_umask()!=r['umask']: raise RuntimeError('umask drift')
    env={k:os.environ.get(k) for k in r['environment']}
    if env!=r['environment']: raise RuntimeError('environment drift')
    if provider.provider_identity()!=r['provider']: raise RuntimeError('provider identity drift')
def nofollow(path, expected, typ):
    if os.path.realpath(path)!=path: raise RuntimeError(f'noncanonical {path}')
    st=os.lstat(path); at='directory' if stat.S_ISDIR(st.st_mode) else 'regular' if stat.S_ISREG(st.st_mode) else 'other'
    if at!=typ or (st.st_dev,st.st_ino)!=(expected['dev'],expected['inode']): raise RuntimeError(f'binding mismatch {path}')
    return {'path':path,'type':at,'dev':st.st_dev,'inode':st.st_ino,'mode':stat.S_IMODE(st.st_mode)}
def verify_bindings(c):
    out=[]
    for root in c['protected_sources']: out.append(nofollow(root['path'],root['binding'],'directory'))
    for item in [c['target'],c['owner_record'],c['root_readme']]: out.append(nofollow(item['path'],item['binding'],item['binding']['type']))
    rh,rs=hash_file(c['root_readme']['path'])
    if rh!=c['root_readme']['sha256'] or rs!=c['root_readme']['size']: raise RuntimeError('root README drift')
    out.append(nofollow(c['paper']['parent'],c['paper']['parent_binding'],'directory'))
    out.append(nofollow(c['paper']['path'],c['paper']['binding'],'regular'))
    for m in c['managed_prefixes']: out.append(nofollow(m['path'],m['binding'],'directory'))
    return out
def denial_probe(path):
    try: fd=os.open(path,os.O_WRONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    except OSError as e:
        if e.errno not in (errno.EPERM,errno.EACCES): raise RuntimeError(f'unexpected errno {e.errno} for {path}')
        return {'path':path,'operation':'open_write_no_trunc','denied':True,'errno':e.errno}
    else:
        os.close(fd); raise RuntimeError(f'write-capable open succeeded {path}')
def parse_source_events(path, envelope_id, bundle_hash, run_id):
    events=[]; total=0; reg=0; dirs=0
    with open(path,'rb') as f:
        for line in f:
            r=json.loads(line); op='read_file' if r['type']=='regular' else 'enumerate_directory' if r['type']=='directory' else 'read_metadata'
            bytes_read=int(r.get('size',0)) if r['type']=='regular' else 0
            if r['type']=='regular': reg+=1; total+=bytes_read
            if r['type']=='directory': dirs+=1
            events.append({'schema':'experiments7-protected-access-event/v1','phase':'pre','sealed_run_id':run_id,'task_id':'G008-g0v2-cp0v2-sealed-recovery','agent_type':'executor','bundle_lock_sha256':bundle_hash,'envelope_id':envelope_id,'operation':op,'root':r['root'],'path':r['path'],'source_record_id':r['record_id'],'bytes_read':bytes_read,'write_count':0})
    return events, {'regular_read_count':reg,'directory_enumeration_count':dirs,'bytes_read':total,'event_count':len(events)}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--verify-envelope',action='store_true',required=True); p.add_argument('--task-id',required=True); p.add_argument('--audit-output',required=True); p.add_argument('--access-output',required=True); p.add_argument('--bundle-lock',default=DEFAULT_BUNDLE_LOCK); p.add_argument('command',choices=['build-pre','build-post']); a=p.parse_args()
    audit={'schema':'experiments7-envelope-audit/v2','status':'BLOCKED','started_at':now(),'task_id':a.task_id,'agent_type':'executor','command':a.command,'wrapper_argv':sys.argv[1:]}
    try:
        bundle,bhash=verify_bundle(os.path.abspath(a.bundle_lock)); config=load_json(bundle['config_path']); argv=load_json(bundle['argv_path']); runtime=load_json(bundle['runtime_path'])
        expected=argv['commands'][a.command]['wrapper_args']
        if sys.argv[1:]!=expected: raise RuntimeError('frozen argv mismatch')
        if a.task_id!=argv['commands'][a.command]['task_id']: raise RuntimeError('task id mismatch')
        verify_runtime(runtime); bindings=verify_bindings(config)
        material={'schema':'experiments7-envelope-identity/v2','sealed_run_id':config['sealed_run_id'],'bundle_lock_sha256':bhash,'provider':runtime['provider'],'bindings':bindings,'writable_directories':config['writable_during_g0']}
        envelope_id=hashlib.sha256(canonical(material)).hexdigest(); os.environ['EXPERIMENTS7_ENVELOPE_ID']=envelope_id
        provider_evidence=provider.apply_readonly_envelope(config['writable_during_g0'])
        probes=[denial_probe(config['paper']['path']), denial_probe(config['root_readme']['path']), denial_probe(config['owner_record']['path'])]
        source_out=config['outputs'][a.command]['source']; paper_out=config['outputs'][a.command]['paper']
        source_res=manifest.publish_source_manifest(config,source_out); paper_res=manifest.publish_paper_manifest(config,bhash,paper_out)
        events, summary=parse_source_events(source_out,envelope_id,bhash,config['sealed_run_id'])
        events.append({'schema':'experiments7-protected-access-event/v1','phase':'pre','sealed_run_id':config['sealed_run_id'],'task_id':'G008-g0v2-cp0v2-sealed-recovery','agent_type':'executor','bundle_lock_sha256':bhash,'envelope_id':envelope_id,'operation':'read_exact_pdf','root':'experiments7/_paper','path':config['paper']['path'],'source_record_id':'paper-pre','bytes_read':paper_res['paper_size'],'write_count':0,'sha256':paper_res['paper_sha256'],'pages':paper_res['pages']})
        for probe in probes:
            events.append({'schema':'experiments7-protected-access-event/v1','phase':'pre','sealed_run_id':config['sealed_run_id'],'task_id':'G008-g0v2-cp0v2-sealed-recovery','agent_type':'executor','bundle_lock_sha256':bhash,'envelope_id':envelope_id,'operation':'write_denial_probe','root':'protected_or_immutable','path':probe['path'],'source_record_id':'denial-probe','bytes_read':0,'write_count':0,'denied':probe['denied'],'errno':probe['errno']})
        manifest.atomic_publish(a.access_output, (canonical(e) for e in events))
        audit.update({'status':'PASS','finished_at':now(),'sealed_run_id':config['sealed_run_id'],'envelope_id':envelope_id,'bundle_lock_sha256':bhash,'bindings':bindings,'provider_evidence':provider_evidence,'denial_probes':probes,'source_manifest':source_res,'paper_manifest':paper_res,'protected_access_summary':summary|{'paper_bytes_read':paper_res['paper_size'],'write_denial_probe_count':len(probes)}})
    except Exception as e:
        audit.update({'status':'BLOCKED','finished_at':now(),'reason':repr(e)})
        try: manifest.atomic_publish(a.audit_output,[canonical(audit)])
        except Exception: pass
        raise SystemExit(2)
    manifest.atomic_publish(a.audit_output,[canonical(audit)])
    print(json.dumps({'status':'PASS','audit':a.audit_output,'access':a.access_output,'source':source_res,'paper':paper_res},sort_keys=True))
if __name__=='__main__': main()
