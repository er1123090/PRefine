#!/usr/bin/env python3
from __future__ import annotations
import ctypes, errno, os, platform, stat
from collections.abc import Iterable
SYS_LANDLOCK_CREATE_RULESET=444; SYS_LANDLOCK_ADD_RULE=445; SYS_LANDLOCK_RESTRICT_SELF=446
LANDLOCK_CREATE_RULESET_VERSION=1; LANDLOCK_RULE_PATH_BENEATH=1
LL_EXECUTE=1<<0; LL_WRITE_FILE=1<<1; LL_READ_FILE=1<<2; LL_READ_DIR=1<<3; LL_REMOVE_DIR=1<<4; LL_REMOVE_FILE=1<<5; LL_MAKE_CHAR=1<<6; LL_MAKE_DIR=1<<7; LL_MAKE_REG=1<<8; LL_MAKE_SOCK=1<<9; LL_MAKE_FIFO=1<<10; LL_MAKE_BLOCK=1<<11; LL_MAKE_SYM=1<<12; LL_REFER=1<<13; LL_TRUNCATE=1<<14; LL_READ_EXEC=LL_EXECUTE|LL_READ_FILE|LL_READ_DIR; LL_ALL=(1<<15)-1
PR_SET_NO_NEW_PRIVS=38; PR_SET_SECCOMP=22; SECCOMP_MODE_FILTER=2; SECCOMP_RET_ALLOW=0x7FFF0000; SECCOMP_RET_ERRNO=0x00050000; BPF_LD_W_ABS=0x20; BPF_JMP_JEQ_K=0x15; BPF_RET_K=0x06
DENIED_METADATA_SYSCALLS=(90,91,92,93,94,132,188,189,190,197,198,199,235,260,261,268,280,452)
class RulesetAttr(ctypes.Structure): _fields_=[('handled_access_fs',ctypes.c_uint64)]
class PathBeneathAttr(ctypes.Structure): _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32),('reserved',ctypes.c_uint32)]
class SockFilter(ctypes.Structure): _fields_=[('code',ctypes.c_ushort),('jt',ctypes.c_ubyte),('jf',ctypes.c_ubyte),('k',ctypes.c_uint32)]
class SockFprog(ctypes.Structure): _fields_=[('len',ctypes.c_ushort),('filter',ctypes.POINTER(SockFilter))]
LIBC=ctypes.CDLL(None,use_errno=True)
def _syscall(n:int,*args:object)->int:
    r=int(LIBC.syscall(n,*args))
    if r<0:
        e=ctypes.get_errno(); raise OSError(e,os.strerror(e))
    return r
def query_landlock_abi()->int: return _syscall(SYS_LANDLOCK_CREATE_RULESET,0,0,LANDLOCK_CREATE_RULESET_VERSION)
def provider_identity()->dict[str,object]:
    return {'provider':'landlock_path_beneath+seccomp_metadata_deny','landlock_abi':query_landlock_abi(),'seccomp_mode':'classic_bpf_errno_eperm','denied_metadata_syscalls_x86_64':list(DENIED_METADATA_SYSCALLS),'kernel':platform.release(),'machine':platform.machine(),'claim':'per_process_zero_write_enforcement_only_not_global_read_denial'}
def _real_directory(path:str):
    absolute=os.path.abspath(path); cur='/'
    for comp in [p for p in absolute.split(os.sep) if p]:
        cur=os.path.join(cur,comp); st=os.lstat(cur)
        if stat.S_ISLNK(st.st_mode): raise RuntimeError(f'symlink component rejected: {cur}')
    st=os.lstat(absolute)
    if not stat.S_ISDIR(st.st_mode) or os.path.realpath(absolute)!=absolute: raise RuntimeError(f'not exact real dir: {absolute}')
    return st
def _add_rule(fd:int,path:str,rights:int):
    st=_real_directory(path); pfd=os.open(path,os.O_PATH|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        attr=PathBeneathAttr(rights,pfd,0); _syscall(SYS_LANDLOCK_ADD_RULE,fd,LANDLOCK_RULE_PATH_BENEATH,ctypes.byref(attr),0)
        chk=os.fstat(pfd)
        if (chk.st_dev,chk.st_ino)!=(st.st_dev,st.st_ino): raise RuntimeError('binding changed')
        return {'path':path,'dev':st.st_dev,'inode':st.st_ino,'rights':rights}
    finally: os.close(pfd)
def _seccomp():
    if platform.machine()!='x86_64': raise RuntimeError('unsupported seccomp arch')
    ins=[SockFilter(BPF_LD_W_ABS,0,0,0)]
    for n in DENIED_METADATA_SYSCALLS:
        ins.append(SockFilter(BPF_JMP_JEQ_K,0,1,n)); ins.append(SockFilter(BPF_RET_K,0,0,SECCOMP_RET_ERRNO|errno.EPERM))
    ins.append(SockFilter(BPF_RET_K,0,0,SECCOMP_RET_ALLOW)); arr=(SockFilter*len(ins))(*ins); prog=SockFprog(len(ins),arr)
    if LIBC.prctl(PR_SET_SECCOMP,SECCOMP_MODE_FILTER,ctypes.byref(prog),0,0)!=0:
        e=ctypes.get_errno(); raise OSError(e,os.strerror(e))
def apply_readonly_envelope(writable_directories:Iterable[str])->dict[str,object]:
    abi=query_landlock_abi()
    if abi<4: raise RuntimeError(f'Landlock ABI {abi} lacks required mediation')
    writable=sorted({os.path.abspath(p) for p in writable_directories})
    rs=_syscall(SYS_LANDLOCK_CREATE_RULESET,ctypes.byref(RulesetAttr(LL_ALL)),ctypes.sizeof(RulesetAttr),0); rules=[]
    try:
        rules.append(_add_rule(rs,'/',LL_READ_EXEC))
        for p in writable: rules.append(_add_rule(rs,p,LL_ALL))
        if LIBC.prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0)!=0:
            e=ctypes.get_errno(); raise OSError(e,os.strerror(e))
        _syscall(SYS_LANDLOCK_RESTRICT_SELF,rs,0)
    finally: os.close(rs)
    _seccomp()
    return {**provider_identity(),'landlock_rules':rules,'default_filesystem_rights':'read_execute_only','writable_directories':writable,'writes_allowed_only_under':writable}
