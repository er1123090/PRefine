"""Linux write-boundary providers, loaded only after bundle verification."""
import ctypes, errno, os, platform, stat

libc = ctypes.CDLL(None, use_errno=True)
PR_SET_NO_NEW_PRIVS = 38
SECCOMP_SET_MODE_FILTER = 1
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1
AUDIT_ARCH_X86_64 = 0xC000003E

# Landlock ABI v3 write-like filesystem rights, exactly (no read rights).
ACCESS_EXECUTE=1<<0; ACCESS_WRITE_FILE=1<<1; ACCESS_READ_FILE=1<<2
ACCESS_READ_DIR=1<<3; ACCESS_REMOVE_DIR=1<<4; ACCESS_REMOVE_FILE=1<<5
ACCESS_MAKE_CHAR=1<<6; ACCESS_MAKE_DIR=1<<7; ACCESS_MAKE_REG=1<<8
ACCESS_MAKE_SOCK=1<<9; ACCESS_MAKE_FIFO=1<<10; ACCESS_MAKE_BLOCK=1<<11
ACCESS_MAKE_SYM=1<<12; ACCESS_REFER=1<<13; ACCESS_TRUNCATE=1<<14
WRITE_BITS=(ACCESS_WRITE_FILE|ACCESS_REMOVE_DIR|ACCESS_REMOVE_FILE|ACCESS_MAKE_CHAR|
            ACCESS_MAKE_DIR|ACCESS_MAKE_REG|ACCESS_MAKE_SOCK|ACCESS_MAKE_FIFO|
            ACCESS_MAKE_BLOCK|ACCESS_MAKE_SYM|ACCESS_REFER|ACCESS_TRUNCATE)

class Ruleset(ctypes.Structure): _fields_=[('handled_access_fs',ctypes.c_uint64)]
class PathRule(ctypes.Structure): _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int)]
class SockFilter(ctypes.Structure): _fields_=[('code',ctypes.c_ushort),('jt',ctypes.c_ubyte),('jf',ctypes.c_ubyte),('k',ctypes.c_uint32)]
class SockFprog(ctypes.Structure): _fields_=[('len',ctypes.c_ushort),('filter',ctypes.POINTER(SockFilter))]

def _syscall(n,*args):
    rc=libc.syscall(n,*args)
    if rc < 0:
        e=ctypes.get_errno(); raise OSError(e,os.strerror(e))
    return rc

def landlock_abi():
    try: return _syscall(444,0,0,LANDLOCK_CREATE_RULESET_VERSION)
    except OSError: return 0

def enforce_landlock(write_dir_fds):
    """Install the per-process zero-write boundary using descriptor-bound allows."""
    abi=landlock_abi()
    if abi < 3: raise RuntimeError('Landlock ABI >=3 required')
    attr=Ruleset(WRITE_BITS); ruleset=_syscall(444,ctypes.byref(attr),ctypes.sizeof(attr),0)
    try:
        for pfd in write_dir_fds:
            st=os.fstat(pfd)
            if not stat.S_ISDIR(st.st_mode): raise RuntimeError('Landlock allow is not directory')
            rule=PathRule(WRITE_BITS,pfd)
            _syscall(445,ruleset,LANDLOCK_RULE_PATH_BENEATH,ctypes.byref(rule),0)
        if libc.prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0):
            e=ctypes.get_errno(); raise OSError(e,os.strerror(e))
        _syscall(446,ruleset,0)
    finally: os.close(ruleset)
    return {'provider':'landlock','result':'enforced','abi':abi,
            'handled_access_fs':WRITE_BITS,'allow_rule_count':len(write_dir_fds),
            'descriptor_bound':True}

def enforce_publisher_landlock(read_dir_fds, write_dir_fds):
    """Deny publisher reads outside its stage/publication capability roots."""
    abi=landlock_abi()
    if abi < 3: raise RuntimeError('Landlock ABI >=3 required')
    read_bits=ACCESS_READ_FILE|ACCESS_READ_DIR
    handled=WRITE_BITS|read_bits
    attr=Ruleset(handled); ruleset=_syscall(444,ctypes.byref(attr),ctypes.sizeof(attr),0)
    try:
        for pfd in read_dir_fds:
            st=os.fstat(pfd)
            if not stat.S_ISDIR(st.st_mode): raise RuntimeError('Landlock read allow is not directory')
            allowed=read_bits|(WRITE_BITS if pfd in write_dir_fds else 0)
            rule=PathRule(allowed,pfd)
            _syscall(445,ruleset,LANDLOCK_RULE_PATH_BENEATH,ctypes.byref(rule),0)
        if libc.prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0):
            e=ctypes.get_errno(); raise OSError(e,os.strerror(e))
        _syscall(446,ruleset,0)
    finally: os.close(ruleset)
    return {'provider':'landlock','result':'enforced','abi':abi,'handled_access_fs':handled,
            'read_allow_rule_count':len(read_dir_fds),
            'write_allow_rule_count':len(write_dir_fds),'descriptor_bound':True}

def enforce_metadata_seccomp():
    """Default allow; deny only versioned metadata-write syscalls on x86_64."""
    if platform.machine()!='x86_64': raise RuntimeError('seccomp supports x86_64 only')
    # chmod/fchmod/chown/fchown/lchown, utime/utimes, *xattr writes/removals,
    # fchownat/futimesat/fchmodat/utimensat, and fchmodat2.
    denied=[90,91,92,93,94,132,188,189,190,197,198,199,235,260,261,268,280,452]
    # Validate seccomp_data.arch before inspecting nr. Wrong architecture is killed.
    ins=[SockFilter(0x20,0,0,4),
         SockFilter(0x15,1,0,AUDIT_ARCH_X86_64),
         SockFilter(0x06,0,0,0x80000000),
         SockFilter(0x20,0,0,0)]
    for nr in denied:
        ins.extend((SockFilter(0x15,0,1,nr),SockFilter(0x06,0,0,0x00050000|errno.EPERM)))
    ins.append(SockFilter(0x06,0,0,0x7fff0000))
    arr=(SockFilter*len(ins))(*ins); prog=SockFprog(len(ins),arr)
    if libc.prctl(PR_SET_NO_NEW_PRIVS,1,0,0,0):
        e=ctypes.get_errno(); raise OSError(e,os.strerror(e))
    _syscall(317,SECCOMP_SET_MODE_FILTER,0,ctypes.byref(prog))
    return {'provider':'seccomp-bpf','result':'enforced','audit_arch':'AUDIT_ARCH_X86_64',
            'default':'allow','denied_syscalls':denied,'errno':'EPERM'}

def renameat2_noreplace(old_dir_fd, old_name, new_dir_fd, new_name):
    """Exercise renameat2(RENAME_NOREPLACE) for the disposable denial matrix."""
    return _syscall(316, old_dir_fd, ctypes.c_char_p(os.fsencode(old_name)),
                    new_dir_fd, ctypes.c_char_p(os.fsencode(new_name)), 1)
