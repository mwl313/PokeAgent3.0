import csv, datetime, hashlib, json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path('/home/minidc/Desktop/PokeAgent3.0')
OUT = ROOT / 'runs/perf/cooling_20261011'
OUT.mkdir(parents=True, exist_ok=True)
FIELDS = ['index','temperature.gpu','power.draw','power.limit','clocks.sm','clocks.mem',
          'utilization.gpu','utilization.memory','memory.used','pcie.link.gen.current',
          'pcie.link.width.current','clocks_event_reasons.sw_power_cap',
          'clocks_event_reasons.hw_thermal_slowdown','clocks_event_reasons.sw_thermal_slowdown',
          'clocks_event_reasons.hw_power_brake_slowdown']

def command(*args):
    return subprocess.check_output(args, text=True, cwd=ROOT)

def snapshot(name):
    (OUT / f'{name}_gpu.xml').write_text(command('nvidia-smi','-q','-x'))

def sample(proc):
    gpu = [dict(zip(FIELDS, row)) for row in csv.reader(command('nvidia-smi',
        '--query-gpu='+','.join(FIELDS), '--format=csv,noheader,nounits').splitlines(), skipinitialspace=True)]
    processes = {}
    for folder in pathlib.Path('/proc').iterdir():
        if not folder.name.isdigit(): continue
        try:
            raw = (folder/'stat').read_text().rsplit(')',1)[1].split()
            processes[int(folder.name)] = {'pid':int(folder.name),'ppid':int(raw[1]),
                'cpu_s':(int(raw[11])+int(raw[12]))/os.sysconf('SC_CLK_TCK'),
                'rss_bytes':int(raw[21])*os.sysconf('SC_PAGE_SIZE'),'threads':int(raw[17])}
        except (OSError,ValueError,IndexError): pass
    owned = {proc.pid}
    while True:
        more = {pid for pid,r in processes.items() if r['ppid'] in owned}
        if more <= owned: break
        owned |= more
    rows = []
    for pid in sorted(owned):
        if pid not in processes: continue
        row = processes[pid]
        try:
            folder = pathlib.Path('/proc')/str(pid)
            cmd = (folder/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            row['worker'] = '--worker' in cmd
            row['io'] = {k:int(v) for k,v in (line.split(':') for line in (folder/'io').read_text().splitlines())}
        except (OSError,ValueError): pass
        rows.append(row)
    mem = {k:int(v.split()[0]) for k,v in (line.split(':') for line in pathlib.Path('/proc/meminfo').read_text().splitlines())}
    vm = dict(line.split() for line in pathlib.Path('/proc/vmstat').read_text().splitlines())
    return {'utc_s':time.time(),'monotonic_s':time.monotonic(),'gpu':gpu,'processes':rows,
            'mem_available_kib':mem['MemAvailable'],'swap_free_kib':mem['SwapFree'],
            'vm':{k:int(vm[k]) for k in ['pgmajfault','pswpin','pswpout']}}

def run_case(name, iterations, executable=None, extra=None):
    executable = executable or ROOT/'scripts/run_ddp_ppo.py'
    report = OUT/f'{name}.json'
    cmd = [str(ROOT/'.venv/bin/python'),str(executable),'--games','1024','--envs','1024',
           '--workers','16','--microbatch','1024','--minibatch','4096','--iterations',str(iterations),
           '--timeout','1800','--report',str(report),'--checkpoint',str(OUT/'latest_verified.pt'),*(extra or [])]
    env = dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
               PYTHONPATH=str(ROOT/'engine/python')+os.pathsep+str(ROOT))
    started = time.time()
    snapshot(name+'_before')
    with (OUT/f'{name}.log').open('w') as log, (OUT/f'{name}_telemetry.jsonl').open('w') as samples:
        proc = subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None:
                tick = time.monotonic()
                samples.write(json.dumps(sample(proc))+'\n'); samples.flush()
                time.sleep(max(0,1-(time.monotonic()-tick)))
            if proc.returncode: raise RuntimeError(f'{name} exited {proc.returncode}; see {name}.log')
        finally:
            if proc.poll() is None: proc.terminate(); proc.wait(timeout=30)
    snapshot(name+'_after')
    r = json.loads(report.read_text())
    assert r['digests_equal']
    assert all(x['operational_errors']==0 and x['report']['epochs_run']==4 and
               x['report']['optimizer_steps_skipped']==0 for rank in r['ranks'] for x in rank['iterations'])
    (OUT/f'{name}_case.json').write_text(json.dumps({'command':cmd,'start_utc_s':started,
        'end_utc_s':time.time(),'telemetry_interval_s':1,'unprofiled':executable==ROOT/'scripts/run_ddp_ppo.py'},indent=2)+'\n')
    print(json.dumps({'case':name,'all_in':r['all_in_committed_games_per_s'],
                     'steady':r['steady_state'],'gates':r['digests_equal']}),flush=True)

if __name__ == '__main__':
    if len(sys.argv)>1:
        run_case(sys.argv[1],int(sys.argv[2]),pathlib.Path(sys.argv[3]) if len(sys.argv)>3 else None)
    else:
        manifest = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'git_sha':command('git','rev-parse','HEAD').strip(),'git_status':command('git','status','--short'),
            'engine_sha256':hashlib.sha256((ROOT/'engine/python/pa3_engine/pa3_engine.so').read_bytes()).hexdigest(),
            'lscpu':command('lscpu'),'topology':command('nvidia-smi','topo','-m'),
            'memory':pathlib.Path('/proc/meminfo').read_text(),
            'plan':'Interleaved cold1/persistent1/cold2/persistent2/cold3/persistent3. Each persistent has 3 iterations.'}
        (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for i in range(1,4):
            run_case(f'cold{i}',1)
            run_case(f'persistent{i}',3)
