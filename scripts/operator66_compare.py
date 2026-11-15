"""Compare saved endpoint reports without executing market computations."""
import json,sys
from pathlib import Path

def comparison(actual,expected):
    result={'decisions':[], 'risk_flags':[], 'weight_abs':0.,'numeric':{}}
    def numbers(a,b,path,identity):
        if isinstance(a,dict) and isinstance(b,dict):
            assert a.keys()==b.keys(),(identity,path)
            for key in a:numbers(a[key],b[key],path+'.'+key,identity)
        elif isinstance(a,list) and isinstance(b,list):
            if len(a)!=len(b):
                result['decisions'].append((identity,path,a,b))
                return
            for i,(x,y) in enumerate(zip(a,b)):numbers(x,y,path+f'[{i}]',identity)
        elif isinstance(a,(float,int)) and isinstance(b,(float,int)):
            error=abs(a-b);scaled=error/max(1.,abs(b));record=result['numeric'].setdefault(path,{'abs':0.,'scaled':0.,'id':identity})
            if error>record['abs']:record.update(abs=error,id=identity)
            record['scaled']=max(record['scaled'],scaled)
        elif (a is None)!=(b is None):
            result['decisions'].append((identity,path,'finite mismatch'))
        elif isinstance(a,str) and a!=b:
            result['decisions'].append((identity,path,a,b))
    for phase in ('query','commit'):
        aa,bb=actual[phase]['rows'],expected[phase]['rows'];assert len(aa)==len(bb)
        for a,b in zip(aa,bb):
            identity=phase+'/'+a['id'];assert a['id']==b['id']
            for key in ('ok','error','accepted'):
                if a[key]!=b[key]:result['decisions'].append((identity,key,a[key],b[key]))
            da,db=a['decision'],b['decision']
            for key in ('passed','stopped_at','reason'):
                if da[key]!=db[key]:result['decisions'].append((identity,key,da[key],db[key]))
            pa,pb=da['proposal'],db['proposal']
            if (pa is None)!=(pb is None):result['decisions'].append((identity,'proposal presence'))
            if pa is not None and pb is not None:
                ca=[(c['name'],c['passed']) for c in pa['checks']];cb=[(c['name'],c['passed']) for c in pb['checks']]
                if ca!=cb:result['decisions'].append((identity,'checks',ca,cb))
                result['weight_abs']=max(result['weight_abs'],abs(pa['weight']-pb['weight']))
                # Device labels, version hashes and human text containing rounded
                # metrics are representation, not admission semantics. Checks,
                # reason codes and numerical fields are compared explicitly.
                numeric_keys=pa.keys()-{'backend','reference_version','vetoes'}
                numbers({k:pa[k] for k in numeric_keys},{k:pb[k] for k in numeric_keys},'proposal',identity)
            result['weight_abs']=max(result['weight_abs'],abs(a['weight']-b['weight']))
            numbers(a['metrics'],b['metrics'],'metrics',identity)
            numbers(a['risk'],b['risk'],'risk',identity)
            if a['risk'] is not None and b['risk'] is not None:
                for key in ('liquidations','first_liquidation_ms','bankrupt'):
                    if a['risk'][key]!=b['risk'][key]:result['risk_flags'].append((identity,key,a['risk'][key],b['risk'][key]))
    result['max_abs']=max((n['abs'] for n in result['numeric'].values()),default=0.)
    result['max_scaled']=max((n['scaled'] for n in result['numeric'].values()),default=0.)
    return result

root=Path('output/operator66');tag=sys.argv[1];reference=sys.argv[2]
cpu=json.loads((root/reference/'cpu_reference.json').read_text())
production=json.loads((root/tag/'production.json').read_text())
single=json.loads((root/tag/'single.json').read_text())
result=dict(cpu=comparison(production,cpu),single=comparison(single,cpu),
            ratio=production['timings']['query_commit']/single['timings']['query_commit'])
(root/tag/'comparison.json').write_text(json.dumps(result,indent=2))
for mode in ('cpu','single'):
    row=result[mode]
    print(mode,'decisions',len(row['decisions']),'weights',row['weight_abs'],'max_scaled',row['max_scaled'])
    print('first differences',row['decisions'][:10])
    print('largest numeric errors',sorted(row['numeric'].items(),key=lambda p:-p[1]['scaled'])[:10])
print('ratio',result['ratio'])
assert not result['cpu']['decisions'] and not result['cpu']['risk_flags']
assert result['ratio']<=1.5
