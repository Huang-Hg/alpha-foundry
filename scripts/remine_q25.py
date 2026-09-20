"""全量候选重挖编排:经常驻 GPU 服务评分、并入、更新组合后全量再评。

python scripts/remine_q25.py <候选 JSON> <输出目录>
候选 = [{expr,lane}],Python 源引用 q25_funnel 下相对路径:函数名。
每轮全部候选对同一组合评估;仅主控按组合增量排序逐条重新判定后接受。
结果逐批落盘,状态文件包含进度/失败原因;不启动独立 GPU 进程。
"""
import json
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FUNNEL=ROOT/'tmp/q25_funnel'
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(FUNNEL))
import client
from config.config import Q25ComboConfig,Q25CudaConfig
PROTOCOL = 'combo-cuda-v3'


def main():
    pool=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
    store=FUNNEL/'combo_library.json'
    resume='--resume' in sys.argv[3:]
    restored=[]
    first_round=1
    if resume:
        status=json.loads((out/'status.json').read_text(encoding='utf-8'))
        assert status.get('protocol')==PROTOCOL, '计算协议已变，必须使用新输出目录全量重挖'
        assert status['phase']=='full-screen', '恢复只允许在完整筛选阶段;入库阶段不能复用旧参照'
        first_round=status['round']
        reference=out/('initial_combo.json' if first_round==1 else f'round{first_round-1}_combo.json')
        assert store.read_bytes()==reference.read_bytes(), '组合参照已改变,不能复用旧筛选结果'
        rows_path=out/f'round{first_round}_rows.jsonl'
        with rows_path.open('r+',encoding='utf-8') as previous:
            for i in range(status['done']):
                row=json.loads(previous.readline())
                assert row['source']==pool[i], '候选池顺序已变'
                if row['passed']: restored.append(row)
            # 未完成 checkpoint 的批次重算;已经提交的前缀保留。
            previous.truncate(previous.tell())
        start=time.time()-status.get('elapsed_seconds',0)
        status['state']='running'
    else:
        assert not (out/'status.json').exists(), '输出目录已有任务;使用 --resume 恢复筛选'
        (out/'initial_combo.json').write_bytes(store.read_bytes())
        start=time.time()
        status=dict(state='running',protocol=PROTOCOL,total=len(pool),round=0,done=0,accepted=0)
    def checkpoint(**kw):
        status.update(kw,elapsed_seconds=round(time.time()-start,1))
        p=out/'status.tmp'
        p.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
        p.replace(out/'status.json')
    def request(req):
        response=client.request(req,timeout=3600)
        assert response and response.get('ok'),response
        return response['rows']
    def source(item):
        rel,name=item['expr'].rsplit(':',1)
        return (FUNNEL/rel).read_text(encoding='utf-8'),name
    try:
        checkpoint()
        for rnd in range(first_round,Q25ComboConfig().max_members+1):
            continuing=resume and rnd==first_round
            offset=status['done'] if continuing else 0
            checkpoint(round=rnd,done=offset,phase='full-screen')
            finalists=restored if continuing else []
            with (out/f'round{rnd}_rows.jsonl').open('a' if continuing else 'w',encoding='utf-8') as f:
                batch_size=Q25CudaConfig().candidate_batch
                for lo in range(offset,len(pool),batch_size):
                    batch=pool[lo:lo+batch_size]
                    fast=[x for x in batch if x['lane']!='python']
                    rows=request(dict(op='eval',exprs=[x['expr'] for x in fast],accept=False,window='is')) if fast else []
                    by={r['expr']:r for r in rows}
                    for item in batch:
                        if item['lane']=='python':
                            src,name=source(item)
                            rr=request(dict(op='python',src=src,path=item['expr'],names=[name],window='is',accept=False))
                            assert len(rr)==1, item
                            row=rr[0]
                        else: row=by[item['expr']]
                        row['source']=item
                        f.write(json.dumps(row,ensure_ascii=False)+'\n')
                        if row['passed']: finalists.append(row)
                    f.flush()
                    checkpoint(done=min(lo+len(batch),len(pool)),passing=len(finalists))
                    print(json.dumps(status,ensure_ascii=False),flush=True)
            accepted=0
            checkpoint(phase='accept',passing=len(finalists))
            for row in sorted(finalists,key=lambda x:(x['combo']['delta_sharpe'],x['combo']['delta_cagr']),reverse=True):
                item=row['source']
                if item['lane']=='python':
                    src,name=source(item)
                    res=request(dict(op='python',src=src,path=item['expr'],names=[name],window='is',accept=True))[0]
                else:
                    res=request(dict(op='eval',exprs=[item['expr']],window='is',accept=True))[0]
                with (out/'acceptance.jsonl').open('a',encoding='utf-8') as f:
                    f.write(json.dumps(dict(round=rnd,source=item,result=res),ensure_ascii=False)+'\n')
                if res['accepted']:
                    accepted+=1
                    checkpoint(accepted=status['accepted']+1)
            (out/f'round{rnd}_combo.json').write_bytes(store.read_bytes())
            if not accepted:
                checkpoint(state='complete',phase='done',reason='完整一轮无可接受组合增量')
                break
        else:
            checkpoint(state='complete',phase='done',reason='达到预设轮数上限')
        (out/'final_combo.json').write_bytes(store.read_bytes())
    except BaseException:
        checkpoint(state='failed',error=traceback.format_exc())
        raise


if __name__=='__main__': main()
