"""Render saved benchmark evidence; no device, data loading or strategy execution."""
import csv
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


SCHEMA = 'foundry-benchmark-v1'
METRIC_ATOL = 1e-6
METRIC_RTOL = 1e-6
WEIGHT_ATOL = 1e-7
OPERATOR_SCALED_TOL = 1e-10


def compare_reports(actual, expected):
    """Compare semantic decisions exactly and metrics with named scale-aware limits."""
    result = dict(decisions=[], finite=[], numeric={}, numeric_failures=[])

    def walk(a, b, path, identity):
        if isinstance(a, dict) and isinstance(b, dict):
            if a.keys() != b.keys():
                raise ValueError(f'Report schema differs at {path}')
            for key in a:
                walk(a[key], b[key], f'{path}.{key}', identity)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                result['decisions'].append(dict(id=identity, field=path, actual=len(a), expected=len(b)))
            else:
                for x, y in zip(a, b):
                    walk(x, y, path+'[]', identity)
        elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str):
            if a != b:
                result['decisions'].append(dict(id=identity, field=path, actual=a, expected=b))
        elif a is None or b is None:
            if (a is None) != (b is None):
                result['finite'].append(dict(id=identity, field=path))
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            error = abs(a-b)
            scaled = error / max(1., abs(b))
            limit = WEIGHT_ATOL if path.endswith('.weight') else METRIC_ATOL + METRIC_RTOL * abs(b)
            record = result['numeric'].setdefault(path, dict(max_abs=0., max_scaled=0., worst_id=identity, failures=0))
            if error > record['max_abs']:
                record.update(max_abs=error, worst_id=identity)
            record['max_scaled'] = max(record['max_scaled'], scaled)
            if not math.isfinite(error) or error > limit:
                record['failures'] += 1
                result['numeric_failures'].append(dict(id=identity, field=path, error=error, limit=limit))
        else:
            raise ValueError(f'Unexpected report shape at {path}')

    for phase in ('query', 'commit'):
        aa, bb = actual[phase]['rows'], expected[phase]['rows']
        if [r['id'] for r in aa] != [r['id'] for r in bb]:
            raise ValueError('Candidate identity or ordering differs')
        for a, b in zip(aa, bb):
            identity = phase + '/' + a['id']
            for key in ('source', 'ok', 'accepted', 'weight', 'metrics', 'risk'):
                walk(a[key], b[key], 'row.'+key, identity)
            da, db = a['decision'], b['decision']
            for key in ('passed', 'stopped_at', 'reason'):
                walk(da[key], db[key], 'decision.'+key, identity)
            pa, pb = da['proposal'], db['proposal']
            excluded = {'backend', 'reference_version', 'vetoes'}
            walk({k:v for k,v in pa.items() if k not in excluded} if pa is not None else None,
                 {k:v for k,v in pb.items() if k not in excluded} if pb is not None else None,
                 'proposal', identity)
            if a['risk'] is not None and b['risk'] is not None:
                for key in ('liquidations', 'first_liquidation_ms', 'bankrupt'):
                    if a['risk'][key] != b['risk'][key]:
                        result['decisions'].append(dict(id=identity, field='risk.'+key,
                                                       actual=a['risk'][key], expected=b['risk'][key]))
    result['passed'] = not (result['decisions'] or result['finite'] or result['numeric_failures'])
    return result


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(directory, *, exit_code=0):
    """Produce bilingual summaries and portable artifacts from one pytest run."""
    root = Path(directory)
    raw = json.loads((root / 'benchmark.json').read_text())
    context = raw['foundry']
    if context['schema'] != SCHEMA:
        raise ValueError('Unsupported benchmark schema')
    tests = ET.parse(root / 'junit.xml').getroot()
    counts = {key: sum(int(s.get(key, 0)) for s in tests.iter('testsuite'))
              for key in ('tests', 'failures', 'errors', 'skipped')}
    operators, flows = {}, {}
    for row in raw['benchmarks']:
        info = row['extra_info']
        key = (info['operator'], info['window']) if info['suite'] == 'operator' else info['candidates']
        group = operators if info['suite'] == 'operator' else flows
        group.setdefault(key, {})[info['backend']] = dict(info, wall_s=row['stats']['mean'])
    expected_cases = {(name, window) for name, window in operators}
    incomplete = [str(key) for key, group in list(operators.items()) + list(flows.items()) if set(group) != {'cpu', 'cuda'}]
    missing_flows = sorted(set(context['plan']['sizes']) - flows.keys())
    operator_rows = []
    for (name, window), group in operators.items():
        if set(group) != {'cpu', 'cuda'}:
            continue
        cpu, gpu = group['cpu'], group['cuda']
        operator_rows.append(dict(operator=name, window=window, days=gpu['shape'][0], assets=gpu['shape'][1],
            cpu_us=cpu['wall_s']*1e6, cuda_us=gpu['wall_s']*1e6, speedup=cpu['wall_s']/gpu['wall_s'], **gpu['error']))
    flow_rows, comparisons = [], {}
    for size, group in flows.items():
        if set(group) != {'cpu', 'cuda'}:
            continue
        details = {}
        for backend, row in group.items():
            payload = (root / row['detail']).read_bytes()
            if hashlib.sha256(payload).hexdigest() != row['detail_sha256']:
                raise ValueError('Benchmark detail checksum mismatch')
            details[backend] = json.loads(payload)
        for field in ('study', 'data_snapshot', 'execution_snapshot', 'allocation'):
            if details['cpu']['snapshot'][field] != details['cuda']['snapshot'][field]:
                raise ValueError(f'CPU/CUDA workload differs: {field}')
        comparisons[str(size)] = compare_reports(details['cuda'], details['cpu'])
        for backend, row in group.items():
            flow_rows.append(dict(candidates=size, backend=backend, **row['stages'],
                                  pytest_wall_s=row['wall_s'], valid=row['valid'], accepted=row['accepted'],
                                  sampled_device_peak_mib=row['sampled_device_peak_mib']))
    operator_ok = all(r['max_scaled'] <= OPERATOR_SCALED_TOL and
                      r['nan_mismatch'] == r['inf_mismatch'] == r['sign_mismatch'] == 0 for r in operator_rows)
    coverage = len({name for name, _ in expected_cases})
    passed = (exit_code == 0 and not counts['failures'] and not counts['errors'] and not incomplete and
              not missing_flows and coverage == 66 and operator_ok and all(c['passed'] for c in comparisons.values()))
    summary = dict(schema=SCHEMA, passed=passed, validation=counts, operator_count=coverage,
                   report_renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   incomplete=incomplete, missing_flows=missing_flows, comparisons=comparisons,
                   operator_tolerance=OPERATOR_SCALED_TOL,
                   metric_atol=METRIC_ATOL, metric_rtol=METRIC_RTOL, weight_atol=WEIGHT_ATOL,
                   precision_budget='not_evaluated: no single-arithmetic probe in this CPU/CUDA suite')
    (root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    if operator_rows:
        write_csv(root / 'operators.csv', operator_rows)
    if flow_rows:
        write_csv(root / 'flows.csv', flow_rows)
    for chinese, filename in ((True, 'report.zh-CN.md'), (False, 'report.md')):
        lines = markdown(context, summary, operator_rows, flows, chinese)
        (root / filename).write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return passed


def markdown(context, summary, operators, flows, zh):
    def tr(chinese, english):
        return chinese if zh else english
    lines = ['# '+tr('CPU 与 CUDA：公开接口性能基准', 'CPU vs CUDA: public API benchmarks'), '',
        tr('本报告回答：同一批策略在 CPU 和 CUDA 上分别需要多久，输出误差有多大，是否做出相同的接纳决策。',
           'This report compares CPU and CUDA latency, numerical error and admission decisions for identical strategies.'), '',
        f"**{tr('验收', 'Result')}: {'PASS' if summary['passed'] else 'FAIL'}** · {context['gpu']}", '',
        '[JSON](summary.json) · [pytest JSON](benchmark.json) · [HTML](checks.html) · [JUnit](junit.xml) · [CSV](operators.csv)', '',
        '## '+tr('完整调用：加载 → 评价 → 接纳 → 释放', 'Complete call: open → evaluate → commit → close'), '',
        tr('评价计算公式、持仓、回测指标和候选权重；接纳按输入顺序更新组合。每轮从相同的等权基准开始，使用内存存储。',
           'Evaluate computes formulas, positions, backtest metrics and proposed weights. Commit updates the portfolio in input order. Every round starts from the same equal-weight baseline in memory.'), '',
        '| '+tr('候选数 | CPU 全调用 / s | CUDA 全调用 / s | 加速比 | CPU 评价＋接纳 / s | CUDA 评价＋接纳 / s | 有效 / 接纳 |',
                 'Candidates | CPU full call / s | CUDA full call / s | Speedup | CPU evaluate+commit / s | CUDA evaluate+commit / s | Valid / admitted |'),
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    parameters = context['plan']['parameters']
    dimensions = f"{operators[0]['days']} × {operators[0]['assets']}" if operators else tr('未测量', 'not measured')
    scope = tr(
        f"市场：`{parameters['market']}`；面板：**{dimensions}（日期 × 资产，含预热历史）**；研究区间：{parameters['start']} 至 {parameters['end']}；CPU OpenMP 线程数：{context['omp_num_threads']}。",
        f"Market: `{parameters['market']}`; panel: **{dimensions} (dates × assets, including warm-up history)**; study: {parameters['start']} through {parameters['end']}; CPU OpenMP threads: {context['omp_num_threads']}.")
    lines[6:6] = [scope, '']
    for size, group in sorted(flows.items()):
        if set(group) != {'cpu', 'cuda'}:
            continue
        c, g = group['cpu'], group['cuda']
        lines.append(f"| {size} | {c['wall_s']:.4f} | {g['wall_s']:.4f} | {c['wall_s']/g['wall_s']:.2f}× | "
                     f"{c['stages']['evaluate_commit_s']:.4f} | {g['stages']['evaluate_commit_s']:.4f} | {g['valid']} / {g['accepted']} |")
    lines += ['', tr('全调用包含会话打开和关闭。评价＋接纳只是一部分，不能标为全流程耗时。阶段明细见 [flows.csv](flows.csv)。',
                    'Full call includes opening and closing the session. Evaluate+commit is a subset, not total latency. See [flows.csv](flows.csv) for stages.'), '',
              '## '+tr('结果是否一致', 'Are the results consistent?'), '',
              '| '+tr('候选数 | 决策差异 | 有限值差异 | 数值超限 | 结论 |', 'Candidates | Decision mismatches | Finite-value mismatches | Numerical violations | Result |'),
              '| ---: | ---: | ---: | ---: | --- |']
    for size, comparison in summary['comparisons'].items():
        lines.append(f"| {size} | {len(comparison['decisions'])} | {len(comparison['finite'])} | {len(comparison['numeric_failures'])} | {'PASS' if comparison['passed'] else 'FAIL'} |")
    if summary['comparisons']:
        largest = max(summary['comparisons'], key=int)
        numeric = summary['comparisons'][largest]['numeric']
        lines += ['', tr(f'最大批量（{largest} 条）的误差，取评价与接纳阶段的最坏值：',
                         f'Worst errors across evaluate and commit for the largest batch ({largest} candidates):'), '',
                  '| '+tr('比较量 | 最大绝对误差 | 最大尺度误差 |', 'Quantity | Max absolute error | Max scaled error |'),
                  '| --- | ---: | ---: |']
        for key, label in (('row.metrics.sharpe', 'Sharpe'), ('row.metrics.cagr', 'CAGR'),
                           ('row.metrics.maxdd', tr('最大回撤', 'Max drawdown')),
                           ('proposal.weight', tr('选择权重', 'Selected weight')),
                           ('row.metrics.concentration', tr('收益集中度', 'Return concentration'))):
            if key in numeric:
                record = numeric[key]
                lines.append(f"| {label} | {record['max_abs']:.3e} | {record['max_scaled']:.3e} |")
    lines += ['', tr('逐项比较有效性、接纳身份、原因码、约束布尔值、强平事件和权重。设备标签、版本哈希和带舍入数字的说明文字不参与语义比较。',
                    'Validity, admitted IDs, reason codes, constraint booleans, liquidation events and weights are compared. Device labels, version hashes and rounded prose are not semantic decisions.'), '',
              tr('指标容差为 `1e-6 + 1e-6 × |CPU 值|`，权重绝对容差为 `1e-7`；决策必须完全一致。误差详情及最坏候选 ID 见 summary.json。',
                 'Metric tolerance is `1e-6 + 1e-6 × |CPU value|`; weight absolute tolerance is `1e-7`. Decisions must match exactly. summary.json contains errors and worst candidate IDs.'), '',
              '## '+tr('66 个算子的逐项耗时', 'Per-operator latency'), '',
              tr('下表使用相同真实面板；滚动算子展示 20 日窗口，其余展示唯一条件。CPU 和 CUDA 都是主机墙钟时间，CUDA 在返回前同步。2/20/120 日的全部条件见 operators.csv。',
                 'Both backends use the same real panel. The table shows window 20 for rolling operators and the single case otherwise. Both columns measure host wall time; CUDA synchronizes before returning. operators.csv includes windows 2/20/120.'), '',
              '| '+tr('算子 | CPU / μs | CUDA / μs | 加速比 | 最大绝对误差 | 最大尺度误差 |',
                       'Operator | CPU / μs | CUDA / μs | Speedup | Max absolute error | Max scaled error |'),
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in operators:
        if row['window'] in (0, 20):
            lines.append(f"| `{row['operator']}` | {row['cpu_us']:.1f} | {row['cuda_us']:.1f} | {row['speedup']:.2f}× | {row['max_abs']:.2e} | {row['max_scaled']:.2e} |")
    counts = summary['validation']
    lines += ['', '## '+tr('如何解读与复现', 'Interpretation and reproduction'), '',
              tr('- 每个条件预热一次、测量一次；这是一次观测，没有统计置信区间。微秒级差异尤其不宜据此排名。',
                 '- One warm-up and one measured round per condition. This is one observation, not a confidence interval; avoid ranking small microsecond differences.'),
              tr('- 预热和计时使用新会话，不修改私有缓存。预热时间单独保存在 JSON；其他条件可能已编译相关内核，因此不称为冷启动。',
                 '- Warm-up and timing use fresh sessions without private-cache changes. Warm-up stages are recorded separately. Other cases may already have compiled kernels, so this is not a cold-start claim.'),
              tr('- 算子上传与准备在计时外并单独记录；完整调用包含加载/准备，不含安装、原生构建、RPC 网络、磁盘组合保存。',
                 '- Operator upload/preparation is recorded outside operator timing. Full calls include loading/preparation but exclude installation, native builds, RPC and portfolio disk persistence.'),
              tr('- 显存每 50 ms 采样整个设备，包括上下文，可能遗漏更短峰值；采样覆盖预热与计时。',
                 '- Device-wide memory is sampled every 50 ms, including context and both warm-up and measured calls; short peaks may be missed.'),
              tr('- 本轮比较正式 CPU/CUDA 后端，不包含单精度实验对照，不据此重新宣称满足 1.5× 精度预算。',
                 '- This compares production CPU/CUDA backends, not a single-arithmetic experiment; it does not requalify the 1.5× precision budget.'),
              f"- {tr('验证用例', 'Validation cases')}: {counts['tests']}; {tr('失败', 'failures')}: {counts['failures']}; "
              f"{tr('错误', 'errors')}: {counts['errors']}; {tr('跳过', 'skipped')}: {counts['skipped']}.",
              tr('- 旧私有生产基线因协议过时明确跳过，原文件保留。PIT、A 股、现货、资金费与强平检查继续执行。',
                 '- The obsolete private production baseline is explicitly skipped and preserved. PIT, A-share, spot, funding and liquidation checks still run.'), '',
              '```bash', 'python -m pip install -r requirements-benchmark.txt', 'bash build.sh',
              'OMP_NUM_THREADS=2 python -m foundry benchmark output/benchmark-new', '```', '',
              tr('在授权 GPU 环境执行；CUDA 的 CuPy 包需与运行时匹配。完整参数、源码摘要、数据/执行快照、公式摘要及硬件信息保存在 JSON 中。',
                 'Run on the authorized GPU with a matching CuPy package. JSON records parameters, source hashes, data/execution snapshots, corpus identity and hardware.')]
    return lines
