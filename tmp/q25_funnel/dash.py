"""回测服务的**实时版面** —— 原地重绘,占用**你正坐在前面的那一个终端**。

**默认形态:面板长在你敲命令的那个终端里,不需要任何额外动作。**
`client.evaluate()` 等结果的时候顺手把面板画出来(见 `Feed`)—— 跑回测就有可视化,
一个终端,零额外命令。

    run.py "表达式"          ← 回测 + 实时面板,都在本终端(默认)
    run.py --daemon serve    服务也跑在本终端(前台调试用)
    run.py --daemon dash     只看面板,不发作业(服务已在跑时的只读旁观)

为什么面板不能长在服务端:实测从 WSL interop 起的 Windows 进程再拉子进程,
`CREATE_NEW_CONSOLE` **不会给可见窗口**(见 client._spawn)⇒ 被脚本自动拉起的服务
注定无窗口,它画什么都没人看得见。所以面板必须归**调用方**。

原地重绘而不是追加日志:一次 sweep 有几百个批次,追加流里每批一行、每行十二个字段,
信息全在但没人读得下去。真正要盯的是**此刻**的队列深度、漏斗形状和峰值 —— 那是状态,
不是事件。事件只留最近几条,全量进 `.q25d.log` 存档。

`render(stats) -> Panel` 是纯函数,喂什么画什么;轮询只在 `main()` 里。
"""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# 漏斗各层的显示名与配色。顺序即候选被扼杀的先后 —— 版面上的自上而下就是流程的自前向后。
LAYERS = [('G1', '结构/因果', 'dim'),
          ('G2', '组合准入', 'red'),
          ('PASS', '全过', 'bold green')]


def _hms(s: float) -> str:
    s = int(s)
    return f'{s // 3600}h{s // 60 % 60:02d}m' if s >= 3600 else f'{s // 60}m{s % 60:02d}s'


def _bar(frac: float, width: int = 22) -> str:
    """整块 + 八分之一残块,让 1% 这种小占比也看得见方向(纯空条会误读成"没有")。"""
    if frac <= 0:
        return '·' * width
    full = int(frac * width)
    rest = int((frac * width - full) * 8)
    return '█' * full + ('▏▎▍▌▋▊▉'[rest - 1] if rest else '') + \
           '·' * (width - full - (1 if rest else 0))


def header(st: dict) -> Text:
    t = Text()
    busy = st.get('cur')
    t.append('  ● 计算中' if busy else '  ○ 空闲  ', style='bold green' if busy else 'dim')
    if busy:
        t.append(f"   {busy['n']:,} 条 · {busy['clients']} 客户端合批 · "
                 f"{busy['elapsed']:.1f}s", style='white')
        # 批内进度。漏斗只在**整批**结束时才动,一个批几十秒 —— 没有这一行,
        # 中途每一帧看起来都一模一样,像卡死。
        tot = busy.get('total') or 0
        if tot:
            done = min(busy.get('done', 0), tot)
            frac = done / tot
            se = busy.get('stage_elapsed', busy['elapsed'])
            eta = se / frac - se if frac > 0.02 else None
            t.append(f"\n  {busy.get('stage', '')}  ", style='cyan')
            t.append(_bar(frac, 28), style='cyan')
            t.append(f"  {done:,}/{tot:,}  {frac * 100:.0f}%", style='white')
            t.append(f"  {done / se:,.0f} 条/s" if se > 0.5 and done else '', style='white')
            t.append(f"  剩 ~{eta:.0f}s" if eta else '', style='dim')
    q = st.get('queue', 0)
    t.append(f"\n  队列 ", style='dim')
    t.append(f'{q}', style='bold red' if q > 3 else 'white')
    t.append(f"   客户端 {st.get('clients', 0)}   显存 {st.get('vram', '-')}"
             f"   运行 {_hms(st.get('uptime', 0))}", style='dim')
    if (p:=st.get('portfolio')):
        t.append(f"\n  组合 {p['n']} 构件 · Sharpe {p['sharpe']:.3f} · 近期 {p['wsharpe']:.3f}"
                 f" · CAGR {p['cagr']:.1%} · 最大回撤 {p['maxdd']:.1%}",style='cyan')
    return t


def funnel(st: dict) -> Table | Text:
    f = st.get('funnel') or {}
    tot = sum(f.values())
    tb = Table.grid(padding=(0, 1))
    tb.add_column(width=2)               # 左缩进,与上下文的两格对齐
    tb.add_column(width=4)
    tb.add_column(width=10)              # '全期Sharpe' 显示宽 10(CJK 占 2)
    tb.add_column(width=23)
    tb.add_column(justify='right', width=9)
    tb.add_column(justify='right', width=6)
    if not tot:
        return Text('      还没有算过候选', style='dim')
    for key, name, style in LAYERS:
        n = f.get(key, 0)
        frac = n / tot
        tb.add_row('', Text(key, style=style), Text(name, style=style),
                   Text(_bar(frac), style=style),
                   Text(f'{n:,}', style=style), Text(f'{frac * 100:4.1f}%', style='dim'))
    return tb


def recent(st: dict, n: int = 8) -> Table:
    tb = Table.grid(padding=(0, 2))
    tb.add_column(width=1)               # 左缩进
    for w, j in ((8, 'left'), (9, 'right'), (7, 'right'), (7, 'right'), (9, 'right')):
        tb.add_column(width=w, justify=j)
    ev = (st.get('events') or [])[-n:]
    if not ev:
        tb.add_row('', Text('尚无批次', style='dim'), '', '', '', '')
        return tb
    for e in reversed(ev):
        pas = e.get('n_pass', 0)
        tb.add_row(
            '', Text(time.strftime('%H:%M:%S', time.localtime(e['t'])), style='dim'),
            Text(f"{e['n']:,}条"),
            Text(f"{e['dt']:.1f}s", style='dim'),
            Text(f'过{pas}', style='green' if pas else 'dim'),
            Text(f"{e['best']:+.4f}" if e.get('best') is not None else '—',
                 style='cyan' if e.get('best') is not None else 'dim'))
    return tb


def peak(st: dict) -> Text:
    b = st.get('best')
    # 表达式一律**单行截断**:换行会把整块版面顶开,而峰值表达式随时可能很长
    t = Text(overflow='ellipsis', no_wrap=True)
    t.append('  峰值  ', style='bold')
    if not b:
        t.append('还没有读数', style='dim')
        return t
    t.append(f"{b['sharpe']:+.4f}   ", style='bold cyan')
    t.append(b['expr'], style='white')
    return t


def throughput(st: dict) -> Text:
    n, busy = st.get('exprs', 0), max(st.get('busy', 0.0), 1e-9)
    t = Text()
    t.append(f'  已算 {n:,} 条 / {st.get("batches", 0)} 批 / '
             f'{st.get("served", 0)} 次请求', style='dim')
    t.append(f'   计算中吞吐 {n / busy:.0f}/s', style='white')
    t.append(f'   计算占比 {busy / max(st.get("uptime", 1), 1e-9) * 100:.0f}%', style='dim')
    lanes = st.get('lanes') or {}
    if lanes:
        t.append('\n  ' + '  '.join(f'{k} {v:,}' for k, v in sorted(lanes.items())),
                 style='dim')
    return t


def logs(st: dict, n: int = 5) -> Text:
    """服务自身的日志行(预热、异常、停止)。稳态下基本是空的 —— 有东西就是要看的东西。"""
    return Text('\n'.join('  ' + l.rstrip() for l in (st.get('log') or [])[-n:]) if n else '',
                style='dim', overflow='ellipsis', no_wrap=True)


def _where(st: dict) -> str:
    """服务在哪、用什么卡。`via` 由 client.request 在走 SSH 通道时填 = 云端主机名。"""
    at = f"☁ {st['via']}" if st.get('via') else '本机'
    return f"{at} · {st.get('device', '?')} · pid {st.get('pid')} · :{st.get('port')}"


FIXED_ROWS = 22      # 版面里不伸缩的行:边框 2 + 头 3 + 漏斗 7 + 峰值 1 + 吞吐 2 + 批次标题 1 + 空行 5 + 余量 1


def fit(console: Console, out) -> int:
    """把 console 的尺寸钉成 `out` 那个终端的**真实**尺寸,→ 行数。

    rich 在 Windows 上只问 fd 1 的尺寸。服务进程的 stdout 被接回 WSL 管道(面板写的是 CONOUT$),
    fd 1 不是终端 ⇒ rich 回退成 80×25,而 Live 按 25 行裁版面 —— 窗口明明有 40 行,面板却在
    第 25 行被截成 `…`,日志区和下边框全丢(2026-09-19 截图实锤)。尺寸必须问**实际写的那个句柄**。
    每帧重问一次 ⇒ 拖动窗口大小后版面跟着变。"""
    import os
    w, h = os.get_terminal_size(out.fileno())
    console.size = (w, h)
    return h


def render(st: dict | None, rows: int | None = None, owner: bool = False) -> Panel:
    """`rows` = 终端行数(None = 不限,如管道里的整帧输出)。版面里只有「最近批次」和「日志」两块
    可伸缩,按剩余行数分配;其余固定。超高的版面会被 Live 截断,截掉的恰好是最该看的日志区。
    `owner` = 画在服务进程自己的窗口里(只有那里 Ctrl-C 才是停服务;客户端 / 云端面板上不是)。"""
    spare = 13 if rows is None else max(1, rows - FIXED_ROWS)
    n_log = max(0, min(5, (spare - 1) // 3))
    n_ev = max(1, min(8, spare - (n_log + 1 if n_log else 0)))
    if st is None:
        return Panel(Text('\n  服务未运行 —— 起一个:\n'
                          '    python tmp/q25_funnel/run.py --daemon serve\n',
                          style='bold red'),
                     title='Q25 回测服务', border_style='red')
    if not st.get('ok'):
        return Panel(Text(f'\n  服务应答异常:{st.get("error", st)}\n\n'
                          '  「未知 op stats」= 服务是改面板之前起的旧进程,重起一次即可\n',
                          style='bold yellow'),
                     title='Q25 回测服务', border_style='yellow')
    if st.get('phase') == 'warm':
        body = Group(Text(f"\n  预热中 · {st.get('warm', '')}", style='bold yellow'),
                     Text('  (装载面板 + 建 CUDA context + 上传显存,只此一次)\n',
                          style='dim'),
                     logs(st, 8))
        return Panel(body, title=f'[bold]Q25 回测服务[/]  {_where(st)}', border_style='yellow',
                     subtitle=time.strftime('%H:%M:%S'))
    tot = sum((st.get('funnel') or {}).values())
    body = Group(
        header(st), Text(),
        Text(f'  门漏斗(累计 {tot:,} 条)', style='bold'), funnel(st), Text(),
        peak(st), Text(),
        throughput(st), Text(),
        Text('  最近批次', style='bold'), recent(st, n_ev),
        *((Text(), logs(st, n_log)) if n_log else ()))
    return Panel(body, border_style='green' if st.get('cur') else 'blue',
                 title=f"[bold]Q25 回测服务[/]  {_where(st)}",
                 subtitle=time.strftime('%H:%M:%S') + (' · Ctrl-C 停服务' if owner else ''))


class Feed:
    """把实时面板画在**调用方自己的终端**里 —— 就是你正敲命令的那一个。

    ## 为什么面板归客户端,不归服务端

    服务端几乎总是被脚本自动拉起的:DETACHED、无窗口(实测见 `client._spawn`)。
    它在自己进程里画的面板**没有任何人看得见**。要看就得另开一个 `--daemon dash`,
    那正是被否掉的"两个终端"。而调用方本来就要卡在 socket 上等结果 —— 那段等待时间
    正好拿来显示进度,且它的 stdout 就是你在看的地方。

    能这么做的前提是服务端的 `op=stats` 由**每连接一个 handler 线程**直接应答
    (`daemon._handle`),不进 worker 队列 ⇒ 作业正在跑的时候照样查得动,面板不会冻住。

    ## 两种输出模式,缺一不可

      · **TTY**(你手敲命令)      → `rich.Live` 原地重绘,始终一屏。
      · **非 TTY**(输出被管道接走)→ 每隔 `snap_every` 秒**追加一整帧**。

    第二支是这次修复的关键:`rich.Live` 对着管道写不出任何有用的东西 —— 它靠 ANSI
    光标控制原地覆盖,管道里既没有光标也没有屏幕。凡是"把命令输出捕获下来给人看"
    的场合(重定向到日志、CI、以及把终端输出贴给别人)全落在这一支上,而它以前是空的。
    这就是"跑了却什么都没有"的真正原因,不是服务没起来。

    短请求不刷屏:`delay` 秒内结束的作业一帧都不画。
    """

    def __init__(self, poll: float = 0.5, snap_every: float = 4.0,
                 delay: float = 0.8, out=None):
        self.poll, self.snap_every, self.delay = poll, snap_every, delay
        self.out = out or sys.stdout
        self._stop = threading.Event()
        self._th: threading.Thread | None = None
        self._started = False

    def __enter__(self) -> 'Feed':
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=3.0)
        if self._started and not self.out.isatty():
            self._frame(self._console())          # 收尾帧:留下最终漏斗形状
        return False

    def _console(self) -> Console:
        tty = self.out.isatty()
        return Console(file=self.out, width=None if tty else 84,
                       highlight=False, soft_wrap=False)

    def _frame(self, console: Console) -> None:
        import client
        console.print(render(client.request(dict(op='stats'), timeout=5.0)))

    def _loop(self) -> None:
        if self._stop.wait(self.delay):
            return                                 # 秒回的请求:一帧都不画
        try:
            import client                          # 惰性:client 也 import dash,避免环
            console = self._console()
            self._started = True
            if self.out.isatty():
                with Live(render(None), console=console, refresh_per_second=4) as live:
                    while not self._stop.is_set():
                        live.update(render(client.request(dict(op='stats'), timeout=5.0),
                                           fit(console, self.out)))
                        self._stop.wait(self.poll)
            else:
                while not self._stop.is_set():
                    self._frame(console)
                    self._stop.wait(self.snap_every)
        except Exception as e:
            # 绝不静默:面板挂了要能看出来,否则又变成"没有可视化"这种无法诊断的现象
            print(f'[dash] 面板渲染失败(回测不受影响):{type(e).__name__}: {e}',
                  file=sys.stderr, flush=True)


def main(poll: float = 0.5) -> None:
    """接上已在跑的服务(通常是被 pipeline / agent 自动拉起的那个),把面板画在本终端。
    只读 `op=stats`,退出面板不影响服务。"""
    if not sys.stdout.isatty():                  # 边界:原地重绘要真终端(Claude Code 的 `!` 前缀会截获输出)
        sys.exit('面板要在独立终端窗口里跑(PowerShell / Windows Terminal / WSL 终端);'
                 '被截获输出的环境(如 Claude Code 的 ! 前缀)画不了原地重绘面板')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import client
    console = Console()
    with Live(render(None), console=console, refresh_per_second=4) as live:
        try:
            while True:
                live.update(render(client.request(dict(op='stats'), timeout=5.0),
                                   fit(console, sys.stdout)))
                time.sleep(poll)
        except KeyboardInterrupt:
            console.print('[dim]面板退出(服务不受影响)[/]')


if __name__ == '__main__':
    main()
