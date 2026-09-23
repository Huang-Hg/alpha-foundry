/* foundry/backtest/src/kernel.hpp —— 市场无关的日频执行内核(CPU 参考实现)。
 *
 * 基底语义 = Quantiacs qnt stats.calc_relative_return(per_asset=False)逐步复刻:昨日权重 → 今日开盘
 * 按目标市值调仓(成本按成交股数计)→ 收盘盯市。三处推广让同一递推覆盖多市场:
 *   cost_sell  卖出(份额减少)单独计价(印花税类单边成本);nullptr = 买卖同价 cost_buy。
 *   block      方向冻结位:bit0 当日禁止增加份额,bit1 当日禁止减少份额(涨跌停、停牌 = 3)。
 *              方向先按"全部可交易"时的目标市值 eq·wt/max(Σ|wt|,1) 对比当前市值判定;命中则当日不交易、
 *              市值从可操作权益中扣除、权重不参与分配,但份额隔夜保留(区别于 live==0 的收盘清零)。
 *              冻结会改变其余资产的分配比例:若另一受限资产因此朝被禁方向成交,它也冻结并重算,直到闭合
 *              —— 受限资产永不朝被禁方向成交。
 *   carry      空头隔夜持有成本率:Σ_{shares<0} carry·close·|shares| 从当日权益扣除,记入递推状态、持续生效。
 * 三者全为 nullptr 时每一步浮点运算与 qnt 复刻逐位一致;统计量(Sharpe、回撤等)由调用方从 rr 统一计算。
 *
 * 递推依赖 IEEE NaN 传染(NaN 价 → 权重归零),构建须保持 -fno-finite-math-only。
 */
#pragma once
#include <cstdint>

namespace bt {

/* 全部 (T,S) C 连续,行 = 交易日:
 *   w           目标权重,第 t 行在第 t+1 日开盘执行(非有限值当 0)
 *   open_raw    原始开盘价(NaN = 缺失);prev_close 前一日收盘价,二者之比把权重调整到开盘
 *   open_ff0    开盘价 ffill 后缺失填 0,close_ff0 同理 —— 份额、盯市与成本的计价基准
 *   cost_buy    每股成本(价格单位);cost_sell 同义,仅用于份额减少,可为 nullptr
 *   live        非 0 = 当日可交易;0 = 不交易、市值锁定,收盘后份额清零
 *   block       方向冻结位,可为 nullptr;carry 空头日持有成本率,可为 nullptr
 * rr_out:(T,) 日收益,调用方分配。 */
void simulate(const double* w,
              const double* open_raw, const double* prev_close,
              const double* open_ff0, const double* close_ff0,
              const double* cost_buy, const double* cost_sell,
              const uint8_t* live, const uint8_t* block, const double* carry,
              int64_t T, int64_t S, double* rr_out);

}  // namespace bt
