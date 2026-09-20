/* backtest/engine/engine.hpp
 *
 * Policy 抽象组合回测引擎 —— 接口头(市场 policy + 引擎声明)。
 *
 * 市场机制(成本 / carry(借券·分红))抽象成可插拔 MarketPolicy;引擎核心(engine.cpp)
 * 市场无关:MTM / rebalance 骨架 + cash 底线,在固定 hook 点回调。
 *   EquityPolicy —— US 权益:佣金(+spread)、空腿借券费(leverage_cap≤1 + cash 底线)。
 *   ASharePolicy —— 中国 A 股:印花税卖出单边 + 涨跌停 trade_block。
 * (旧 PerpPolicy(Binance 永续 funding / isolated·cross 强平 / trailing-stop / stop_scan)
 *  随 1h/5m crypto 研究栈 2026-07-25 删除;Quantiacs crypto_daily 走独立内核 qnt_bt.hpp。)
 * 纯 C++(无 Python/numpy),只碰 double*;binding(_bt_module.cpp)构造具体 policy 后传引擎。
 */
#ifndef BT_ENGINE_HPP
#define BT_ENGINE_HPP

#include <cmath>
#include <cstdint>

namespace bt {

class MarketPolicy {
public:
    virtual ~MarketPolicy() = default;

    /* 单笔 rebalance / 平仓成本。notional=|delta|·price;
     * is_sell=减仓方向(delta<0)→ 非对称成本(如 A 股印花税卖出单边)用;对称成本 policy 忽略。 */
    virtual double trade_cost(double notional, bool is_sell) const = 0;

    /* bar 末 carry:借券费(equity 空腿)/ 分红;就地改 cash。close = close[t] 行。 */
    virtual void carry(double* pos, const double* close, int64_t S, double& cash) const = 0;
};


/* ============================ Equity(US 权益)============================
 * 佣金(+半点差)对称单边成本;空腿按日借券费(年化→日);无强平(leverage_cap≤1 + cash 底线
 * 已在引擎处理)。MTM/rebalance 喂**复权** OHLC(总收益口径)。**long-short**(空腿计借券费);
 * 唯一归零路径 = 引擎 cash 底线(cash≤0,与 policy 无关),非 margin-call 强平。 */
class EquityPolicy final : public MarketPolicy {
public:
    EquityPolicy(double cost_rate, double daily_borrow)
        : cost_rate_(cost_rate), daily_borrow_(daily_borrow) {}

    double trade_cost(double notional, bool /*is_sell*/) const override {
        return notional * cost_rate_;                   /* 对称佣金(美股,买卖同费),忽略 is_sell */
    }

    void carry(double* pos, const double* close, int64_t S, double& cash) const override {
        if (daily_borrow_ <= 0.0) return;
        for (int64_t s = 0; s < S; s++)
            if (pos[s] < 0.0) cash -= std::fabs(pos[s]) * close[s] * daily_borrow_;  /* 空腿借券费 */
    }

private:
    double cost_rate_, daily_borrow_;
};


/* ============================ A-share(中国 A 股)============================
 * 佣金(buy+sell 对称)+ **印花税卖出单边**(asymmetric)+ 过户费(双边);**long-only**(无做空 →
 * 无 carry/借券)、无 margin-call 强平。daily cadence 天然满足 T+1(开盘买、最早次日开盘卖)。
 * 涨跌停 fill 限制属执行层(由权重端 tradeable-mask / valid 处理,不在 policy)。MTM/rebalance 喂 qfq 复权 OHLC。 */
class ASharePolicy final : public MarketPolicy {
public:
    ASharePolicy(double commission, double stamp_tax_sell, double transfer_fee)
        : buy_rate_(commission + transfer_fee),
          sell_rate_(commission + transfer_fee + stamp_tax_sell) {}

    double trade_cost(double notional, bool is_sell) const override {
        return notional * (is_sell ? sell_rate_ : buy_rate_);   /* 卖单含印花税 → 非对称 */
    }
    void carry(double*, const double*, int64_t, double&) const override {}   /* long-only:无借券 */

private:
    double buy_rate_, sell_rate_;
};


/* ============================ 引擎入口 ============================ */

/* 组合回测:单 bar 直达 target、rebalance @ open。outputs caller 预分配(长度 T)。
 * 成本 / carry 全经 policy;leverage_cap 仅 raw_weights=0 时用。
 * trade_block:(T,S) int8 方向冻结掩码 | NULL(=全可交易)。bit0=禁增仓(涨停/停牌),
 *   bit1=禁减仓(跌停/停牌);受限方向的 rebalance 被冻结(target←pos,不成交不计费)——
 *   A 股涨跌停 / 停牌不可成交建模,市场无关(NULL 时 equity 逐位复现)。
 * liq_at 仅由市场无关 cash 底线(cash≤0 硬停)置位。 */
void run_portfolio(
    const double* alpha, const double* close, const double* open,
    const int8_t* trade_block,
    int64_t T, int64_t S,
    double initial_cash, int64_t skip_warmup,
    double leverage_cap, int raw_weights,
    const MarketPolicy& policy,
    double* equity, double* turnover, double* cost_drag, int64_t* liq_at);

} // namespace bt

#endif /* BT_ENGINE_HPP */
