/* Bounded chronological bar accounts shared by native CPU and CUDA.
 * All numerical decisions use R (FP32 or paired FP32); double is interchange.
 * A synchronized linear OHLC path defines order/mark crossing chronology. */
#pragma once
#include "order_digest.hpp"
#ifdef __CUDACC__
#define ORDER_FN __device__ __forceinline__
#else
#define ORDER_FN inline
#endif

namespace foundry_orders {
enum { TRADE_OPEN,TRADE_HIGH,TRADE_LOW,TRADE_CLOSE,VOLUME,MARK_OPEN,MARK_HIGH,MARK_LOW,MARK_CLOSE,
       FEE_BUY,FEE_SELL,FUNDING_RATE,FUNDING_MARK,MAINTENANCE,LIQ_FEE,QTY_STEP,PRICE_TICK,
       MAKER_FEE,TAKER_FEE,SPLIT_FACTOR,CASH_DIVIDEND,MIN_QUANTITY,LIMIT_UP,LIMIT_DOWN,FILL_STEP,PANELS };
enum { BUY_ALLOWED,SELL_ALLOWED,QUOTE_VALID,SESSION_START,SESSION_END,INVENTORY_T1,CASH_T1,ODD_LOT_SELL,RULES };
enum { QUANTITY,MARGIN,LAST_MARK,ENTRY,LOCKED,ASSET_VALUES };
enum { CASH,EQUITY,PREVIOUS,RESERVED,REALIZED,FEES,FUNDING,LIQUIDATION_FEES,ISOLATED_LOSS,MAX_GROSS,MEASUREMENT_RETURN,MEASUREMENT_EQUITY,BAR_TRADED_NOTIONAL,OPENING_EQUITY,UNSETTLED_CASH,CASH_VALUES };
enum { UNUSED_HIGH_ID,NEXT_SEQUENCE,LIQUIDATIONS,BANKRUPT,FIRST_LIQUIDATION,META_VALUES };
enum { ID,OWNER,ASSET,SIDE,TYPE,EXPIRY,PRIORITY,REDUCE,ATTEMPTED,RESTING,ORDER_INTS };
enum { REMAINING,PRICE,RESERVATION_PRICE,ORDER_RESERVE,TOTAL_QUANTITY,RESERVED_QUANTITY,ORDER_VALUES };
enum { ACTION,C_OWNER,C_ID,C_ASSET,C_SIDE,C_TYPE,C_EXPIRY,C_REDUCE,COMMAND_INTS };
enum { C_QUANTITY,C_PRICE,C_RESERVATION_PRICE,COMMAND_VALUES };
enum { A_TIME,A_STATUS,A_REASON,A_STAGE,A_SEQUENCE,A_OWNER,A_ID,A_ASSET,A_LIQUIDITY,AUDIT_INTS };
enum { A_QUANTITY,A_PRICE,A_FEE,A_REMAINING,AUDIT_VALUES };
enum { ACCEPTED=1,PARTIAL=2,FILLED=3,CANCELLED=4,REJECTED=5,EXPIRED=6,LIQUIDATED=7 };
enum { NONE=0,CAPACITY=1,INSUFFICIENT_FUNDS=2,NOT_TRADABLE=3,UNKNOWN_ORDER=4,INVALID_ORDER=5,
       GROSS_LIMIT=6,REDUCE_ONLY=7,EXPIRED_REASON=8,LIQUIDATION_REASON=9,UNSETTLED=10,CORPORATE_ACTION=11 };
enum { CASH_ACCOUNT=0,ISOLATED_ACCOUNT=1,CROSS_ACCOUNT=2 };
enum { NO_LIQUIDITY=0,MAKER=1,TAKER=2 };
enum { TURNOVER,GROSS_SUM,NET_SUM,MAX_ASSET,FILL_COUNT,MEASUREMENTS,ACTIVITY_VALUES };
enum { COMMAND_STAGE=1,OPEN_STAGE=2,FIRST_STAGE=3,SECOND_STAGE=4,CLOSE_STAGE=5,FUNDING_STAGE=6,LIQUIDATION_STAGE=7 };

template<class R> ORDER_FN R absolute(R x) {return x<R(0.)?-x:x;}
template<class R> ORDER_FN R minimum(R a,R b) {return a<b?a:b;}
template<class R> ORDER_FN R maximum(R a,R b) {return a>b?a:b;}
// Central rule-grid tolerances scale with the selected arithmetic width.
template<class R> ORDER_FN R rule_tolerance() {return R(sizeof(R)==sizeof(float)?0x1p-20:0x1p-44);}
template<class R> ORDER_FN R quantize(R amount,R step) {
    if(step<=R(0.))return amount;
    const R units=amount/step;
    long long count=(long long)double(units);
    if(R(double(count+1))-units<=rule_tolerance<R>()*maximum(R(1.),absolute(units)))++count;
    return R(double(count))*step;
}
template<class R> ORDER_FN bool on_grid(R amount,R step) {
    if(step<=R(0.))return true;
    if(amount/step>R(0x1p62))return false;
    return absolute(quantize(amount,step)-amount)<=rule_tolerance<R>()*maximum(absolute(amount),step);
}

struct Input {
    const double* panels;
    const unsigned char* rules;
    long long assets,capacity,commands,audit_capacity,time,owners;
    int margin_mode,path,measured;
    double leverage,gross_limit,participation;
};

struct State {
    double* assets;
    double* cash;
    long long* meta;
    long long* order_ints;
    double* order_values;
    const long long* command_ints;
    const double* command_values;
    long long* audit_ints;
    double* audit_values;
    long long* audit_count;
    double* owner_quantity;
    long long* owner_highwater;
    const double* targets;
    const unsigned char* target_updates;
    const long long* target_units;
    const double* coefficients;
    double* pending_targets;
    unsigned char* target_dirty;
    double* activity;
    unsigned int* digest;
    double* owner_locked;
};

template<class R> struct Account {
    Input in;
    State st;
    ORDER_FN R owned(long long owner,long long a) const {return R(st.owner_quantity[owner*in.assets+a]);}
    ORDER_FN void set_owned(long long owner,long long a,R value) {st.owner_quantity[owner*in.assets+a]=double(value);}
    ORDER_FN R locked(long long owner,long long a) const {return R(st.owner_locked[owner*in.assets+a]);}
    ORDER_FN void set_locked(long long owner,long long a,R value) {st.owner_locked[owner*in.assets+a]=double(value);}
    ORDER_FN bool derivative() const {return in.margin_mode!=CASH_ACCOUNT;}
    ORDER_FN bool cross() const {return in.margin_mode==CROSS_ACCOUNT;}
    ORDER_FN bool rule(int f,long long a) const {return in.rules[f*in.assets+a]!=0;}
    ORDER_FN void hash_integer(long long value) {foundry_order_digest::integer(st.digest,(unsigned long long)value);}
    ORDER_FN void hash_number(R value) {foundry_order_digest::number(st.digest,double(value));}
    ORDER_FN R panel(int f,long long a) const {return R(in.panels[f*in.assets+a]);}
    ORDER_FN R asset(long long a,int f) const {return R(st.assets[a*ASSET_VALUES+f]);}
    ORDER_FN void set_asset(long long a,int f,R value) {st.assets[a*ASSET_VALUES+f]=double(value);}
    ORDER_FN R cash(int f=CASH) const {return R(st.cash[f]);}
    ORDER_FN void set_cash(int f,R value) {st.cash[f]=double(value);}
    ORDER_FN long long& oi(long long q,int f) {return st.order_ints[q*ORDER_INTS+f];}
    ORDER_FN R ov(long long q,int f) const {return R(st.order_values[q*ORDER_VALUES+f]);}
    ORDER_FN void set_ov(long long q,int f,R value) {st.order_values[q*ORDER_VALUES+f]=double(value);}
    ORDER_FN void audit(long long owner,long long id,long long a,int status,int reason,int stage,
                        R quantity=R(0.),R price=R(0.),R fee=R(0.),R remaining=R(0.),int liquidity=NO_LIQUIDITY) {
        hash_integer(status);hash_integer(reason);hash_integer(stage);hash_integer(owner);hash_integer(id);hash_integer(a);
        hash_number(quantity);hash_number(price);hash_number(fee);hash_number(remaining);
        hash_integer(liquidity);
        const long long row=(*st.audit_count)++;
        long long* i=st.audit_ints+row*AUDIT_INTS;
        double* v=st.audit_values+row*AUDIT_VALUES;
        i[A_TIME]=in.time;i[A_STATUS]=status;i[A_REASON]=reason;i[A_STAGE]=stage;
        i[A_SEQUENCE]=++st.meta[NEXT_SEQUENCE];i[A_OWNER]=owner;i[A_ID]=id;i[A_ASSET]=a;
        i[A_LIQUIDITY]=liquidity;
        v[A_QUANTITY]=double(quantity);v[A_PRICE]=double(price);v[A_FEE]=double(fee);v[A_REMAINING]=double(remaining);
    }
    ORDER_FN R reserved(long long except=-1) const {
        R value(0.);
        for(long long q=0;q<in.capacity;++q)
            if(q!=except&&st.order_ints[q*ORDER_INTS+ID])value+=ov(q,ORDER_RESERVE);
        return value;
    }
    ORDER_FN R reserved_quantity(long long a,int side,long long except=-1,long long owner=-1) const {
        R value(0.);
        for(long long q=0;q<in.capacity;++q) {
            const long long* i=st.order_ints+q*ORDER_INTS;
            if(q!=except&&i[ID]&&i[ASSET]==a&&i[SIDE]==side&&(owner<0||i[OWNER]==owner))value+=ov(q,RESERVED_QUANTITY);
        }
        return value;
    }
    ORDER_FN R equity() const {
        R value=cash();
        for(long long a=0;a<in.assets;++a)
            value+=derivative()?asset(a,MARGIN):asset(a,QUANTITY)*asset(a,LAST_MARK);
        return value;
    }
    ORDER_FN R gross() const {
        R value(0.);
        for(long long a=0;a<in.assets;++a)value+=absolute(asset(a,QUANTITY))*asset(a,LAST_MARK);
        return value;
    }
    ORDER_FN R available() const {
        return cross()?equity()-gross()/R(in.leverage):cash()-cash(UNSETTLED_CASH);
    }
    ORDER_FN R fee_rate(long long a,int side,int liquidity) const {
        return panel(side>0?FEE_BUY:FEE_SELL,a)+panel(liquidity==MAKER?MAKER_FEE:TAKER_FEE,a);
    }
    ORDER_FN R reserve_rate(long long a,int side) const {
        return maximum(R(0.),maximum(fee_rate(a,side,MAKER),fee_rate(a,side,TAKER)));
    }
    ORDER_FN void refresh() {
        const R eq=equity(),g=gross();
        set_cash(EQUITY,eq);set_cash(RESERVED,reserved());
        if(eq>R(0.))set_cash(MAX_GROSS,maximum(cash(MAX_GROSS),g/eq));
        else if(g>R(0.))set_cash(MAX_GROSS,R(1.)/R(0.));
    }
    ORDER_FN void terminal(long long q,int status,int reason,int stage) {
        audit(oi(q,OWNER),oi(q,ID),oi(q,ASSET),status,reason,stage,R(0.),R(0.),R(0.),ov(q,REMAINING));
        oi(q,ID)=0;set_ov(q,ORDER_RESERVE,R(0.));
    }
    ORDER_FN R required_reserve(long long a,int side,R quantity,R estimate,R closing) const {
        const R fee=quantity*estimate*reserve_rate(a,side);
        if(!derivative())return quantity*estimate+fee;
        if(cross())return maximum((quantity-R(2.)*closing)*estimate/R(in.leverage)+fee,R(0.));
        const R owned=asset(a,QUANTITY);
        const R released=owned!=R(0.)?asset(a,MARGIN)*closing/absolute(owned):R(0.);
        const R closing_pnl=(owned>R(0.)?R(1.):R(-1.))*closing*(estimate-asset(a,LAST_MARK));
        return maximum((quantity-closing)*estimate/R(in.leverage)+fee-released-closing_pnl,R(0.));
    }
    ORDER_FN void liquidate(long long a,int stage) {
        const R q=asset(a,QUANTITY),balance=asset(a,MARGIN),mark=asset(a,LAST_MARK);
        const R nominal=panel(LIQ_FEE,a)*absolute(q)*mark;
        const R fee=minimum(maximum(balance,R(0.)),nominal);
        set_cash(CASH,cash()+maximum(balance-nominal,R(0.)));
        set_cash(LIQUIDATION_FEES,cash(LIQUIDATION_FEES)+fee);
        set_cash(ISOLATED_LOSS,cash(ISOLATED_LOSS)+maximum(-balance,R(0.)));
        set_asset(a,QUANTITY,R(0.));set_asset(a,MARGIN,R(0.));set_asset(a,ENTRY,R(0.));
        ++st.meta[LIQUIDATIONS];
        if(st.meta[FIRST_LIQUIDATION]<0)st.meta[FIRST_LIQUIDATION]=in.time;
        audit(-1,0,a,LIQUIDATED,LIQUIDATION_REASON,stage,absolute(q),mark,fee);
        R claims(0.);for(long long owner=0;owner<in.owners;++owner)claims+=absolute(owned(owner,a));
        for(long long owner=0;owner<in.owners;++owner) {
            const R contribution=owned(owner,a);
            if(contribution!=R(0.))audit(owner,0,a,LIQUIDATED,LIQUIDATION_REASON,stage,contribution,mark,
                claims>R(0.)?fee*absolute(contribution)/claims:R(0.));
            set_owned(owner,a,R(0.));st.target_dirty[owner*in.assets+a]=0;
        }
        for(long long order=0;order<in.capacity;++order)
            if(oi(order,ID)&&oi(order,ASSET)==a&&(oi(order,REDUCE)||oi(order,ID)<0))
                terminal(order,CANCELLED,REDUCE_ONLY,stage);
    }
    ORDER_FN R maintenance() const {
        R total(0.);
        for(long long a=0;a<in.assets;++a)total+=panel(MAINTENANCE,a)*absolute(asset(a,QUANTITY))*asset(a,LAST_MARK);
        return total;
    }
    ORDER_FN void liquidate_cross(int stage) {
        // Cancel orders first, then close the pooled account at synchronized marks.
        for(long long order=0;order<in.capacity;++order)
            if(oi(order,ID))terminal(order,CANCELLED,LIQUIDATION_REASON,stage);
        for(long long a=0;a<in.assets;++a) {
            const R q=asset(a,QUANTITY);
            if(q==R(0.))continue;
            const R price=asset(a,LAST_MARK),fee=absolute(q)*price*panel(LIQ_FEE,a);
            set_cash(CASH,cash()+asset(a,MARGIN)-fee);
            set_cash(REALIZED,cash(REALIZED)+asset(a,MARGIN));
            set_cash(LIQUIDATION_FEES,cash(LIQUIDATION_FEES)+fee);
            ++st.meta[LIQUIDATIONS];
            if(st.meta[FIRST_LIQUIDATION]<0)st.meta[FIRST_LIQUIDATION]=in.time;
            audit(-1,0,a,LIQUIDATED,LIQUIDATION_REASON,stage,absolute(q),price,fee);
            R claims(0.);for(long long owner=0;owner<in.owners;++owner)claims+=absolute(owned(owner,a));
            for(long long owner=0;owner<in.owners;++owner) {
                const R contribution=owned(owner,a);
                if(contribution!=R(0.))audit(owner,0,a,LIQUIDATED,LIQUIDATION_REASON,stage,contribution,price,
                    claims>R(0.)?fee*absolute(contribution)/claims:R(0.));
                set_owned(owner,a,R(0.));st.target_dirty[owner*in.assets+a]=0;
            }
            set_asset(a,QUANTITY,R(0.));set_asset(a,MARGIN,R(0.));set_asset(a,ENTRY,R(0.));
        }
        // A gap shortfall belongs to the pooled account, never to individual legs.
        set_cash(ISOLATED_LOSS,cash(ISOLATED_LOSS)+maximum(-cash(),R(0.)));
        set_cash(CASH,maximum(cash(),R(0.)));
        refresh();
    }
    ORDER_FN void mark(long long a,R price) {
        if(derivative())set_asset(a,MARGIN,asset(a,MARGIN)+asset(a,QUANTITY)*(price-asset(a,LAST_MARK)));
        set_asset(a,LAST_MARK,price);
    }
    ORDER_FN void risk(int stage) {
        refresh();
        if(!derivative())return;
        if(cross()) {
            if(gross()>R(0.)&&equity()<=maintenance())liquidate_cross(stage);
            return;
        }
        for(long long a=0;a<in.assets;++a)
            if(asset(a,QUANTITY)!=R(0.)&&asset(a,MARGIN)<=panel(MAINTENANCE,a)*absolute(asset(a,QUANTITY))*asset(a,LAST_MARK))
                liquidate(a,stage);
    }
    ORDER_FN void command_values(const long long* c,const double* v,bool target=false) {
        if(!c[ACTION])return;
        hash_integer(100);for(int f=0;f<COMMAND_INTS;++f)hash_integer(c[f]);
        for(int f=0;f<COMMAND_VALUES;++f)foundry_order_digest::number(st.digest,v[f]);
        long long found=-1,slot=-1;
        for(long long q=0;q<in.capacity;++q) {
            if(oi(q,ID)==c[C_ID]&&oi(q,OWNER)==c[C_OWNER])found=q;
            if(!oi(q,ID)&&slot<0)slot=q;
        }
        if(c[ACTION]!=1) {
            if(found<0||oi(found,OWNER)!=c[C_OWNER]) {
                audit(c[C_OWNER],c[C_ID],c[C_ASSET],REJECTED,UNKNOWN_ORDER,COMMAND_STAGE);return;
            }
            if(c[ACTION]==2) {terminal(found,CANCELLED,NONE,COMMAND_STAGE);return;}
            slot=found;
        } else {
            if(!target&&c[C_ID]<=st.owner_highwater[c[C_OWNER]]) {
                audit(c[C_OWNER],c[C_ID],c[C_ASSET],REJECTED,INVALID_ORDER,COMMAND_STAGE);return;
            }
            if(!target)st.owner_highwater[c[C_OWNER]]=c[C_ID];
        }
        const long long a=c[C_ASSET];
        const R quantity(v[C_QUANTITY]),price(v[C_PRICE]),estimate(v[C_RESERVATION_PRICE]);
        const R owner_position=owned(c[C_OWNER],a);
        const bool odd_sell=c[C_SIDE]<0&&rule(ODD_LOT_SELL,a)&&quantity==owner_position;
        int reason=NONE;
        if(slot<0)reason=CAPACITY;
        else if(c[C_EXPIRY]>=0&&c[C_EXPIRY]<in.time)reason=EXPIRED_REASON;
        else if(!in.rules[QUOTE_VALID*in.assets+a]||!in.rules[(c[C_SIDE]>0?BUY_ALLOWED:SELL_ALLOWED)*in.assets+a])reason=NOT_TRADABLE;
        else if(!odd_sell&&(!on_grid(quantity,panel(QTY_STEP,a))||quantity<panel(MIN_QUANTITY,a)))reason=INVALID_ORDER;
        else if(c[C_TYPE]!=1&&!on_grid(price,panel(PRICE_TICK,a)))reason=INVALID_ORDER;
        else if(c[C_TYPE]!=1&&((panel(LIMIT_UP,a)>R(0.)&&price>panel(LIMIT_UP,a))
                ||(panel(LIMIT_DOWN,a)>R(0.)&&price<panel(LIMIT_DOWN,a))))reason=INVALID_ORDER;
        const bool reducing=(!derivative()&&c[C_SIDE]<0)||c[C_REDUCE];
        const R owned=asset(a,QUANTITY);
        const R available=owned*R(double(c[C_SIDE]))<R(0.)
            ?maximum(absolute(owned)-reserved_quantity(a,int(c[C_SIDE]),slot),R(0.)):R(0.);
        const R owner_available=owner_position*R(double(c[C_SIDE]))<R(0.)
            ?maximum(absolute(owner_position)-reserved_quantity(a,int(c[C_SIDE]),slot,c[C_OWNER]),R(0.)):R(0.);
        const R closing=minimum(quantity,available);
        R reserve(0.);
        if(!reason&&reducing) {
            if((owned==R(0.)||owned*R(double(c[C_SIDE]))>=R(0.))
                    ||quantity>available||quantity>owner_available)reason=REDUCE_ONLY;
            else if(!derivative()&&c[C_SIDE]<0&&quantity>owner_available-locked(c[C_OWNER],a))reason=UNSETTLED;
        } else if(!reason) {
            reserve=required_reserve(a,int(c[C_SIDE]),quantity,estimate,closing);
            if(reserve>this->available()-reserved(slot)&&(!cross()||closing<quantity)) {
                if(target)reserve=maximum(this->available()-reserved(slot),R(0.));else reason=INSUFFICIENT_FUNDS;
            }
        }
        if(reason) {
            if(target&&(reason==UNSETTLED||reason==NOT_TRADABLE))st.target_dirty[c[C_OWNER]*in.assets+a]=1;
            audit(c[C_OWNER],c[C_ID],a,REJECTED,reason,COMMAND_STAGE,R(0.),R(0.),R(0.),quantity);return;
        }
        oi(slot,ID)=c[C_ID];oi(slot,OWNER)=c[C_OWNER];oi(slot,ASSET)=a;oi(slot,SIDE)=c[C_SIDE];
        oi(slot,TYPE)=c[C_TYPE];oi(slot,EXPIRY)=c[C_EXPIRY];oi(slot,REDUCE)=reducing;
        oi(slot,PRIORITY)=++st.meta[NEXT_SEQUENCE];
        oi(slot,ATTEMPTED)=0;
        oi(slot,RESTING)=c[C_TYPE]==2&&(c[C_SIDE]>0?panel(TRADE_OPEN,a)>price:panel(TRADE_OPEN,a)<price);
        set_ov(slot,REMAINING,quantity);set_ov(slot,PRICE,price);
        set_ov(slot,TOTAL_QUANTITY,quantity);
        set_ov(slot,RESERVED_QUANTITY,closing);
        set_ov(slot,RESERVATION_PRICE,estimate);set_ov(slot,ORDER_RESERVE,reserve);
        audit(c[C_OWNER],c[C_ID],a,ACCEPTED,NONE,COMMAND_STAGE,R(0.),R(0.),R(0.),quantity);
    }
    ORDER_FN R cash_change(long long a,int side,R amount,R price,int liquidity) const {
        const R fee=amount*price*fee_rate(a,side,liquidity);
        if(!derivative())return -R(double(side))*amount*price-fee;
        const R old=asset(a,QUANTITY),delta=R(double(side))*amount;
        const R closed=old*delta<R(0.)?minimum(absolute(old),amount):R(0.);
        const R released=old!=R(0.)?asset(a,MARGIN)*closed/absolute(old):R(0.);
        const R realized=(old>R(0.)?R(1.):R(-1.))*closed*(price-asset(a,LAST_MARK));
        return released+realized-(cross()?R(0.):(amount-closed)*price/R(in.leverage))-fee;
    }
    ORDER_FN bool feasible(long long order,R amount,R price,int liquidity,int& reason) const {
        const long long a=st.order_ints[order*ORDER_INTS+ASSET];
        const int side=int(st.order_ints[order*ORDER_INTS+SIDE]);
        const R change=cash_change(a,side,amount,price,liquidity);
        if(!cross()&&cash()+change-cash(UNSETTLED_CASH)<reserved(order)) {reason=INSUFFICIENT_FUNDS;return false;}
        const R fee=amount*price*fee_rate(a,side,liquidity);
        const R eq=equity()+R(double(side))*amount*(asset(a,LAST_MARK)-price)-fee;
        const R old=asset(a,QUANTITY),next=old+R(double(side))*amount;
        const R g=gross()+(absolute(next)-absolute(old))*asset(a,LAST_MARK);
        if(cross()&&absolute(next)>absolute(old)&&eq-g/R(in.leverage)<reserved(order)) {
            reason=INSUFFICIENT_FUNDS;return false;
        }
        // Reductions remain possible when adverse moves breach the limit.
        if(absolute(next)>absolute(old)&&(eq<=R(0.)||g>R(in.gross_limit)*eq)) {
            reason=GROSS_LIMIT;return false;
        }
        return true;
    }
    ORDER_FN void fill(long long order,R price,R* capacity,int stage) {
        const long long a=oi(order,ASSET);const int side=int(oi(order,SIDE));
        const int liquidity=oi(order,TYPE)==2&&oi(order,RESTING)?MAKER:TAKER;
        const R original=ov(order,REMAINING);
        const long long owner=oi(order,OWNER);
        if(!in.rules[(side>0?BUY_ALLOWED:SELL_ALLOWED)*in.assets+a])return;
        if((panel(LIMIT_UP,a)>R(0.)&&price>panel(LIMIT_UP,a))
                ||(panel(LIMIT_DOWN,a)>R(0.)&&price<panel(LIMIT_DOWN,a)))return;
        if(oi(order,REDUCE)) {
            const R owned=asset(a,QUANTITY);
            if(owned==R(0.)||owned*R(double(side))>=R(0.)||this->owned(owner,a)*R(double(side))>=R(0.)) {
                terminal(order,REJECTED,REDUCE_ONLY,stage);return;
            }
        }
        R amount=minimum(original,capacity[a]);
        const R owned=asset(a,QUANTITY);
        const R other_closing=reserved_quantity(a,side,order);
        const R available_closing=maximum(absolute(owned)-other_closing,R(0.));
        const bool closing_locked=owned*R(double(side))<R(0.)&&other_closing>R(0.);
        if(oi(order,REDUCE)||closing_locked)amount=minimum(amount,available_closing);
        if(oi(order,REDUCE))amount=minimum(amount,maximum(absolute(this->owned(owner,a))-reserved_quantity(a,side,order,owner),R(0.)));
        if(!derivative()&&side<0)amount=minimum(amount,maximum(this->owned(owner,a)-locked(owner,a)-reserved_quantity(a,side,order,owner),R(0.)));
        amount=quantize(amount,panel(FILL_STEP,a));
        if(amount<=R(0.))return;
        int reason=NONE;
        const bool full=feasible(order,amount,price,liquidity,reason);
        if(!full) {
            R low(0.),high=amount;
            for(int i=0;i<48;++i) {
                const R mid=(low+high)*R(.5);int ignored=NONE;
                if(mid==low||mid==high)break;
                if(feasible(order,mid,price,liquidity,ignored))low=mid;else high=mid;
            }
            amount=quantize(low,panel(FILL_STEP,a));
        }
        if(amount>R(0.)) {
            const R old=asset(a,QUANTITY),delta=R(double(side))*amount,next=old+delta;
            set_cash(BAR_TRADED_NOTIONAL,cash(BAR_TRADED_NOTIONAL)+amount*price);
            st.activity[FILL_COUNT]=double(R(st.activity[FILL_COUNT])+R(1.));
            set_owned(owner,a,this->owned(owner,a)+delta);
            const R fee=amount*price*fee_rate(a,side,liquidity);
            const R change=cash_change(a,side,amount,price,liquidity);
            if(derivative()) {
                const R closed=old*delta<R(0.)?minimum(absolute(old),amount):R(0.);
                const R kept=old!=R(0.)?asset(a,MARGIN)*(R(1.)-closed/absolute(old)):R(0.);
                const R opened=amount-closed;
                set_asset(a,MARGIN,kept+(cross()?R(0.):opened*price/R(in.leverage))+R(double(side))*opened*(asset(a,LAST_MARK)-price));
                set_cash(REALIZED,cash(REALIZED)+(old>R(0.)?R(1.):R(-1.))*closed*(price-asset(a,ENTRY)));
            }
            if(!derivative()&&side>0&&rule(INVENTORY_T1,a)) {
                set_locked(owner,a,locked(owner,a)+amount);set_asset(a,LOCKED,asset(a,LOCKED)+amount);
            }
            if(!derivative()&&side<0&&rule(CASH_T1,a))set_cash(UNSETTLED_CASH,cash(UNSETTLED_CASH)+change);
            if(next==R(0.))set_asset(a,ENTRY,R(0.));
            else if(old==R(0.)||old*next<R(0.))set_asset(a,ENTRY,price);
            else if(old*delta>R(0.))set_asset(a,ENTRY,(absolute(old)*asset(a,ENTRY)+amount*price)/absolute(next));
            set_asset(a,QUANTITY,next);set_cash(CASH,cash()+change);set_cash(FEES,cash(FEES)+fee);
            capacity[a]-=amount;
            const R remaining=maximum(original-amount,R(0.));
            set_ov(order,REMAINING,remaining);
            const R now_owned=asset(a,QUANTITY);
            const R now_closing=now_owned*R(double(side))<R(0.)
                ?minimum(remaining,maximum(absolute(now_owned)-reserved_quantity(a,side,order),R(0.))):R(0.);
            set_ov(order,RESERVED_QUANTITY,now_closing);
            set_ov(order,ORDER_RESERVE,oi(order,REDUCE)?R(0.):required_reserve(a,side,remaining,ov(order,RESERVATION_PRICE),now_closing));
            audit(oi(order,OWNER),oi(order,ID),a,remaining==R(0.)?FILLED:PARTIAL,NONE,stage,amount,price,fee,remaining,liquidity);
            if(remaining==R(0.)) {oi(order,ID)=0;set_ov(order,ORDER_RESERVE,R(0.));}
        }
        if(!full&&oi(order,ID))terminal(order,REJECTED,reason,stage);
        else if(oi(order,ID)&&available()<reserved())terminal(order,REJECTED,INSUFFICIENT_FUNDS,stage);
        else if(oi(order,ID)&&oi(order,REDUCE)&&asset(a,QUANTITY)==R(0.))terminal(order,REJECTED,REDUCE_ONLY,stage);
        risk(stage);
    }
    ORDER_FN void targets(long long owner) {
        const R basis=equity();
        for(long long a=0;a<in.assets;++a) {
            const long long index=owner*in.assets+a;
            if(st.target_updates[index]) {
                hash_integer(101);hash_integer(owner);hash_integer(a);hash_integer(st.target_units[owner]);
                foundry_order_digest::number(st.digest,st.targets[index]);
                foundry_order_digest::number(st.digest,st.coefficients[owner]);
                st.pending_targets[index]=st.targets[index];st.target_dirty[index]=1;
                for(long long q=0;q<in.capacity;++q)
                    if(oi(q,ID)<0&&oi(q,OWNER)==owner&&oi(q,ASSET)==a)terminal(q,CANCELLED,NONE,COMMAND_STAGE);
            }
            if(!st.target_dirty[index]||!in.rules[QUOTE_VALID*in.assets+a])continue;
            const R price=panel(TRADE_OPEN,a);
            R desired=R(st.pending_targets[index])*R(st.coefficients[owner]);
            if(st.target_units[owner]==0)desired*=basis/price;
            else if(st.target_units[owner]==2)desired/=price;
            const R delta=desired-owned(owner,a);
            st.target_dirty[index]=0;
            const bool odd_sell=delta< R(0.)&&rule(ODD_LOT_SELL,a)&&absolute(delta)==owned(owner,a);
            const R quantity=odd_sell?absolute(delta):quantize(absolute(delta),panel(QTY_STEP,a));
            if(quantity<=R(0.))continue;
            const long long c[COMMAND_INTS]={1,owner,-(a+1),a,delta>R(0.)?1:-1,1,-1,0};
            const double v[COMMAND_VALUES]={double(quantity),0.,double(price)};
            command_values(c,v,true);
        }
    }
    ORDER_FN R path_value(bool mark_price,long long a,int point) const {
        int field;
        if(!point)field=mark_price?MARK_OPEN:TRADE_OPEN;
        else if(point==3)field=mark_price?MARK_CLOSE:TRADE_CLOSE;
        else {
            const bool high=in.path?point==1:point==2;
            field=mark_price?(high?MARK_HIGH:MARK_LOW):(high?TRADE_HIGH:TRADE_LOW);
        }
        return panel(field,a);
    }
    ORDER_FN void session_actions() {
        if(rule(CASH_T1,0)&&rule(SESSION_START,0))set_cash(UNSETTLED_CASH,R(0.));
        for(long long a=0;a<in.assets;++a) {
            if(rule(SESSION_START,a)) {
                set_asset(a,LOCKED,R(0.));
                for(long long owner=0;owner<in.owners;++owner)set_locked(owner,a,R(0.));
            }
            const R split=panel(SPLIT_FACTOR,a),dividend=panel(CASH_DIVIDEND,a);
            if(split==R(1.)&&dividend==R(0.))continue;
            // The declared payment is per pre-event share; prices remain unadjusted.
            set_cash(CASH,cash()+asset(a,QUANTITY)*dividend);
            for(long long q=0;q<in.capacity;++q)if(oi(q,ID)&&oi(q,ASSET)==a) {
                if(oi(q,ID)<0)st.target_dirty[oi(q,OWNER)*in.assets+a]=1;
                terminal(q,CANCELLED,CORPORATE_ACTION,OPEN_STAGE);
            }
            set_asset(a,QUANTITY,asset(a,QUANTITY)*split);
            set_asset(a,LOCKED,asset(a,LOCKED)*split);
            set_asset(a,ENTRY,asset(a,ENTRY)/split);
            set_asset(a,LAST_MARK,asset(a,LAST_MARK)/split);
            for(long long owner=0;owner<in.owners;++owner) {
                set_owned(owner,a,owned(owner,a)*split);set_locked(owner,a,locked(owner,a)*split);
                if(st.target_units[owner]==1)st.pending_targets[owner*in.assets+a]=double(R(st.pending_targets[owner*in.assets+a])*split);
            }
        }
    }
    ORDER_FN void run(R* capacity) {
        *st.audit_count=0;
        hash_integer(in.time);
        set_cash(BAR_TRADED_NOTIONAL,R(0.));set_cash(OPENING_EQUITY,R(0.));
        if(st.meta[BANKRUPT]) {
            for(long long k=0;k<in.commands;++k) {
                const long long* c=st.command_ints+k*COMMAND_INTS;
                if(c[ACTION])audit(c[C_OWNER],c[C_ID],c[C_ASSET],REJECTED,INSUFFICIENT_FUNDS,COMMAND_STAGE);
            }
            if(in.measured) {
                set_cash(MEASUREMENT_RETURN,R(0.));set_cash(MEASUREMENT_EQUITY,R(0.));
                st.activity[MEASUREMENTS]=double(R(st.activity[MEASUREMENTS])+R(1.));
            }
            return;
        }
        session_actions();
        for(long long q=0;q<in.capacity;++q)oi(q,ATTEMPTED)=0;
        for(long long a=0;a<in.assets;++a) {
            mark(a,panel(MARK_OPEN,a));
            capacity[a]=panel(VOLUME,a)*R(in.participation);
        }
        risk(OPEN_STAGE);
        if(derivative()) {
            for(long long a=0;a<in.assets;++a) {
                const R flow=-asset(a,QUANTITY)*panel(FUNDING_MARK,a)*panel(FUNDING_RATE,a);
                if(cross())set_cash(CASH,cash()+flow);
                else set_asset(a,MARGIN,asset(a,MARGIN)+flow);
                set_cash(FUNDING,cash(FUNDING)+flow);
            }
            risk(FUNDING_STAGE);
        }
        set_cash(OPENING_EQUITY,equity());
        for(long long q=0;q<in.capacity;++q)
            if(oi(q,ID)&&oi(q,EXPIRY)>=0&&oi(q,EXPIRY)<in.time)terminal(q,EXPIRED,EXPIRED_REASON,COMMAND_STAGE);
        for(long long owner=0;owner<in.owners;++owner) {
            targets(owner);
            for(long long k=0;k<in.commands;++k) {
                const long long* command=st.command_ints+k*COMMAND_INTS;
                if(command[C_OWNER]==owner)command_values(command,st.command_values+k*COMMAND_VALUES);
            }
        }
        // At each segment, find the next chronological crossing across all assets.
        for(int segment=0;segment<3;++segment) {
            R cursor(0.);
            while(true) {
                R best(2.);long long chosen=-1,liquidation=-1;bool stop=false;
                if(cross()&&gross()>R(0.)) {
                    const R margin_now=equity()-maintenance();
                    R margin_end=equity();
                    for(long long a=0;a<in.assets;++a) {
                        const R end=path_value(true,a,segment+1),q=asset(a,QUANTITY);
                        margin_end+=q*(end-asset(a,LAST_MARK))-panel(MAINTENANCE,a)*absolute(q)*end;
                    }
                    if(margin_end<=R(0.)) {
                        best=margin_now<=R(0.)?cursor:cursor+(R(1.)-cursor)*margin_now/(margin_now-margin_end);
                        liquidation=in.assets;
                    }
                }
                for(long long a=0;a<in.assets;++a) {
                    if(in.margin_mode!=ISOLATED_ACCOUNT||asset(a,QUANTITY)==R(0.))continue;
                    const R end=path_value(true,a,segment+1),current=asset(a,LAST_MARK);
                    const R q=asset(a,QUANTITY),bal=asset(a,MARGIN),maintenance=panel(MAINTENANCE,a)*absolute(q);
                    const R margin_end=bal+q*(end-current)-maintenance*end;
                    if(margin_end<=R(0.)) {
                        const R margin_now=bal-maintenance*current;
                        const R fraction=margin_now<=R(0.)?cursor:cursor+(R(1.)-cursor)*margin_now/(margin_now-margin_end);
                        if(fraction<best) {best=fraction;chosen=-1;liquidation=a;}
                    }
                }
                for(long long order=0;order<in.capacity;++order) {
                    if(!oi(order,ID)||oi(order,ATTEMPTED))continue;
                    const long long a=oi(order,ASSET);const int side=int(oi(order,SIDE)),type=int(oi(order,TYPE));
                    if(!in.rules[QUOTE_VALID*in.assets+a]||!in.rules[(side>0?BUY_ALLOWED:SELL_ALLOWED)*in.assets+a])continue;
                    const R start=path_value(false,a,segment),end=path_value(false,a,segment+1);
                    const R here=start+(end-start)*cursor,level=ov(order,PRICE);
                    R fraction(2.);
                    bool trigger=false;
                    if(type==1) {if(capacity[a]>R(0.))fraction=cursor;}
                    else if(type==2) {
                        const bool now=side>0?here<=level:here>=level;
                        const bool later=side>0?end<=level:end>=level;
                        if(capacity[a]>R(0.)) {
                            if(now)fraction=cursor;
                            else if(later)fraction=(level-start)/(end-start);
                        }
                    } else {
                        const bool now=side>0?here>=level:here<=level;
                        const bool later=side>0?end>=level:end<=level;
                        if(now) {fraction=cursor;trigger=true;}
                        else if(later) {fraction=(level-start)/(end-start);trigger=true;}
                    }
                    if(fraction<best||(fraction==best&&liquidation<0&&(chosen<0||oi(order,PRIORITY)<oi(chosen,PRIORITY)))) {
                        best=fraction;chosen=order;liquidation=-1;stop=trigger;
                    }
                }
                if(best>R(1.))break;
                for(long long a=0;a<in.assets;++a) {
                    const R start=path_value(true,a,segment),end=path_value(true,a,segment+1);
                    mark(a,start+(end-start)*best);
                }
                cursor=best;
                const int stage=segment==0&&cursor==R(0.)?OPEN_STAGE:segment==0?FIRST_STAGE:segment==1?SECOND_STAGE:CLOSE_STAGE;
                if(liquidation>=0) {
                    if(liquidation==in.assets)liquidate_cross(stage);else liquidate(liquidation,stage);
                    continue;
                }
                const long long a=oi(chosen,ASSET);
                oi(chosen,ATTEMPTED)=1;
                if(stop)oi(chosen,TYPE)=1;
                const R start=path_value(false,a,segment),end=path_value(false,a,segment+1);
                const R price=start+(end-start)*best;
                fill(chosen,price,capacity,stage);
            }
            for(long long a=0;a<in.assets;++a)mark(a,path_value(true,a,segment+1));
            risk(segment==0?FIRST_STAGE:segment==1?SECOND_STAGE:CLOSE_STAGE);
        }
        for(long long order=0;order<in.capacity;++order)if(oi(order,ID)) {
            if(rule(SESSION_END,oi(order,ASSET))) {
                if(oi(order,ID)<0)st.target_dirty[oi(order,OWNER)*in.assets+oi(order,ASSET)]=1;
                terminal(order,EXPIRED,EXPIRED_REASON,CLOSE_STAGE);
            } else if(oi(order,TYPE)==2)oi(order,RESTING)=1;
        }
        refresh();
        if(cash(OPENING_EQUITY)>R(0.))st.activity[TURNOVER]=double(R(st.activity[TURNOVER])+cash(BAR_TRADED_NOTIONAL)/cash(OPENING_EQUITY));
        if(cash(EQUITY)<=R(0.)) {
            st.meta[BANKRUPT]=1;
            for(long long q=0;q<in.capacity;++q)if(oi(q,ID))terminal(q,REJECTED,INSUFFICIENT_FUNDS,CLOSE_STAGE);
            set_cash(CASH,R(0.));
            set_cash(UNSETTLED_CASH,R(0.));
            for(long long a=0;a<in.assets;++a) {set_asset(a,QUANTITY,R(0.));set_asset(a,MARGIN,R(0.));set_asset(a,ENTRY,R(0.));set_asset(a,LOCKED,R(0.));}
            for(long long owner=0;owner<in.owners;++owner)for(long long a=0;a<in.assets;++a) {set_owned(owner,a,R(0.));set_locked(owner,a,R(0.));}
            refresh();
        }
        hash_integer(102);for(long long f=0;f<CASH_VALUES;++f)hash_number(cash(int(f)));
        for(long long a=0;a<in.assets;++a)for(int f=0;f<ASSET_VALUES;++f)hash_number(asset(a,f));
        for(long long owner=0;owner<in.owners;++owner)for(long long a=0;a<in.assets;++a) {hash_number(owned(owner,a));hash_number(locked(owner,a));}
        if(in.measured) {
            const R nav=cash(EQUITY),previous=cash(PREVIOUS);
            set_cash(MEASUREMENT_RETURN,previous>R(0.)?nav/previous-R(1.):R(0.));
            set_cash(MEASUREMENT_EQUITY,nav);set_cash(PREVIOUS,nav);
            R net(0.),maximum_asset(0.);for(long long a=0;a<in.assets;++a) {
                const R exposure=nav>R(0.)?asset(a,QUANTITY)*asset(a,LAST_MARK)/nav:R(0.);
                net+=exposure;maximum_asset=maximum(maximum_asset,absolute(exposure));
            }
            st.activity[GROSS_SUM]=double(R(st.activity[GROSS_SUM])+(nav>R(0.)?gross()/nav:R(0.)));
            st.activity[NET_SUM]=double(R(st.activity[NET_SUM])+net);st.activity[MAX_ASSET]=double(maximum(R(st.activity[MAX_ASSET]),maximum_asset));
            st.activity[MEASUREMENTS]=double(R(st.activity[MEASUREMENTS])+R(1.));
        }
    }
};
template<class R> ORDER_FN void sample_independent(const double* quantity,const double* equities,
    const double* marks,R unallocated,long long M,long long S,double* state,double* output) {
    R nav=unallocated,gross(0.),net(0.),largest(0.);
    for(long long m=0;m<M;++m)nav+=R(equities[m]);
    for(long long a=0;a<S;++a) {
        R asset_net(0.);for(long long m=0;m<M;++m) {
            const R nominal=R(quantity[m*S+a])*R(marks[a]);gross+=absolute(nominal);asset_net+=nominal;
        }
        net+=asset_net;largest=maximum(largest,absolute(asset_net));
    }
    const R g=nav>R(0.)?gross/nav:R(0.),n=nav>R(0.)?net/nav:R(0.);
    const R max_asset=nav>R(0.)?largest/nav:R(0.);
    state[0]=double(R(state[0])+g);state[1]=double(R(state[1])+n);
    state[2]=double(maximum(R(state[2]),max_asset));state[3]=double(R(state[3])+R(1.));
    state[4]=double(maximum(R(state[4]),g));
    output[0]=double(nav);output[1]=double(g);output[2]=double(n);output[3]=double(max_asset);output[4]=gross>R(0.)?1.:0.;
    const R previous(state[6]);output[5]=previous>R(0.)?double(nav/previous-R(1.)):0.;state[6]=double(nav);
}
template<class R> ORDER_FN void independent_turnover(const double* values,R unallocated,long long M,double* state) {
    R traded(0.),opening=unallocated;
    for(long long m=0;m<M;++m) {traded+=R(values[m*2]);opening+=R(values[m*2+1]);}
    if(opening>R(0.))state[5]=double(R(state[5])+traded/opening);
}
} // namespace foundry_orders
#undef ORDER_FN
