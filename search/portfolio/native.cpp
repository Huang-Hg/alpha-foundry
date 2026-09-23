// Typed prefix GFlowNet. All sampling, learning and CEM numerics are C++.
// The C ABI exchanges metadata and scalar rewards; market arrays stay on CUDA.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
namespace {
constexpr int F=80, H=32, C=16;
struct Action { std::string name; int out,cost; std::vector<int> children; bool ts=false; };
// nest[i] = time-series operators already stacked above open slot i (windows on windows).
struct State { std::vector<int> seq,slots{0},nest{0}; int cost=0; };
struct Forward { std::array<double,F> x{}; std::array<double,H> h{}; std::vector<double> p; };
struct Probe { State state; std::array<int,3> actions; std::array<int,12> ids; };
thread_local std::string error,result;
void require(bool yes,const char* why) { if(!yes) throw std::runtime_error(why); }
struct Model {
    std::vector<Action> actions;
    std::vector<double> w1,w2,z,grad1,grad2,gradz;
    std::array<double,C> context{};
    std::mt19937_64 rng;
    std::vector<State> samples;
    std::vector<Probe> probes;
    int maxlen,maxcost,maxts,ordinary=0,epoch=0,trained=-1;
    std::string grammar;
    Model(const std::string& spec,int len,int cost,int ts,uint64_t seed):rng(seed),maxlen(len),maxcost(cost),maxts(ts),grammar(spec) {
        require(len>=3 && len<=64 && cost>=3 && cost<=100 && ts>=0 && ts<=64,"invalid grammar budget");
        std::istringstream rows(spec); std::string line;
        while(std::getline(rows,line)) {
            std::istringstream r(line); Action a; int n,t;
            require(bool(r>>a.name>>a.out>>a.cost>>n),"invalid grammar row");
            require(a.out>=0&&a.out<=5&&a.cost>=0&&n>=0&&n<=4,"invalid production");
            require(a.name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")==std::string::npos,"invalid token name");
            for(int i=0;i<n;i++) { require(bool(r>>t)&&t>=0&&t<=5,"invalid slot"); a.children.push_back(t); }
            a.ts=a.name.rfind("ts_",0)==0;
            actions.push_back(a);
        }
        require(!actions.empty(),"empty grammar");
        for(int slot=0;slot<4;slot++) {
            bool leaf=false; for(auto&a:actions) if(a.children.empty() && (a.out==slot||(slot==3&&a.out==2&&std::stod(a.name)<=31))) leaf=true;
            require(leaf,"grammar lacks terminal for a slot");
        }
        w1.resize(H*F); w2.resize(actions.size()*(H+1)); z.assign(C+1,0);
        std::normal_distribution<double> normal(0,.04);
        for(auto&v:w1)v=normal(rng);
        for(auto&v:w2)v=normal(rng);
        // Initial mass is split across fields and operators, not vocabulary size.
        int leaves=0,ops=0;
        for(auto&a:actions) if(a.out==0) (a.children.empty()?leaves:ops)++;
        for(size_t i=0;i<actions.size();i++) if(actions[i].out==0)
            w2[i*(H+1)+H]=std::log(actions[i].children.empty()?.35/std::max(1,leaves):.65/std::max(1,ops));
        z[0]=std::log(double(std::max(1,leaves)))-7;
    }
    bool legal(const State&s,int id) const {
        if(s.slots.empty())return false;
        const auto&a=actions[id];int slot=s.slots.back();
        if(a.out!=slot && !(slot==3&&a.out==2&&std::stod(a.name)<=31)
                       && !(slot==4&&a.out==1&&std::stod(a.name)!=0))return false;
        if(a.ts&&s.nest.back()>=maxts)return false;
        int need=s.cost+a.cost;
        for(size_t i=0;i+1<s.slots.size();i++)need+=s.slots[i]<=1||s.slots[i]==4;
        for(int c:a.children)need+=c<=1||c==4;
        return need<=maxcost && s.seq.size()+s.slots.size()+a.children.size()<=size_t(maxlen);
    }
    void step(State&s,int a) const {
        require(legal(s,a),"illegal action");int nest=s.nest.back()+actions[a].ts;
        s.slots.pop_back();s.nest.pop_back();s.seq.push_back(a);s.cost+=actions[a].cost;
        for(auto it=actions[a].children.rbegin();it!=actions[a].children.rend();++it){s.slots.push_back(*it);s.nest.push_back(nest);}
    }
    Forward forward(const State&s) const {
        Forward f; f.x[0]=1;
        for(int i=0;i<C;i++)f.x[1+i]=context[i];
        f.x[17]=double(s.seq.size())/maxlen;f.x[18]=double(s.cost)/maxcost;f.x[19]=double(s.slots.size())/maxlen;
        if(!s.slots.empty()) f.x[20+s.slots.back()]=1;
        // Position-sensitive hashed prefix features retain ordering and context.
        for(size_t i=0;i<s.seq.size();i++) {
            uint64_t hash=(uint64_t(s.seq[i]+1)*0x9e3779b97f4a7c15ULL)^(uint64_t(i+1)*0xbf58476d1ce4e5b9ULL);
            f.x[26+hash%54]+=(hash>>63?1.:-1.)/std::sqrt(double(std::max(size_t(1),s.seq.size())));
        }
        for(int j=0;j<H;j++) { double v=0;for(int i=0;i<F;i++)v+=w1[j*F+i]*f.x[i];f.h[j]=std::tanh(v); }
        f.p.assign(actions.size(),0);double maximum=-1e100;
        for(size_t a=0;a<actions.size();a++) if(legal(s,int(a))) {
            double v=w2[a*(H+1)+H];for(int j=0;j<H;j++)v+=w2[a*(H+1)+j]*f.h[j];
            f.p[a]=v;maximum=std::max(maximum,v);
        }
        require(maximum>-1e99,"grammar dead end");double total=0;
        for(size_t a=0;a<actions.size();a++)if(legal(s,int(a)))total+=(f.p[a]=std::exp(f.p[a]-maximum));else f.p[a]=0;
        for(auto&v:f.p)v/=total;
        return f;
    }
    int draw(const std::vector<double>&p) { return int(std::discrete_distribution<int>(p.begin(),p.end())(rng)); }
    std::string render(const State&s,size_t&at) const {
        const auto&a=actions[s.seq.at(at++)];std::string out=a.name;
        if(!a.children.empty()) { out+='(';for(size_t j=0;j<a.children.size();j++){if(j)out+=',';out+=render(s,at);}out+=')'; }
        return out;
    }
    std::string expression(const State&s) const { size_t at=0;auto out=render(s,at);require(at==s.seq.size(),"incomplete expression");return out; }
    void sample(int count,int nprobes,const double*ctx,int mode) {
        require(count>0&&count<=8192&&nprobes>=0&&nprobes<=count&&mode>=0&&mode<=1,"invalid batch");
        for(int i=0;i<C;i++){require(std::isfinite(ctx[i]),"nonfinite context");context[i]=ctx[i];}
        samples.clear();probes.clear();ordinary=count;++epoch;
        for(int i=0;i<count;i++) {
            State s;
            while(!s.slots.empty()) {auto p=forward(s).p;if(mode)for(size_t a=0;a<p.size();a++)p[a]=legal(s,int(a))?1:0;step(s,draw(p));}
            samples.push_back(s);
        }
        if(mode)return;
        // Shared suffixes are drawn from the geometric mean on common support.
        for(int attempt=0;attempt<nprobes*24&&int(probes.size())<nprobes;attempt++) {
            const auto&src=samples[rng()%ordinary];
            int pos=int(rng()%src.seq.size());State base;
            for(int k=0;k<pos;k++)step(base,src.seq[k]);
            int a0=src.seq[pos];const auto&first=actions[a0];auto pf=forward(base).p;
            std::vector<int> sib;
            for(size_t a=0;a<actions.size();a++) if(int(a)!=a0&&pf[a]>1e-4&&actions[a].children==first.children&&actions[a].cost==first.cost&&actions[a].out==first.out)sib.push_back(int(a));
            if(sib.size()<2)continue;
            std::shuffle(sib.begin(),sib.end(),rng);
            Probe probe{base,{a0,sib[0],sib[1]}, {}};
            std::vector<std::array<State,3>> rows;std::vector<std::vector<int>> suffixes;
            for(int tries=0;tries<40&&rows.size()<4;tries++) {
                std::array<State,3> branches{base,base,base};std::vector<int> suffix;
                for(int j=0;j<3;j++)step(branches[j],probe.actions[j]);
                while(!branches[0].slots.empty()) {
                    auto p0=forward(branches[0]).p,p1=forward(branches[1]).p,p2=forward(branches[2]).p;
                    for(size_t a=0;a<p0.size();a++)p0[a]=std::cbrt(p0[a]*p1[a]*p2[a]);
                    require(std::accumulate(p0.begin(),p0.end(),0.)>0,"empty common support");
                    int a=draw(p0);suffix.push_back(a);for(auto&b:branches)step(b,a);
                }
                if(std::find(suffixes.begin(),suffixes.end(),suffix)!=suffixes.end())continue;
                suffixes.push_back(suffix);rows.push_back(branches);
            }
            if(rows.size()!=4)continue;
            for(int k=0;k<4;k++)for(int j=0;j<3;j++){probe.ids[k*3+j]=int(samples.size());samples.push_back(rows[k][j]);}
            probes.push_back(probe);
        }
    }
    double logz() const { double v=z[0];for(int i=0;i<C;i++)v+=z[i+1]*context[i];return v; }
    void cleargrad() {grad1.assign(w1.size(),0);grad2.assign(w2.size(),0);gradz.assign(z.size(),0);}
    void backward(const Forward&f,const std::vector<double>&dlogit) {
        std::array<double,H> dh{};
        for(size_t a=0;a<actions.size();a++)if(dlogit[a]) {
            for(int j=0;j<H;j++){grad2[a*(H+1)+j]+=dlogit[a]*f.h[j];dh[j]+=dlogit[a]*w2[a*(H+1)+j];}
            grad2[a*(H+1)+H]+=dlogit[a];
        }
        for(int j=0;j<H;j++)for(int i=0;i<F;i++)grad1[j*F+i]+=dh[j]*(1-f.h[j]*f.h[j])*f.x[i];
    }
    double tb(const double*r,bool gradients) {
        double loss=0;
        for(int n=0;n<ordinary;n++) {
            State s;double sum=logz();std::vector<Forward> fs;
            for(int a:samples[n].seq){fs.push_back(forward(s));sum+=std::log(std::max(1e-300,fs.back().p[a]));step(s,a);}
            double d=sum-r[n];loss+=d*d/ordinary;
            if(!gradients)continue;
            double g=2*d/ordinary;gradz[0]+=g;for(int i=0;i<C;i++)gradz[i+1]+=g*context[i];
            for(size_t t=0;t<fs.size();t++) {
                auto dl=fs[t].p;for(auto&v:dl)v*=-g;dl[samples[n].seq[t]]+=g;backward(fs[t],dl);
            }
        }
        return loss;
    }
    int teacher(const double*r,double strength) {
        int used=0;
        for(const auto&probe:probes) {
            auto f=forward(probe.state);std::array<double,3> p{},credit{},target{};double total=0;int votes[3]={0,0,0};
            for(int j=0;j<3;j++)total+=(p[j]=f.p[probe.actions[j]]);
            for(auto&v:p)v/=total;
            for(int k=0;k<4;k++) {
                int best=0;for(int j=0;j<3;j++){credit[j]+=r[probe.ids[k*3+j]]/4;if(r[probe.ids[k*3+j]]>r[probe.ids[k*3+best]])best=j;}
                // Ties provide no reliable preference.
                bool unique=true;for(int j=0;j<3;j++)if(j!=best&&std::abs(r[probe.ids[k*3+j]]-r[probe.ids[k*3+best]])<1e-10)unique=false;
                if(unique)votes[best]++;
            }
            if(*std::max_element(votes,votes+3)<3)continue;
            auto make=[&](double alpha){double zt=0,m=*std::max_element(credit.begin(),credit.end());for(int j=0;j<3;j++)zt+=(target[j]=p[j]*std::exp(alpha*(credit[j]-m)));double kl=0;for(int j=0;j<3;j++){target[j]/=zt;kl+=target[j]*std::log(target[j]/p[j]);}return kl;};
            double lo=0,hi=1;for(int iter=0;iter<30;iter++){double mid=(lo+hi)/2;if(make(mid)>.03)hi=mid;else lo=mid;}make(lo);
            std::array<double,4> gain{};double mean=0,var=0;
            for(int k=0;k<4;k++){for(int j=0;j<3;j++)gain[k]+=(target[j]-p[j])*r[probe.ids[k*3+j]];mean+=gain[k]/4;}
            for(double v:gain)var+=(v-mean)*(v-mean)/3;
            double se=std::sqrt(var/4),lcb=mean-se;if(lcb<=0)continue;
            make(lo*std::clamp(lcb/(std::abs(mean)+1e-12),0.,1.));
            // The shared proposal term cancels in every within-row comparison.
            mean=0;var=0;gain.fill(0);
            for(int k=0;k<4;k++){for(int j=0;j<3;j++)gain[k]+=(target[j]-p[j])*r[probe.ids[k*3+j]];mean+=gain[k]/4;}
            for(double v:gain)var+=(v-mean)*(v-mean)/3;
            if(mean-std::sqrt(var/4)<=0)continue;
            std::vector<double> dl(actions.size(),0);
            for(int j=0;j<3;j++)dl[probe.actions[j]]=strength*(p[j]-target[j])/std::max(size_t(1),probes.size());
            backward(f,dl);used++;
        }
        return used;
    }
    void apply(double lr) {
        double norm=0;for(double v:grad1)norm+=v*v;for(double v:grad2)norm+=v*v;
        double scale=lr*std::min(1.,10/(std::sqrt(norm)+1e-12));
        for(size_t i=0;i<w1.size();i++)w1[i]-=scale*grad1[i];
        for(size_t i=0;i<w2.size();i++)w2[i]-=scale*grad2[i];
        for(size_t i=0;i<z.size();i++)z[i]-=lr*std::clamp(gradz[i],-20.,20.);
    }
};
}
API const char* q25_error(){return error.c_str();}
API void* q25_create(const char*grammar,int len,int cost,int ts,uint64_t seed){try{return new Model(grammar,len,cost,ts,seed);}catch(const std::exception&e){error=e.what();return nullptr;}}
API void q25_destroy(void*p){delete static_cast<Model*>(p);}
API const char* q25_sample(void*p,int count,int probes,const double*ctx,int mode){try{
    auto&m=*static_cast<Model*>(p);m.sample(count,probes,ctx,mode);std::ostringstream o;
    o<<"{\"epoch\":"<<m.epoch<<",\"ordinary\":"<<m.ordinary<<",\"probe_count\":"<<m.probes.size()<<",\"expressions\":[";
    for(size_t i=0;i<m.samples.size();i++){if(i)o<<',';o<<'"'<<m.expression(m.samples[i])<<'"';}o<<"]}";result=o.str();return result.c_str();
}catch(const std::exception&e){error=e.what();return nullptr;}}
API const char* q25_train(void*p,const double*r,int n,int epoch,int steps,double lr,double opd){try{
    auto&m=*static_cast<Model*>(p);require(epoch==m.epoch&&m.trained!=epoch,"stale or already trained batch");
    require(n==int(m.samples.size())&&steps>0&&steps<=100&&lr>0&&lr<=.1&&opd>=0&&opd<=10,"invalid training input");
    for(int i=0;i<n;i++)require(std::isfinite(r[i]),"nonfinite reward");
    double before=m.tb(r,false);int teachers=0;
    for(int k=0;k<steps;k++){
        m.cleargrad();m.tb(r,true);
        auto tb1=m.grad1,tb2=m.grad2;double tb_norm=0,aux_norm=0;
        for(double v:tb1)tb_norm+=v*v;
        for(double v:tb2)tb_norm+=v*v;
        teachers=m.teacher(r,1);
        for(size_t i=0;i<tb1.size();i++){double v=m.grad1[i]-tb1[i];aux_norm+=v*v;}
        for(size_t i=0;i<tb2.size();i++){double v=m.grad2[i]-tb2[i];aux_norm+=v*v;}
        double scale=std::min(1e4,opd*std::sqrt(tb_norm/(aux_norm+1e-30)));
        for(size_t i=0;i<tb1.size();i++)m.grad1[i]=tb1[i]+scale*(m.grad1[i]-tb1[i]);
        for(size_t i=0;i<tb2.size();i++)m.grad2[i]=tb2[i]+scale*(m.grad2[i]-tb2[i]);
        m.apply(lr);
    }
    m.trained=epoch;std::ostringstream o;o<<"{\"loss_before\":"<<before<<",\"loss_after\":"<<m.tb(r,false)<<",\"teachers\":"<<teachers<<",\"logz\":"<<m.logz()<<"}";result=o.str();return result.c_str();
}catch(const std::exception&e){error=e.what();return nullptr;}}
API const char* q25_checkpoint(void*p,const char*path,int load){try{
    auto&m=*static_cast<Model*>(p);
    if(load){
        std::ifstream f(path);require(bool(f),"cannot read checkpoint");std::string magic,spec;int len=0,cost=0,ts=0;
        std::getline(f,magic);require(magic=="q25-native-gfn-v2","checkpoint protocol mismatch");f>>len>>cost>>ts>>std::quoted(spec);
        require(len==m.maxlen&&cost==m.maxcost&&ts==m.maxts&&spec==m.grammar,"checkpoint grammar mismatch");
        auto a=m.w1,b=m.w2,c=m.z;for(auto*v:{&a,&b,&c})for(auto&x:*v){f>>x;require(bool(f)&&std::isfinite(x),"corrupt checkpoint");}
        auto rng=m.rng;f>>rng;require(bool(f),"corrupt RNG checkpoint");m.w1=a;m.w2=b;m.z=c;m.rng=rng;
        m.samples.clear();m.probes.clear();m.ordinary=0;m.trained=m.epoch;
    }else{
        std::ofstream f(path);require(bool(f),"cannot write checkpoint");f<<"q25-native-gfn-v2\n"<<m.maxlen<<' '<<m.maxcost<<' '<<m.maxts<<' '<<std::quoted(m.grammar)<<'\n'<<std::setprecision(17);
        for(auto*v:{&m.w1,&m.w2,&m.z}){for(double x:*v)f<<x<<' ';f<<'\n';}f<<m.rng;require(bool(f),"checkpoint write failed");
    }
    return "{}";
}catch(const std::exception&e){error=e.what();return nullptr;}}
API const char* q25_gradient_check(void*p,const double*r,int n){try{
    auto&m=*static_cast<Model*>(p);require(n==int(m.samples.size()),"reward shape");m.cleargrad();m.tb(r,true);double err=0;
    for(auto pair:{std::make_pair(&m.w1,&m.grad1),std::make_pair(&m.w2,&m.grad2),std::make_pair(&m.z,&m.gradz)}) {
        auto&w=*pair.first;auto&g=*pair.second;
        for(size_t i=0;i<w.size();i+=std::max(size_t(1),w.size()/19)) {double old=w[i],eps=1e-5;w[i]=old+eps;double a=m.tb(r,false);w[i]=old-eps;double b=m.tb(r,false);w[i]=old;err=std::max(err,std::abs((a-b)/(2*eps)-g[i]));}
    }
    std::ostringstream o;o<<"{\"max_gradient_error\":"<<err<<"}";result=o.str();return result.c_str();
}catch(const std::exception&e){error=e.what();return nullptr;}}
API int q25_cem_sample(const double*mean,const double*sd,int n,int count,uint64_t seed,double*out){try{
    require(n>0&&n<=64&&count>0,"CEM shape");std::mt19937_64 rng(seed);std::normal_distribution<double> normal;
    // Member weights are additive scales in (0,1), independent of each other: no simplex.
    for(int b=0;b<count;b++)for(int j=0;j<n;j++){require(std::isfinite(mean[j])&&std::isfinite(sd[j])&&sd[j]>=0,"CEM parameters");out[b*n+j]=1/(1+std::exp(-(mean[j]+sd[j]*normal(rng))));}
    return 1;
}catch(const std::exception&e){error=e.what();return 0;}}
API int q25_cem_update(const double*weights,const double*scores,int n,int count,int elite,double*mean,double*sd){try{
    require(n>0&&count>=elite&&elite>0,"CEM update shape");std::vector<int> ids(count);std::iota(ids.begin(),ids.end(),0);
    std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return scores[a]>scores[b];});
    std::vector<double> x(elite*n);for(int k=0;k<elite;k++)for(int j=0;j<n;j++){double w=std::clamp(weights[ids[k]*n+j],1e-12,1-1e-12);x[k*n+j]=std::log(w/(1-w));}
    for(int j=0;j<n;j++){double mu=0,var=0;for(int k=0;k<elite;k++)mu+=x[k*n+j]/elite;for(int k=0;k<elite;k++)var+=(x[k*n+j]-mu)*(x[k*n+j]-mu)/elite;mean[j]=.3*mean[j]+.7*mu;sd[j]=std::max(.1,.3*sd[j]+.7*std::sqrt(var));}return 1;
}catch(const std::exception&e){error=e.what();return 0;}}
