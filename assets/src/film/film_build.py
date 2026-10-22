"""Write film.html (the MK-SH-V launch film) and cues.json.

    python assets/src/film/film_build.py
    python assets/src/film/film_render.py      # frames -> mp4 (see that file)

One pure render(frame) function drives every pixel (cinetic: everything is a
function of frame; no CSS animation, no Math.random). Timing lives in CUE.
Numbers on screen: real_cases.json (release weights), runs/e41, runs/e42.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import base64  # noqa: E402

E42 = json.loads((ROOT / "runs/e42-calib/e42_summary.json").read_text(encoding="utf-8"))["seeds"]["42"]
T = E42["tables"]["test_nonfr"]["overall"]
E41 = json.loads((ROOT / "runs/e41/e41_report.json").read_text(encoding="utf-8"))["seeds"]["42"]
GUI = E41["real_by_slice"]["gui"]

DATA = {
    "gui_true": 0.999, "gui_false": 0.001,              # real_cases.json, gui-notes-fin.png
    "rc_false": 0.507, "rc_true": 0.493,                # real_cases.json, cord-receipt-1.png @ 45.662
    "tau": E42["tau"]["threshold"],
    "gui_before": round(GUI["e39"] * GUI["n"]) / GUI["n"], "gui_after": round(GUI["e41"] * GUI["n"]) / GUI["n"],  # same model: synthetic-only -> + real data
    "gui_delta_pp": round(GUI["delta_vs_e39"]["delta"] * 100, 1),
    "jev_gui": 0.731,                                   # E39-r2, zero-shot, same 2,010 pairs
    "n": T["n"], "n_acc": T["n_acc"], "errors": T["errors"],
    "cov": T["coverage"], "risk": T["risk"], "lo": T["cp95"][0], "hi": T["cp95"][1],
}
assert abs(DATA["n_acc"] / DATA["n"] - DATA["cov"]) < 1e-3
assert abs(DATA["errors"] / DATA["n_acc"] - DATA["risk"]) < 1e-3
assert DATA["hi"] < 0.05 and DATA["rc_false"] < DATA["tau"] and DATA["gui_true"] >= DATA["tau"]

# cues on the 120 BPM / 60 fps grid (beat 30 f, 8th 15 f)
CUE = {
    "type1": 36, "pullback": 186, "scan": 210, "tauMorph": 246, "rows": 252, "fill1": 276,
    "f2": 315, "choice": 360, "f3": 390, "copyOut1": 465,
    "drain": 480, "reel": 480, "del": 495, "type2": 525, "fill2": 600, "stall": 636,
    "status": 645, "freeze": 660, "f4": 690, "truth": 735, "f5": 765, "copyOut2": 825,
    "cols": 840, "plate": 900, "before": 900, "lift": 945, "count": 951, "chip": 987,
    "compare": 1050, "f6": 1095, "f7": 1140, "copyOut3": 1185,
    "dots": 1200, "dim": 1260, "rings": 1305, "meter": 1320, "bound": 1350, "f8": 1365,
    "copyOut4": 1425, "iris": 1440, "turn": 1452, "glint": 1472, "word": 1485,
    "line": 1515, "url": 1540, "end": 1650,
}

HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>MK-SH-V film</title>
<style>
@font-face{font-family:Geist;src:url(../fonts/geist-500.woff2) format("woff2");font-weight:500}
@font-face{font-family:Geist;src:url(../fonts/geist-700.woff2) format("woff2");font-weight:700}
@font-face{font-family:"Geist Mono";src:url(../fonts/geistmono-400.woff2) format("woff2");font-weight:400}
@font-face{font-family:"Geist Mono";src:url(../fonts/geistmono-500.woff2) format("woff2");font-weight:500}
:root{--stage:#0E1724;--surf:#16233A;--surf2:#1F2D44;--line:#2A3A50;--ink:#EEF1F5;--ink2:#B7C2D0;
      --muted:#8E9BAD;--acc:#E8A13C;--ground:#1B2A41;--paper:#F7F6F3}
*{margin:0;padding:0;box-sizing:border-box}
html,body{width:1920px;height:1080px;overflow:hidden;background:var(--stage)}
body{font-family:Geist,sans-serif;color:var(--ink);font-variant-numeric:tabular-nums}
.abs{position:absolute;left:0;top:0;transform-origin:0 0}
.mono{font-family:"Geist Mono",monospace}
#film{position:relative;width:1920px;height:1080px;overflow:hidden;background:var(--stage)}
#world{width:1920px;height:1080px}
.card{border-radius:28px;overflow:hidden;background:#111A28;box-shadow:0 30px 90px rgba(0,0,0,.35)}
#reel .cell{position:absolute;left:0;width:397px;height:882px;display:flex;align-items:center;justify-content:center;background:#111A28}
#reel img{max-width:100%;max-height:100%;display:block}
#qlabel{font-size:18px;color:var(--muted)}
#pill{width:820px;height:84px;border-radius:42px;background:var(--surf2);box-shadow:inset 0 1px 0 rgba(255,255,255,.10),0 30px 90px rgba(0,0,0,.35);
      display:flex;align-items:center;padding:0 36px;font-size:34px;font-weight:500;letter-spacing:-.01em;white-space:pre}
#caret{display:inline-block;width:3px;height:40px;background:var(--ink);margin-left:3px;vertical-align:-6px}
.row .id{font-size:32px;color:var(--ink)}.row .desc{font-size:22px;color:var(--muted);margin-top:4px}
.track{width:600px;height:22px;border-radius:11px;background:var(--surf2);overflow:hidden}
.fill{height:22px;border-radius:11px;background:var(--ink);width:0}
.val{font-size:32px;color:var(--ink);width:170px;text-align:right}
#taulab{font-size:24px;color:var(--acc)}
#chip{padding:16px 30px;border-radius:40px;background:var(--ink);color:var(--stage);font-size:40px;font-weight:700;letter-spacing:-.02em;white-space:nowrap}
.st{font-size:30px;color:var(--ink2);white-space:nowrap}
#absword{font-size:76px;font-weight:700;letter-spacing:-.03em;color:var(--acc)}
#reason{font-size:26px;padding:8px 16px;border-radius:8px;background:#2A2113;color:var(--acc)}
#truth{font-size:27px;color:var(--ink2);white-space:nowrap}
.copy{font-size:64px;font-weight:700;letter-spacing:-.035em;white-space:nowrap;line-height:1}
.copy span{display:inline-block;margin-right:.24em}
.col{position:absolute;top:0;width:128px;height:1080px;background:var(--surf)}
#cap{font-size:22px;color:var(--muted)}
#before{font-size:64px;font-weight:700;letter-spacing:-.03em;color:var(--muted)}
#beforelab{font-size:22px;color:var(--muted)}
#strike{height:3px;background:var(--muted);width:0}
#hero{font-size:230px;font-weight:700;letter-spacing:-.05em;line-height:.9}
#herolab{font-size:24px;color:var(--ink2)}
#dchip{padding:12px 22px;border-radius:30px;background:var(--ink);color:var(--stage);font-size:40px;font-weight:700;letter-spacing:-.02em}
.cmp .lab{font-size:44px;font-weight:700;letter-spacing:-.03em;color:var(--ink)}.cmp .sub{font-size:24px;color:var(--muted);margin-top:6px}
.cmp.bar{height:40px;border-radius:20px}
.cmp.v{font-size:44px}
#big{font-size:120px;font-weight:700;letter-spacing:-.05em;line-height:.9}
.k{font-size:24px;color:var(--ink2)}
#risk{font-size:72px;font-weight:700;letter-spacing:-.04em;line-height:.9}
#meter{width:500px;height:6px;background:var(--surf2);border-radius:3px}
#lock{background:var(--ground)}
#wm{font-size:150px;font-weight:700;letter-spacing:-.045em;color:var(--paper);line-height:.9}
#tag{font-size:44px;font-weight:500;letter-spacing:-.02em;color:var(--ink2)}
#url{font-size:30px;color:var(--ink2)}
</style></head><body><div id="film">
<div id="world" class="abs">
  <div id="phone" class="abs card" style="width:397px;height:882px"><div id="reel" class="abs"></div></div>
  <div id="qlabel" class="abs mono">question</div>
  <div id="pill" class="abs"><span id="qtext"></span><span id="caret"></span></div>
  <div id="taulab" class="abs mono">&tau; 0.55</div>
  <div id="r0" class="abs row"><div class="id mono">false</div><div class="desc">No</div></div>
  <div id="r1" class="abs row"><div class="id mono">true</div><div class="desc">Yes</div></div>
  <div id="t0" class="abs track"><div class="fill"></div></div><div id="t1" class="abs track"><div class="fill"></div></div>
  <div id="v0" class="abs val mono"></div><div id="v1" class="abs val mono"></div>
  <div id="chip" class="abs">choice: true</div>
  <div id="s0" class="abs st mono"></div>
  <div id="absword" class="abs">abstain</div>
  <div id="reason" class="abs mono">below_validated_threshold</div>
  <div id="truth" class="abs mono">its top pick, false, was wrong: the total is 302.016 IDR</div>
  <svg id="dev" class="abs" width="1920" height="1080" style="overflow:visible"><line id="devl" stroke="#E8A13C" stroke-width="4" stroke-linecap="round"/></svg>
</div>
<div id="cols" class="abs" style="width:1920px;height:1080px"></div>
<div id="actD" class="abs" style="width:1920px;height:1080px">
  <div id="cap" class="abs mono">real screenshots, GUI-Odyssey, 2,010 held-out pairs</div>
  <div id="before" class="abs">0.493</div><div id="beforelab" class="abs mono">same model, synthetic data only</div>
  <div id="strike" class="abs"></div>
  <div id="hero" class="abs">0.493</div><div id="herolab" class="abs mono">same 260M model, plus real CC-BY training data</div>
  <div id="dchip" class="abs"></div>
  <div id="c0" class="abs cmp"><div class="lab">MK-SH-V</div><div class="sub mono">260M, trained</div></div>
  <div id="c1" class="abs cmp"><div class="lab">jev-vlm-4b</div><div class="sub mono">~4B, zero-shot</div></div>
  <div id="q0" class="abs" style="background:#EEF1F5"></div><div id="q1" class="abs" style="background:#66778E"></div>
  <div id="b0" class="abs cmp bar" style="background:#EEF1F5"></div><div id="b1" class="abs cmp bar" style="background:#66778E"></div>
  <div id="bv0" class="abs cmp v mono"></div><div id="bv1" class="abs cmp v mono" style="color:#B7C2D0"></div>
</div>
<div id="actE" class="abs" style="width:1920px;height:1080px">
  <canvas id="dots" class="abs" width="1100" height="660"></canvas>
  <div id="big" class="abs"></div><div id="bigk" class="abs k mono">answered, the rest refused</div>
  <div id="risk" class="abs"></div><div id="riskk" class="abs k mono">wrong among the answered</div>
  <div id="meter" class="abs"></div>
  <svg id="msvg" class="abs" width="520" height="140" style="overflow:visible"></svg>
  <div id="ecap" class="abs mono" style="font-size:20px;color:#8E9BAD">held-out synthetic test, 4,200 cases, seed 42, exact Clopper-Pearson 95 %</div>
</div>
<div id="lock" class="abs" style="width:1920px;height:1080px">
  <div id="markwrap" class="abs" style="width:620px;height:620px;perspective:1200px">
    <img id="mark" src="../../logo-mark.png" width="620" height="620" alt="" style="display:block">
    <div id="glintmask" class="abs" style="width:620px;height:620px;-webkit-mask:url(__MASK__) 0 0/620px 620px no-repeat;mask:url(__MASK__) 0 0/620px 620px no-repeat">
      <div id="glint" class="abs" style="width:150px;height:620px;background:#fff;opacity:0"></div></div>
  </div>
  <div id="wm" class="abs">MK-SH-V</div>
  <div id="tag" class="abs">A 260M visual decision model.</div>
  <div id="url" class="abs mono">github.com/Gjusev/mk-sh-v</div>
</div>
<div id="copies" class="abs"></div>
</div>
<script>
const D = __DATA__, C = __CUE__;
// ---- motion tokens (cinetic) ----
function bez(x1,y1,x2,y2){const cx=3*x1,bx=3*(x2-x1)-cx,ax=1-cx-bx,cy=3*y1,by=3*(y2-y1)-cy,ay=1-cy-by;
  const sx=t=>((ax*t+bx)*t+cx)*t, sy=t=>((ay*t+by)*t+cy)*t, dx=t=>(3*ax*t+2*bx)*t+cx;
  return x=>{if(x<=0)return 0;if(x>=1)return 1;let t=x;for(let i=0;i<8;i++){const e=sx(t)-x,d=dx(t);if(Math.abs(e)<1e-6)break;if(Math.abs(d)<1e-6)break;t-=e/d}
    let lo=0,hi=1;if(Math.abs(sx(t)-x)>1e-5){t=x;for(let i=0;i<40;i++){const v=sx(t);if(v<x)lo=t;else hi=t;t=(lo+hi)/2}}return sy(t)}}
const E={out:bez(.16,1,.3,1),outSoft:bez(.22,1,.36,1),in:bez(.7,0,.84,0),inOut:bez(.87,0,.13,1),smooth:bez(.65,0,.35,1),
  ui:bez(.4,0,.2,1),cam:bez(.48,.1,0,.9),glide:bez(.47,.2,.15,1),rest:bez(.45,0,.1,1),contact:bez(.55,0,.9,.55),lin:t=>t};
const cl=(v,a,b)=>Math.max(a,Math.min(b,v));
const prog=(f,a,b,e=E.lin)=>e(cl((f-a)/(b-a),0,1));
const tw=(f,a,b,v0,v1,e=E.lin)=>v0+(v1-v0)*prog(f,a,b,e);
const lmix=(a,b,t)=>Math.exp(Math.log(a)+(Math.log(b)-Math.log(a))*t);
const pulse=(t,att=2,tau=5)=>t<0?0:t<att?t/att:Math.exp(-(t-att)/tau);
function rnd(seed){let s=seed>>>0;return()=>{s=(s+0x6D2B79F5)>>>0;let t=s;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return((t^t>>>14)>>>0)/4294967296}}
const $=id=>document.getElementById(id);
function put(el,x,y,s=1,o=1,extra=''){el.style.transform=`translate(${x}px,${y}px) scale(${s}) ${extra}`;el.style.opacity=o;}
function vis(el,on){el.style.visibility=on?'visible':'hidden'}

// ---- layout ----
const PH={x:300,y:110}, PILL={x:860,y:150}, TX=1040, TW=600, ROWY=[340,460], TAUX=TX+TW*D.tau;
const reel=$('reel'); const IMGS=['gui-notes-fin.png','cord-receipt-2.png','cord-receipt-1.png'];
IMGS.forEach((n,i)=>{const c=document.createElement('div');c.className='cell';c.style.top=(i*906)+'px';
  const im=document.createElement('img');im.src='../../../demo/examples/'+n;c.appendChild(im);reel.appendChild(c)});
const Q1='Is the goal done?', Q2A='Is the ', Q2B='total at least 45.662 IDR?';

// ---- copy (crisp pace: entry 26 f on out, stagger 6 f, rise .32em, blur 8 px; exit per word 16 f) ----
const CHAINS=[
  {x:748,y:930,parts:[['When it’s sure,',C.f2],['it answers.',C.f3]],out:C.copyOut1},
  {x:748,y:960,parts:[['When it can’t tell,',C.f4],['it refuses.',C.f5,'acc']],out:C.copyOut2},
  {x:300,y:930,parts:[['15× smaller,',C.f6],['still ahead.',C.f7]],out:C.copyOut3},
  {x:160,y:960,parts:[['The risk, bounded.',C.f8]],out:C.copyOut4},
];
const copies=$('copies'); const WORDS=[];
CHAINS.forEach(ch=>{const line=document.createElement('div');line.className='abs copy';line.style.transform=`translate(${ch.x}px,${ch.y-64}px)`;
  let k=0; ch.parts.forEach(([txt,inF,acc])=>txt.split(' ').forEach((w,i)=>{const s=document.createElement('span');s.textContent=w;line.appendChild(s);
    WORDS.push({el:s,in:inF+i*6,out:ch.out+(k++)*3,acc:!!acc})}));copies.appendChild(line)});
function renderCopy(f){for(const w of WORDS){const p=prog(f,w.in,w.in+26,E.out), q=prog(f,w.out,w.out+16,E.in);
  const blur=8*(1-prog(f,w.in,w.in+15.6,E.out)); w.el.style.opacity=(f<w.in?0:p)*(1-q);
  w.el.style.transform=`translateY(${((1-p)*0.32-q*0.2)*64}px)`; w.el.style.filter=blur>0.75?`blur(${blur.toFixed(2)}px)`:'none';
  w.el.style.color=w.acc?'#E8A13C':'#EEF1F5'}}

// ---- dots (act E): seeded statuses with the measured proportions ----
const NC=84,NR=50,PITCH=13; const order=[...Array(NC*NR).keys()]; {const r=rnd(42);for(let i=order.length-1;i>0;i--){const j=Math.floor(r()*(i+1));[order[i],order[j]]=[order[j],order[i]]}}
const STATUS=new Array(NC*NR); order.forEach((idx,rank)=>{STATUS[idx]=rank<D.n_acc-D.errors?'ok':rank<D.n_acc?'err':'ref'});

function render(f){
  // ===== acts A-C: the world under one camera =====
  const inWorld=f<C.cols+60; vis($('world'),inWorld);
  if(inWorld){
    // camera: two pushes onto the caret, then the pull back (log-space zoom)
    const k1=lmix(1.3,1.6,prog(f,24,66,E.cam)), k2=lmix(1,1.5,prog(f,66,110,E.glide)), kp=lmix(1,1.06/2.4,prog(f,C.pullback,C.pullback+48,E.cam));
    let k=k1*k2*kp*lmix(1,1.025,prog(f,C.pullback+48,C.cols,E.lin)); // alive holds: a slow creep
    const qw=$('qtext').offsetWidth;
    const ax0=PILL.x+410, axT=PILL.x+36+Math.min(qw,560)*prog(f,C.type1,C.type1+40,E.smooth);
    let ax=tw(f,0,90,ax0,axT,E.smooth), ay=PILL.y+42;
    const back=prog(f,C.pullback,C.pullback+48,E.cam); ax=ax+(1060-ax)*back; ay=ay+(560-ay)*back;
    const kick=pulse(f-C.choice,1,4)*8+pulse(f-C.stall,1,4)*-6;
    $('world').style.transform=`translate(${960-ax*k+kick}px,${540-ay*k}px) scale(${k})`;
    // phone card with the reel (flick scroll lands the receipt)
    const ph=prog(f,C.pullback+4,C.pullback+44,E.out); put($('phone'),PH.x-60*(1-ph),PH.y,1,ph);
    const tp=30,d=0.95,tau=-1/Math.log(d),Dst=2*906,vmax=Dst/(tp/3+tau); let t=f-C.reel,pos=0;
    if(t>0)pos=t<tp?vmax*t*t*t/(3*tp*tp):vmax*tp/3+vmax*tau*(1-Math.pow(d,t-tp)); pos=Math.abs(Dst-pos)<0.5?Dst:Math.min(pos,Dst);
    reel.style.transform=`translateY(${-pos}px)`;
    // question field: type, then edit the tail (accelerating deletes), keep the anchor
    let txt='';
    if(f<C.del){txt=Q1.slice(0,cl(Math.floor((f-C.type1)*30/60)+1,0,Q1.length))}
    else if(f<C.type2){const tail=Q1.slice(Q2A.length), gaps=[10,5,5,2,2,2,2,2,2,2]; let n=tail.length,acc=C.del; for(const g of gaps){acc+=g;if(f>=acc)n--} txt=Q2A+tail.slice(0,Math.max(0,n))}
    else txt=Q2A+Q2B.slice(0,cl(Math.floor((f-C.type2)*30/60)+1,0,Q2B.length));
    if(f<C.type1)txt='';
    $('qtext').textContent=txt;
    const typing=(f>=C.type1&&f<C.type1+36)||(f>=C.del&&f<C.type2+54); const idle=f-(f<C.del?C.type1+36:C.type2+54);
    $('caret').style.opacity=typing||idle<0?1:(Math.floor(idle/30)%2?0:1);
    put($('pill'),PILL.x,PILL.y); put($('qlabel'),PILL.x+36,PILL.y-34,1,prog(f,C.pullback,C.pullback+30,E.smooth));
    // options rows, bars, values
    const rowIn=i=>prog(f,C.rows+i*6,C.rows+i*6+20,E.out);
    const p1=prog(f,C.fill1,C.fill1+36,E.out), drain=prog(f,C.drain,C.drain+16,E.in), p2=prog(f,C.fill2,C.fill2+36,E.out);
    const vals=[D.gui_false*p1*(1-drain)+D.rc_false*p2, D.gui_true*p1*(1-drain)+D.rc_true*p2];
    for(let i=0;i<2;i++){const r=rowIn(i);put($('r'+i),PILL.x,ROWY[i]-14+12*(1-r),1,r);put($('t'+i),TX,ROWY[i]+8,1,r);
      $('t'+i).firstChild.style.width=(TW*vals[i]).toFixed(2)+'px'; $('v'+i).textContent=(vals[i]*100).toFixed(1)+' %';put($('v'+i),TX+TW+14,ROWY[i]-4,1,r)}
    const dimAns=tw(f,C.stall+2,C.stall+10,1,0.3,E.smooth); // stalled state: the answer slot dims
    $('t0').firstChild.style.opacity=f>C.stall?dimAns:1; $('t1').firstChild.style.opacity=f>C.stall?dimAns:1;
    // device: scan line sweeps the screenshot, then becomes the tau tick
    const sc=prog(f,C.scan,C.scan+36,E.inOut), mo=prog(f,C.tauMorph,C.tauMorph+30,E.inOut);
    const yS=PH.y+12+858*sc, P0=[PH.x+18,yS,PH.x+379,yS], P1=[TAUX,ROWY[0]-18,TAUX,ROWY[1]+40];
    const L=P0.map((v,i)=>v+(P1[i]-v)*mo); const dl=$('devl');
    ['x1','y1','x2','y2'].forEach((a,i)=>dl.setAttribute(a,L[i])); dl.setAttribute('stroke-width',4+2*pulse(f-C.stall,2,5));
    dl.style.opacity=(f<C.scan?0:1)*(1-prog(f,C.cols,C.cols+20,E.smooth));
    put($('taulab'),TAUX-34,ROWY[0]-56,1,prog(f,C.tauMorph+20,C.tauMorph+34,E.out));
    // choice chip (true landing: small overshoot allowed)
    const cp=prog(f,C.choice-8,C.choice+6,E.out), cq=prog(f,C.drain,C.drain+14,E.in);
    put($('chip'),PILL.x,580,0.9+0.1*cp+0.03*pulse(f-C.choice,2,5),cp*(1-cq));
    // stalled state words: +6, +28, +40; +51 first dims; then the truth note
    const s0=prog(f,C.status+6,C.status+14,E.out);
    $('s0').textContent=`max ${(D.rc_false*100).toFixed(1)} % < τ ${D.tau}`;
    put($('s0'),PILL.x,566+6*(1-s0),1,s0*tw(f,C.status+51,C.status+59,1,0.4,E.smooth)*(1-prog(f,C.cols,C.cols+20)));
    const s1=prog(f,C.status+28,C.status+36,E.out); put($('absword'),PILL.x,612+6*(1-s1),1,s1*(1-prog(f,C.cols,C.cols+20)));
    const s2=prog(f,C.status+40,C.status+48,E.out); put($('reason'),PILL.x+340,636+6*(1-s2),1,s2*(1-prog(f,C.cols,C.cols+20)));
    const tr=prog(f,C.truth,C.truth+12,E.out); put($('truth'),PILL.x,736+6*(1-tr),1,tr*(1-prog(f,C.cols,C.cols+20)));
    vis($('chip'),f<C.drain+20);
  }
  // ===== transition: stepped column fill =====
  const cols=$('cols'); vis(cols,f>=C.cols&&f<C.plate+90);
  if(f>=C.cols&&f<C.plate+90){for(let i=0;i<15;i++){const c=cols.children[i],st=Math.abs(i-7)*2,p=prog(f,C.cols+st,C.cols+st+40,E.smooth);
    c.style.transform=`translate(${i*128}px,${1080*(1-p)}px)`; c.style.background=i%2?'#16233A':'#14203A'}
    cols.style.opacity=1-prog(f,C.plate+30,C.plate+60,E.smooth)}
  // ===== act D: delta chip compare (same model), then 260M against ~4B =====
  const inD=f>=C.plate-2&&f<C.dots+30; vis($('actD'),inD);
  if(inD){$('actD').style.background=f<C.dots?'#16233A':'transparent'; $('actD').style.opacity=1-prog(f,C.dots-6,C.dots+12,E.smooth);
    put($('cap'),300,190,1,prog(f,C.before,C.before+16,E.out));
    const x1=prog(f,C.compare-18,C.compare-4,E.in);               // phase 1 leaves, lifting
    const h=prog(f,C.before,C.before+26,E.out), lift=prog(f,C.lift,C.lift+12,E.in), mr=prog(f,C.lift+4,C.lift+34,E.out);
    const cnt=prog(f,C.count,C.count+36,E.out), val=D.gui_before+(D.gui_after-D.gui_before)*cnt;
    $('hero').textContent=(f<C.count?D.gui_before:val).toFixed(3);
    const up=-40*x1;
    put($('hero'),288,380+30*(1-h)+up,1,h*(1-x1));
    put($('before'),300,262+20*(1-mr)+up,1,mr*(f>=C.lift?0.9:0)*(1-x1)); put($('beforelab'),500,286+20*(1-mr)+up,1,mr*(f>=C.lift?1:0)*(1-x1));
    $('strike').style.width=(170*prog(f,C.chip,C.chip+10,E.out))+'px'; put($('strike'),298,302+up,1,1-x1);
    put($('herolab'),300,620+up,1,prog(f,C.count+20,C.count+40,E.out)*(1-x1));
    $('dchip').textContent='+'+D.gui_delta_pp+' pp'; const ch=prog(f,C.chip,C.chip+10,E.out);
    put($('dchip'),960,470+up,0.6+0.4*ch,ch*(1-x1));
    // phase 2: squares by parameter count (area), bars by accuracy, full frame
    const sq=[44,Math.round(44*Math.sqrt(4000/260))], ys=[420,660], BX=880, BW=840;
    for(let i=0;i<2;i++){const p=prog(f,C.compare+i*10,C.compare+i*10+26,E.out),y=ys[i];
      put($('q'+i),390-sq[i]/2,y-sq[i]/2,1,p); $('q'+i).style.width=$('q'+i).style.height=sq[i]+'px';
      put($('c'+i),540,y-44+12*(1-p),1,p);
      const pb=prog(f,C.compare+18+i*10,C.compare+54+i*10,E.out),v=[D.gui_after,D.jev_gui][i];
      $('b'+i).style.width=(BW*v*pb)+'px'; put($('b'+i),BX,y-20,1,p); $('bv'+i).textContent=(v*pb).toFixed(3); put($('bv'+i),BX+BW*v*pb+20,y-24,1,p)}
  }
  // ===== act E: 4,200 held-out cases, the risk meter =====
  const inE=f>=C.dots-10&&f<C.iris+30; vis($('actE'),inE);
  if(inE){const cv=$('dots'),g=cv.getContext('2d'); g.clearRect(0,0,1100,660);
    const dim=prog(f,C.dim,C.dim+30,E.smooth), ring=prog(f,C.rings,C.rings+14,E.out);
    for(let c=0;c<NC;c++){const a=prog(f,C.dots+c*0.5,C.dots+c*0.5+18,E.out); if(a<=0)continue;
      for(let r=0;r<NR;r++){const s=STATUS[r*NC+c],x=8+c*PITCH,y=8+r*PITCH+(1-a)*10;
        g.globalAlpha=a*(s==='ref'?1-0.78*dim:1);
        if(s==='err'&&ring>0){g.fillStyle='#16233A';g.beginPath();g.arc(x,y,4,0,7);g.fill();g.globalAlpha=a*ring;g.strokeStyle='#EEF1F5';g.lineWidth=2;g.beginPath();g.arc(x,y,4.4,0,7);g.stroke()}
        else{g.fillStyle=s==='ref'?'#2A3A50':'#8E9BAD';g.beginPath();g.arc(x,y,4,0,7);g.fill()}}}
    g.globalAlpha=1; put(cv,150,170+tw(f,C.dots,C.iris,0,-12),1,1);
    const bc=prog(f,C.dim,C.dim+36,E.out); $('big').textContent=(D.cov*100*bc).toFixed(1)+' %'; put($('big'),1330,190,1,prog(f,C.dim,C.dim+12,E.out));
    put($('bigk'),1336,310,1,prog(f,C.dim+10,C.dim+24,E.out));
    const rc=prog(f,C.rings,C.rings+30,E.out); $('risk').textContent=(D.risk*100*rc).toFixed(2)+' %'; put($('risk'),1330,400,1,prog(f,C.rings,C.rings+12,E.out));
    put($('riskk'),1336,476,1,prog(f,C.rings+10,C.rings+24,E.out));
    // meter: 0..8 %, budget line (the device) at 5 %, interval lands left of it
    const MX=1300,MY=730,MW=500,X=r=>MX+MW*r/0.08, mt=prog(f,C.meter,C.meter+20,E.out);
    put($('meter'),MX,MY,1,mt); const sv=$('msvg'); put(sv,0,0,1,1); sv.style.transform='none'; sv.setAttribute('width',1920); sv.setAttribute('height',1080); sv.style.left='0px';
    const lb=prog(f,C.bound-10,C.bound,E.contact), kick=6*pulse(f-C.bound,1,4);
    sv.innerHTML=`<line x1="${X(0.05)}" y1="${MY-70}" x2="${X(0.05)}" y2="${MY+30}" stroke="#E8A13C" stroke-width="${4+2*pulse(f-C.bound,2,5)}" stroke-linecap="round" opacity="${mt}"/>
      <text x="${X(0.05)+12}" y="${MY-80}" font-family="Geist Mono" font-size="24" fill="#E8A13C" opacity="${mt}">5 % budget</text>
      ${[0,0.02,0.04,0.06,0.08].map(v=>`<text x="${X(v)}" y="${MY+60}" text-anchor="middle" font-family="Geist Mono" font-size="22" fill="#8E9BAD" opacity="${mt}">${v*100} %</text>`).join('')}
      <line x1="${X(D.lo)}" y1="${MY+3-60*(1-lb)}" x2="${X(D.hi)}" y2="${MY+3-60*(1-lb)}" stroke="#EEF1F5" stroke-width="10" stroke-linecap="round" opacity="${lb}"/>
      <circle cx="${X(D.risk)}" cy="${MY+3-60*(1-lb)}" r="${11+kick*0.3}" fill="#EEF1F5" stroke="#0E1724" stroke-width="3" opacity="${lb}"/>
      <text x="${X(D.lo)}" y="${MY+110}" font-family="Geist Mono" font-size="26" fill="#EEF1F5" opacity="${prog(f,C.bound+4,C.bound+16,E.out)}">CP95 upper ${(D.hi*100).toFixed(1)} %</text>`;
    put($('ecap'),150,846,1,prog(f,C.dots+30,C.dots+50,E.out));
    $('actE').style.transform=`translate(${-kick*0.6}px,0)`;
  }
  // ===== act F: iris from the risk point, the mark turns through a key light =====
  const lock=$('lock'), inF=f>=C.iris; vis(lock,inF);
  if(inF){const ip=prog(f,C.iris,C.iris+20,E.lin), cx=tw(f,C.iris,C.iris+20,1300+500*D.risk/0.08,960,E.smooth), cy=tw(f,C.iris,C.iris+20,733,540,E.smooth);
    const R=lmix(14,2300,ip); lock.style.clipPath=f<C.iris+21?`circle(${R}px at ${cx}px ${cy}px)`:'none';
    const turn=prog(f,C.turn,C.turn+40,E.outSoft), creep=lmix(1,1.03,prog(f,C.turn,C.end,E.lin));
    const rot=`rotateY(${-35*(1-turn)}deg) rotateX(${20*(1-turn)}deg)`; $('mark').style.transform=rot; $('glintmask').style.transform=rot; $('mark').style.opacity=prog(f,C.turn,C.turn+10,E.smooth);
    put($('markwrap'),330+(1-creep)*400,230,creep,1);
    const gl=prog(f,C.glint,C.glint+14,E.smooth); $('glint').style.transform=`translateX(${-200+900*gl}px)`; $('glint').style.opacity=0.5*Math.sin(Math.PI*gl);
    const w=prog(f,C.word,C.word+26,E.out); put($('wm'),980+(creep-1)*300,402+0.32*150*(1-w),creep,w);
    const t=prog(f,C.line,C.line+26,E.out); put($('tag'),986+(creep-1)*300,580+14*(1-t),creep,t);
    const u=prog(f,C.url,C.url+20,E.out); put($('url'),986+(creep-1)*300,668+10*(1-u),creep,u);
  }
  renderCopy(f);
}
// columns for the transition
for(let i=0;i<15;i++){const c=document.createElement('div');c.className='col';$('cols').appendChild(c)}
window.render=render; window.TOTAL=C.end;
window.ready=document.fonts.ready.then(()=>Promise.all([...document.images].map(i=>i.complete?1:new Promise(r=>{i.onload=i.onerror=r}))));
</script></body></html>
"""

if __name__ == "__main__":
    mask = "data:image/png;base64," + base64.b64encode((HERE.parent / "shark-mask.png").read_bytes()).decode()
    html = HTML.replace("__MASK__", mask).replace("__DATA__", json.dumps(DATA)).replace("__CUE__", json.dumps(CUE))
    (HERE / "film.html").write_text(html, encoding="utf-8")
    (HERE / "cues.json").write_text(json.dumps({"fps": 60, "bpm": 120, "cue": CUE, "data": DATA}, indent=1), encoding="utf-8")
    print("film.html written;", CUE["end"], "frames;", DATA)
