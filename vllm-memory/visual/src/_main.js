/* ---------- the notebook's constants (cell 1) and formulas ---------- */
const GiB=1024**3,MiB=1024**2,KiB=1024;
const C={GPU:79.6,BW:3.35,TF:989.0,PARAMS:8.03e9,LAYERS:32,HIDDEN:4096,INTER:14336,VOCAB:128256,QH:32,KVH:8,HD:128,UTIL:0.92,MML:32768,MNS:256,MBT:8192,BLOCK:16,BW_B:2,BKV:2,NONTORCH:0.3,PEAK:4.0,GRAPHS:1.0,CTX:0.5,EFF:0.6};
const kvPerToken=bkv=>C.LAYERS*2*C.KVH*C.HD*bkv;                  // cell 7
const weightsGiB=C.PARAMS*C.BW_B/GiB;                                 // cell 16
function pool(p){const req=C.GPU*p.util,kv=req-weightsGiB-C.NONTORCH-p.peak,kt=kvPerToken(p.bkv);   // cells 16 + 20
  const blocks=Math.floor(kv*GiB/(C.BLOCK*kt));return{req,kv,blocks,tokens:blocks*C.BLOCK,conc:blocks*C.BLOCK/C.MML,kt};}
const prefillMs=n=>2*C.PARAMS*n/(C.TF*1e12*C.EFF)*1e3;                // cell 24
function decodeMs(kvRead,n){const bytes=C.PARAMS*C.BW_B+kvRead*kvPerToken(C.BKV),comp=2*C.PARAMS*(n||1)/(C.TF*1e12*C.EFF);return Math.max(bytes/(C.BW*1e12*C.EFF),comp)*1e3;}
const P={util:0.92,bkv:2,peak:4.0,mns:256};const listeners=[];const changed=()=>listeners.forEach(f=>f());
const f0=n=>Math.round(n).toLocaleString('en-US');

function axes(svg,o){const lx=v=>o.xlog?Math.log(v):v,ly=v=>o.ylog?Math.log(v):v;
  const X=v=>o.x+(lx(v)-lx(o.xmin))/(lx(o.xmax)-lx(o.xmin))*o.w,Y=v=>o.y+o.h-(ly(v)-ly(o.ymin))/(ly(o.ymax)-ly(o.ymin))*o.h;
  (o.yticks||[]).forEach(v=>{S('line',{x1:o.x,x2:o.x+o.w,y1:Y(v),y2:Y(v),style:'stroke:var(--line);stroke-width:1'},svg);T(svg,o.x-6,Y(v)+4,(o.yfmt||String)(v),{'text-anchor':'end',style:'font-size:10.5px;fill:var(--muted)'});});
  (o.xticks||[]).forEach(v=>{S('line',{x1:X(v),x2:X(v),y1:o.y+o.h,y2:o.y+o.h+4,style:'stroke:var(--muted)'},svg);T(svg,X(v),o.y+o.h+17,(o.xfmt||String)(v),{'text-anchor':'middle',style:'font-size:10.5px;fill:var(--muted)'});});
  S('line',{x1:o.x,x2:o.x+o.w,y1:o.y+o.h,y2:o.y+o.h,style:'stroke:var(--muted)'},svg);
  if(o.xlabel)T(svg,o.x+o.w/2,o.y+o.h+34,o.xlabel,{'text-anchor':'middle',class:'lbl'});
  if(o.ylabel)T(svg,o.x,o.y-10,o.ylabel,{class:'lbl'});return{X,Y};}
const path=(pts)=>pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+' '+p[1].toFixed(1)).join(' ');

/* ---------- 00 example cards ---------- */
(function(){const cards=[['GPU','var(--g0)',[['card','NVIDIA H100 80 GB'],['memory PyTorch sees','79.6 GiB'],['memory bandwidth','3.35 TB/s'],['BF16 compute','989 TFLOP/s']]],
  ['Model','var(--g1)',[['name','Llama-3.1-8B-Instruct'],['parameters','8.03 billion'],['layers','32'],['hidden size','4,096'],['MLP width','14,336'],['vocabulary','128,256 tokens'],['attention heads','32 query · 8 key/value · 128 wide']]],
  ['vLLM flags','var(--g3)',[['--gpu-memory-utilization','0.92'],['--max-model-len','32,768'],['--max-num-seqs','256'],['--max-num-batched-tokens','8,192'],['--block-size','16'],['weights and KV cache','BF16, 2 bytes per number']]]];
  $('#ex-cards').innerHTML=cards.map(([t,c,rows])=>`<div class="excard" style="--k:${c}"><b>${t}</b><dl>${rows.map(r=>`<dt>${r[0]}</dt><dd>${r[1]}</dd>`).join('')}</dl></div>`).join('');})();

/* ---------- 01 weights ---------- */
(function(){const svg=$('#w-svg'),{A}=makeKit(svg);const precs=[['FP32',4,'var(--idle)'],['BF16 (this example)',2,'var(--accent)'],['FP8 / INT8',1,'var(--g3)'],['INT4 / NVFP4',0.5,'var(--g2)']];
  const x0=170,W=560,sc=W/80;
  precs.forEach(([n,b,c],i)=>{const v=C.PARAMS*b/GiB,y=30+i*46;T(svg,x0-10,y+20,n,{'text-anchor':'end',style:'font-weight:700'});R(svg,x0,y,v*sc,30,`fill:${c};${A(i*.1)}`,{rx:4});T(svg,x0+v*sc+8,y+20,fmt(v,1)+' GiB',{style:'font-weight:700'});});
  const bx=x0+C.GPU*C.UTIL*sc;S('line',{x1:bx,x2:bx,y1:18,y2:212,style:'stroke:var(--ink);stroke-width:1.5;stroke-dasharray:5 4'},svg);T(svg,bx-6,232,'budget at util 0.92 = '+fmt(C.GPU*C.UTIL,1)+' GiB',{'text-anchor':'end',style:'font-size:11px;font-weight:700'});
  const px=x0+C.GPU*sc;S('line',{x1:px,x2:px,y1:18,y2:212,style:'stroke:var(--bad);stroke-width:1.5;stroke-dasharray:2 3'},svg);T(svg,px,14,'79.6 GiB',{'text-anchor':'end',style:'font-size:11px;fill:var(--bad);font-weight:700'});})();

/* ---------- 02 KV per token ---------- */
(function(){const svg=$('#kv-svg'),r=$('#kv-r'),kt=kvPerToken(2);
  function render(){clear(svg);const n=+r.value;$('#kv-n').textContent=f0(n);
    T(svg,20,22,"ONE TOKEN'S KV ENTRY",{class:'lbl'});
    for(let l=0;l<C.LAYERS;l++){const y=292-l*8.2;R(svg,30,y,46,7,'fill:var(--g1)');R(svg,80,y,46,7,'fill:var(--g0)');}
    T(svg,53,310,'K',{'text-anchor':'middle',style:'font-weight:700'});T(svg,103,310,'V',{'text-anchor':'middle',style:'font-weight:700'});
    T(svg,134,296,'layer 1',{style:'font-size:10.5px;fill:var(--muted)'});T(svg,134,42,'layer 32',{style:'font-size:10.5px;fill:var(--muted)'});
    T(svg,20,328,'32 × (K + V) × 8 heads × 128 × 2 B = 128 KiB',{style:'font-size:11px;font-weight:700'});
    const o=axes(svg,{x:270,y:40,w:460,h:230,xmin:0,xmax:C.MML,ymin:0,ymax:4.5,xticks:[0,8192,16384,24576,32768],xfmt:v=>v?(v/1024)+'K':'0',yticks:[0,1,2,3,4],yfmt:v=>v+' GiB',xlabel:'tokens in the request (prompt + answer so far)',ylabel:'KV CACHE HELD BY ONE REQUEST'});
    S('path',{d:path([[o.X(0),o.Y(0)],[o.X(C.MML),o.Y(C.MML*kt/GiB)]]),style:'stroke:var(--g3);stroke-width:3;fill:none'},svg);
    [1000,8192,32768].forEach(t=>{S('circle',{cx:o.X(t),cy:o.Y(t*kt/GiB),r:3.5,style:'fill:var(--muted)'},svg);});
    const v=n*kt/GiB;S('line',{x1:o.X(n),x2:o.X(n),y1:o.Y(0),y2:o.Y(v),style:'stroke:var(--accent);stroke-dasharray:3 3'},svg);
    S('circle',{cx:o.X(n),cy:o.Y(v),r:7,style:'fill:var(--accent)'},svg);
    const lab=v<1?fmt(n*kt/MiB,0)+' MiB':fmt(v,2)+' GiB';T(svg,Math.min(o.X(n)+10,640),o.Y(v)-12,`${f0(n)} tokens → ${lab}`,{style:'font-weight:800'});
    stats($('#kv-stats'),[{v:'4,096 B',l:'per token, per layer'},{v:'128 KiB',l:'per token, all 32 layers'},{v:'125 MiB',l:'a 1,000-token prompt'},{v:'4.00 GiB',l:'one full 32,768-token request'}]);}
  r.addEventListener('input',render);render();})();

/* ---------- 03 startup ---------- */
(function(){const svg=$('#su-svg'),{A}=makeKit(svg);const b=pool({util:.92,bkv:2,peak:4.0});
  function mem(t){let m=0;if(t>=2)m=C.CTX;if(t>=6&&t<26)m=C.CTX+weightsGiB*(t-6)/20;if(t>=26)m=C.CTX+weightsGiB+C.NONTORCH;
    if(t>=30&&t<33)m+=C.PEAK*Math.sin(Math.PI*(t-30)/3);if(t>=36)m+=b.kv;if(t>=42)m+=C.GRAPHS;return m;}
  const PH=[['A','Init',2,'Python imports, the CUDA context is created (plus NCCL if there are several GPUs), and vLLM takes a <b>snapshot</b> of free memory. A few hundred MiB, outside the budget.'],
    ['B','Budget',4,'requested = total × util = 79.6 × 0.92 = <b>73.2 GiB</b>. If less than that is free right now, vLLM refuses to start. This fixes the ceiling.'],
    ['C','Load weights',6,'The checkpoint streams from disk onto the GPU: <b>+14.96 GiB</b>, permanent.'],
    ['D','Profile run',30,'One <b>dummy step</b> at the worst-case shape: 8,192 tokens over 256 fake requests, 32 tokens each, plus a dummy sampler over 256 × 128,256 scores. Memory spikes, is freed, and the spike height is <b>recorded</b> (4.0 GiB here, the notebook\'s placeholder).'],
    ['E','KV pool',36,'pool = 73.23 − 14.96 − 0.3 − 4.0 = <b>53.97 GiB</b>, allocated <b>all at once</b> as blocks, with zero requests in sight.'],
    ['F','CUDA graphs',42,'Decode kernels are recorded for a set of batch sizes, so each step launches as one unit: <b>+0.5 to 2 GiB</b> (1.0 GiB in the notebook), taken after the pool.'],
    ['G','Ready',50,'The HTTP server listens, and memory stays flat for the life of the server. That is why nvidia-smi shows about 73 GiB in use on an idle vLLM server.']];
  const ST=PH.map(([id,n,t,c],i)=>({n:id+' · '+n,f:'memory in use over the first 60 s after launch',c,d(){
      const o=axes(svg,{x:60,y:30,w:670,h:250,xmin:0,xmax:60,ymin:0,ymax:86,xticks:[0,10,20,30,40,50,60],xfmt:v=>v+' s',yticks:[0,20,40,60,80],yfmt:v=>v,xlabel:'seconds after launch',ylabel:'GPU MEMORY IN USE (GiB)'});
      const tEnd=i<PH.length-1?PH[i+1][2]:60;
      S('rect',{x:o.X(t),y:30,width:o.X(tEnd)-o.X(t),height:250,style:'fill:var(--accent-soft);opacity:.7'},svg);
      S('line',{x1:o.X(0),x2:o.X(60),y1:o.Y(C.GPU),y2:o.Y(C.GPU),style:'stroke:var(--bad);stroke-dasharray:2 3'},svg);T(svg,o.X(60),o.Y(C.GPU)-5,'physical 79.6 GiB',{'text-anchor':'end',style:'font-size:10.5px;fill:var(--bad)'});
      S('line',{x1:o.X(0),x2:o.X(60),y1:o.Y(b.req),y2:o.Y(b.req),style:`stroke:var(--ink);stroke-dasharray:5 4;stroke-width:${id==='B'?2.5:1}`},svg);T(svg,o.X(0)+4,o.Y(b.req)-5,'budget ceiling 73.2 GiB',{style:`font-size:10.5px;font-weight:${id==='B'?800:500}`});
      const pts=[];for(let s=0;s<=tEnd;s+=.05)pts.push([o.X(s),o.Y(mem(s))]);
      S('path',{d:path(pts),style:'stroke:var(--accent);stroke-width:3;fill:none'},svg);
      PH.forEach(([pid,,pt],j)=>{if(j<=i){S('line',{x1:o.X(pt),x2:o.X(pt),y1:30,y2:280,style:'stroke:var(--muted);stroke-dasharray:2 3;opacity:.6'},svg);T(svg,o.X(pt)+3,272,pid,{style:`font-weight:800;fill:${j===i?'var(--accent)':'var(--muted)'}`});}});
      const now=mem(id==='D'?31.5:tEnd-.01);const g=S('g',{style:A(.2)},svg);T(g,o.X(Math.min(tEnd,52)),o.Y(now)-(now<10?30:10),fmt(now,1)+' GiB',{'text-anchor':'middle',style:'font-weight:800'});}}));
  stager({svg,note:$('#su-note'),chips:$('#su-chips'),prev:$('#su-prev'),next:$('#su-next'),play:$('#su-play'),stages:ST});})();

/* ---------- 04 KV pool ---------- */
(function(){const svg=$('#pl-svg');
  const peaks=[['Qwen2.5-0.5B',0.09],['Qwen3-0.6B',0.52],['Gemma-3-27B 4K',1.41],['DeepSeek-R1 TP8',1.52],['this example',4.0],['Gemma-3-27B 131K, old profiler',17.91]];
  function render(){clear(svg);const b=pool(P);$('#pl-uv').textContent=P.util.toFixed(2);
    const segs=[['weights',weightsGiB,'var(--accent)'],['non-torch',C.NONTORCH,'var(--muted)'],['activation peak',P.peak,'var(--g1)'],['CUDA graphs',C.GRAPHS,'var(--g2)'],['KV cache pool',b.kv-C.GRAPHS,'var(--g3)'],['outside (1 − util)',C.GPU-b.req,'url(#hatchBad)']];
    const x0=20,W=720,sc=W/C.GPU;let x=x0,small=0;
    segs.forEach(([n,v,c])=>{const w=v*sc;R(svg,x,60,w,56,`fill:${c}`);
      if(w>90){T(svg,x+w/2,84,n,{'text-anchor':'middle',style:'fill:var(--on-fill);font-weight:700;font-size:12px'});T(svg,x+w/2,102,fmt(v,1)+' GiB',{'text-anchor':'middle',style:'fill:var(--on-fill);font-size:11.5px'});}
      else{const slot=[[34,'end'],[150,'middle'],[18,'start'],[150,'end']][small%4],ly=slot[0],up=ly<60,cx=x+w/2;let anc=slot[1];if(cx>620)anc='end';if(cx<110)anc='start';
        S('path',{d:`M${cx} ${up?60:116} V${up?ly+4:ly-13}`,style:'stroke:var(--muted);fill:none'},svg);T(svg,anc==='end'?cx+6:anc==='start'?cx-6:cx,ly,`${n} ${fmt(v,1)} GiB`,{'text-anchor':anc,style:'font-size:11px;font-weight:700'});small++;}
      x+=w;});
    const bx=x0+b.req*sc;S('line',{x1:bx,x2:bx,y1:28,y2:130,style:'stroke:var(--ink);stroke-width:2;stroke-dasharray:5 4'},svg);T(svg,bx,186,`budget ${fmt(b.req,1)} GiB`,{'text-anchor':'end',style:'font-size:11px;font-weight:700'});
    T(svg,x0,186,'0',{style:'font-size:10.5px;fill:var(--muted)'});T(svg,x0+W,200,'79.6 GiB',{'text-anchor':'end',style:'font-size:10.5px;fill:var(--muted)'});
    $('#pl-log').textContent=`GPU KV cache size: ${f0(b.tokens)} tokens   (${f0(b.blocks)} blocks of 16)\nMaximum concurrency for 32,768 tokens per request: ${b.conc.toFixed(2)}x`;
    stats($('#pl-stats'),[{v:fmt(b.kv,2)+' GiB',l:'KV pool = budget − weights − peak − 0.3'},{v:(b.kt/KiB)+' KiB',l:'KV per token'},{v:f0(b.tokens),l:'tokens resident across all requests',c:'good'},{v:b.conc.toFixed(2)+'×',l:'full 32,768-token requests at once'}]);}
  $('#pl-u').addEventListener('input',e=>{P.util=+e.target.value;changed();});
  seg($('#pl-kv'),[{l:'KV cache BF16 (2 B)',v:2},{l:'FP8 (1 B)',v:1}],2,v=>{P.bkv=v;changed();});
  seg($('#pl-peak'),peaks.map(([l,v])=>({l:`${l} · ${v}`,v})),4.0,v=>{P.peak=v;changed();});
  listeners.push(render);render();})();

/* ---------- 05 one request ---------- */
(function(){const svg=$('#rq-svg'),{A}=makeKit(svg);const b0=pool({util:.92,bkv:2,peak:4.0});
  const ttft=prefillMs(1000),itl=decodeMs(1001,1);let tot=ttft;for(let s=1;s<=300;s++)tot+=decodeMs(1000+s,1);
  function grid(state){T(svg,20,26,'THE REQUEST\'S KV BLOCKS (16 tokens each)',{class:'lbl'});
    for(let i=0;i<84;i++){const x=20+(i%12)*28,y=36+Math.floor(i/12)*30;let st='fill:none;stroke:var(--line);stroke-dasharray:2 2';
      if(state==='free'&&i<82)st='fill:url(#hatch);stroke:var(--line)';
      else if(i<63&&state!=='none')st=`fill:var(--g1);${state==='claim'?A(i*.008):''}`;else if(i>=63&&i<82&&state==='decode')st=`fill:var(--g0);${A(.3+(i-63)*.04)}`;
      R(svg,x,y,24,24,st,{rx:3});}
    T(svg,20,262,state==='none'?'none yet':state==='free'?'82 blocks returned to the free list':state==='decode'?'63 from the prompt + 19 grown while decoding = 82':'63 blocks = 1,008 slots for 1,000 tokens',{style:'font-weight:700'});}
  const pill=(x,y,w,t,col,d)=>{const g=S('g',{style:A(d||0)},svg);R(g,x,y,w,30,`fill:${col}`,{rx:8});T(g,x+w/2,y+20,t,{'text-anchor':'middle',style:'fill:var(--on-fill);font-weight:700;font-size:12px'});};
  function timeline(upto){const x0=400,W=330;T(svg,x0,170,'TIME',{class:'lbl'});
    const sc=W/tot,pw=Math.max(ttft*sc,6);R(svg,x0,180,pw,26,'fill:var(--g1)');T(svg,x0,224,'prefill '+fmt(ttft,0)+' ms = TTFT',{style:'font-size:11px;font-weight:700'});
    if(upto>=6){R(svg,x0+pw,180,W-pw,26,`fill:var(--g0);${A(.1)}`);T(svg,x0+W,224,'300 decodes × ~'+fmt(itl,1)+' ms',{'text-anchor':'end',style:'font-size:11px;font-weight:700'});T(svg,x0+W,244,'total ≈ '+f0(tot)+' ms',{'text-anchor':'end',style:'font-size:11px;fill:var(--muted)'});}}
  const ST=[
   {n:'Tokenize',f:'text → 1,000 token IDs on the CPU',c:'The text becomes 1,000 token IDs. vLLM checks that 1,000 + 300 ≤ <code>--max-model-len</code>, or it rejects the request.',d(){grid('none');pill(400,60,150,'1,000-token prompt','var(--muted)');pill(580,60,150,'1,000 IDs','var(--accent)',.3);T(svg,565,80,'→',{'text-anchor':'middle',class:'big'});}},
   {n:'Prefix cache',f:'hash the prompt in 16-token blocks',c:'The prompt is hashed in 16-token blocks. If an earlier request began with the same blocks (a shared system prompt, say), those blocks are reused and skipped. Here nothing matches.',d(){grid('none');for(let i=0;i<8;i++)pill(400+i*42,60,36,'#','var(--idle)',i*.05);T(svg,400,120,'63 block hashes, no match in the cache',{style:'font-weight:700'});}},
   {n:'Queue',f:'join the waiting list',c:'The request enters the waiting list.',d(){grid('none');T(svg,400,50,'WAITING',{class:'lbl'});pill(400,60,110,'this request','var(--accent)',.1);}},
   {n:'Schedule',f:'ceil(1000 / 16) = 63 blocks',c:'The scheduler sees room in the pool and in the step budget. It claims <b>63 blocks</b> and puts all 1,000 tokens into this step.',d(){grid('claim');T(svg,400,60,'63 of the pool\'s',{style:'font-weight:700'});T(svg,400,80,f0(b0.blocks)+' blocks claimed',{style:'font-weight:700'});T(svg,400,104,'= '+fmt(63/b0.blocks*100,2)+'% of the pool',{style:'fill:var(--muted)'});}},
   {n:'Prefill',f:'TTFT ≈ 2 × 8.03e9 × 1000 FLOP / (989 TFLOP/s × 0.6)',c:'One forward pass pushes all 1,000 tokens through 32 layers, and each layer writes K and V for every position into the 63 blocks. The first output token is sampled. Prefill is limited by <b>compute</b>: <b>TTFT ≈ 27 ms</b>.',d(){grid('claim');timeline(5);}},
   {n:'Decode',f:'ITL ≈ (15.0 GiB weights + KV) / (3.35 TB/s × 0.6)',c:'300 times: 1 token, one pass that reads all the weights and all cached K and V, and a new block every 16 tokens. Decode is limited by <b>memory bandwidth</b>: <b>ITL ≈ 8.1 ms</b>, about 124 tokens per second for one request.',d(){grid('decode');timeline(6);}},
   {n:'Finish',f:'82 blocks = 164 MiB = 0.30% of the pool',c:'After 300 tokens, all <b>82 blocks</b> (164 MiB) go back to the free list at once, ready for the next request.',d(){grid('free');timeline(6);}}];
  stager({svg,note:$('#rq-note'),chips:$('#rq-chips'),prev:$('#rq-prev'),next:$('#rq-next'),play:$('#rq-play'),stages:ST});})();

/* ---------- 06 scheduler ---------- */
(function(){const svg=$('#sc-svg'),cv=$('#sc-grid'),ctx=cv.getContext('2d'),r=$('#sc-r');let sc='sim60',timer=null;
  const css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const snapKeys=Object.keys(SIM.sim60.snaps).map(Number).sort((a,b)=>a-b);
  const hsl=i=>`hsl(${(i*47+205)%360} 70% 55%)`;
  function drawGrid(step){let k=snapKeys[0];for(const s of snapKeys)if(s<=step)k=s;const tb=SIM.sim60.snaps[k];const cw=12,ch=12;
    ctx.clearRect(0,0,cv.width,cv.height);const free=css('--surface-2');
    for(let i=0;i<2400;i++){const v=tb.charCodeAt(i)-48;ctx.fillStyle=v?hsl(v-1):free;ctx.fillRect((i%120)*cw,Math.floor(i/120)*ch,cw-1.5,ch-1.5);}
    const row=SIM.sim60.log[step];$('#sc-gl').textContent=`BLOCK TABLE, FIRST 2,400 OF ${f0(SIM.kv_blocks)} BLOCKS (snapshot at step ${k}) · one colour per request · grey = free`;}
  function render(){const d=SIM[sc],L=d.log,i=+r.value,row=L[Math.min(i,L.length-1)];clear(svg);
    const steps=L.map(q=>q[0]),xmax=steps[steps.length-1],pool=sc==='sim60'?SIM.kv_blocks:d.pool;
    const ymax=sc==='sim60'?100:100;
    const o=axes(svg,{x:56,y:30,w:680,h:220,xmin:0,xmax,ymin:0,ymax,xticks:[0,.25,.5,.75,1].map(f=>Math.round(xmax*f)),xfmt:v=>f0(v),yticks:[0,25,50,75,100],xlabel:'scheduler step',ylabel:'REQUESTS (running · waiting) AND POOL USED (%)'});
    const line=(idx,col,dash,f)=>S('path',{d:path(L.map(q=>[o.X(q[0]),o.Y(f?f(q[idx]):q[idx])])),style:`stroke:${col};stroke-width:2.2;fill:none;${dash?'stroke-dasharray:5 4':''}`},svg);
    line(3,'var(--g3)');line(4,'var(--bad)');line(5,'var(--accent)',1,v=>v/pool*100);
    S('line',{x1:o.X(row[0]),x2:o.X(row[0]),y1:30,y2:250,style:'stroke:var(--ink);stroke-width:1.5'},svg);
    [['running','var(--g3)'],['waiting','var(--bad)'],['pool used %','var(--accent)']].forEach(([t,c],k)=>{R(svg,470+k*95,8,12,4,`fill:${c}`);T(svg,486+k*95,13,t,{style:'font-size:11px'});});
    $('#sc-sv').textContent=f0(row[0]);
    const st=[{v:f0(row[0]),l:'step'},{v:row[3],l:'running'},{v:row[4],l:'waiting'},{v:fmt(row[5]/pool*100,1)+'%',l:'pool used'}];
    if(sc==='sim60'){st.push({v:f0(row[1])+' + '+row[2],l:'prefill + decode tokens in this step'});drawGrid(row[0]);}else st.push({v:d.pre,l:'preemptions over the run'});
    stats($('#sc-stats'),st);}
  function set(v){sc=v;stop();const d=SIM[v];r.max=d.log.length-1;r.value=v==='sim60'?12:Math.round((d.log.length-1)*.3);$('#sc-gridwrap').style.display=v==='sim60'?'':'none';
    $('#sc-note').innerHTML=v==='sim60'?'60 requests arrive together, with prompts of 500 to 3,000 tokens and answers of 150 to 400. For the first ~10 steps the prompts are chewed through at the 8,192-token cap. After that, each step is one token per running request. All done in <b>386 steps</b>, with <b>0 preemptions</b> and a peak pool use of <b>26.1%</b>.':'40 requests each grow to about 20,000 tokens, on a deliberately small pool of 20,000 blocks. Pool use pins at 100%. When a running request needs a block and none is free, the scheduler evicts the newest request and recomputes it later. That gives <b>471 preemptions</b>, but all 40 finish in <b>43,080 steps</b>, and the server never allocates beyond its pool.';render();}
  function stop(){clearInterval(timer);timer=null;$('#sc-play').textContent='Play';}
  $('#sc-play').onclick=()=>{if(timer){stop();return;}const n=+r.max;if(+r.value>=n)r.value=0;$('#sc-play').textContent='Pause';const inc=Math.max(1,Math.round(n/240));
    timer=setInterval(()=>{r.value=Math.min(n,+r.value+inc);render();if(+r.value>=n)stop();},RM?200:45);};
  r.addEventListener('input',()=>{stop();render();});
  seg($('#sc-seg'),[{l:'60 requests at once',v:'sim60'},{l:'Pool too small → preemption',v:'sim4'}],'sim60',set);set('sim60');})();

/* ---------- 07 limits ---------- */
(function(){const svg=$('#lm-svg');const lens=[256,512,1000,2000,4000,8000,16000,32768];
  function render(){clear(svg);const b=pool(P),mns=P.mns;
    const o=axes(svg,{x:70,y:52,w:660,h:210,xmin:256,xmax:32768,xlog:1,ymin:5,ymax:3000,ylog:1,xticks:lens,xfmt:v=>v>=1000?(v/1000)+'K':v,yticks:[10,30,100,300,1000,3000],xlabel:'average tokens per request (prompt + answer)',ylabel:'REQUESTS RESIDENT AT ONCE (log)'});
    const byKv=lens.map(l=>[l,b.tokens/l]);
    S('path',{d:path(lens.map(l=>[o.X(l),o.Y(Math.min(Math.max(Math.min(b.tokens/l,mns),5),3000))])),style:'stroke:var(--ink);stroke-width:9;opacity:.15;fill:none'},svg);
    S('path',{d:path(byKv.map(([l,v])=>[o.X(l),o.Y(Math.min(v,3000))])),style:'stroke:var(--g3);stroke-width:2.5;fill:none'},svg);
    byKv.forEach(([l,v])=>v<=3000&&S('circle',{cx:o.X(l),cy:o.Y(v),r:3.5,style:'fill:var(--g3)'},svg));
    S('line',{x1:o.X(256),x2:o.X(32768),y1:o.Y(mns),y2:o.Y(mns),style:'stroke:var(--bad);stroke-width:2;stroke-dasharray:6 4'},svg);
    const cx=b.tokens/mns;if(cx>256&&cx<32768){S('circle',{cx:o.X(cx),cy:o.Y(mns),r:7,style:'fill:none;stroke:var(--ink);stroke-width:2'},svg);T(svg,o.X(cx),o.Y(mns)-14,`crossover ≈ ${f0(cx)} tokens`,{'text-anchor':'middle',style:'font-weight:800;font-size:12px'});}
    [['KV pool ÷ tokens per request','var(--g3)'],['--max-num-seqs = '+mns,'var(--bad)'],['effective = the smaller','var(--ink)']].forEach(([t,c],k)=>{const x=[130,345,525][k];R(svg,x,8,14,4,`fill:${c}`);T(svg,x+18,13,t,{style:'font-size:10.5px'});});
    stats($('#lm-stats'),[{v:f0(cx),l:'crossover length (tokens)'},{v:f0(Math.min(b.tokens/1300,mns)),l:'at 1,300 tokens per request'},{v:fmt(b.tokens/32768,1),l:'at a full 32,768 tokens'}]);}
  seg($('#lm-seg'),[128,256,512,1024].map(v=>({l:String(v),v})),256,v=>{P.mns=v;render();});listeners.push(render);render();})();

/* ---------- 08 batching ---------- */
(function(){const svg=$('#bt-svg');const ns=[1,2,4,8,16,32,64,128,256,512,1024],ctxT=1500;
  const ms=ns.map(n=>decodeMs(n*ctxT,n)),tps=ns.map((n,i)=>n/ms[i]*1000);
  const a=axes(svg,{x:60,y:40,w:290,h:200,xmin:1,xmax:1024,xlog:1,ymin:0,ymax:Math.ceil(Math.max(...ms)/20)*20+10,xticks:[1,4,16,64,256,1024],yticks:[0,20,40,60,80].filter(v=>v<=Math.ceil(Math.max(...ms)/20)*20+10),yfmt:v=>v+' ms',xlabel:'requests decoding in the step',ylabel:'STEP TIME'});
  S('path',{d:path(ns.map((n,i)=>[a.X(n),a.Y(ms[i])])),style:'stroke:var(--accent);stroke-width:2.5;fill:none'},svg);ns.forEach((n,i)=>S('circle',{cx:a.X(n),cy:a.Y(ms[i]),r:3.5,style:'fill:var(--accent)'},svg));
  const b=axes(svg,{x:440,y:40,w:290,h:200,xmin:1,xmax:1024,xlog:1,ymin:0,ymax:16000,xticks:[1,4,16,64,256,1024],yticks:[0,4000,8000,12000,16000],yfmt:v=>v/1000+'K',xlabel:'requests decoding in the step',ylabel:'OUTPUT TOKENS / SECOND, ALL USERS'});
  S('path',{d:path(ns.map((n,i)=>[b.X(n),b.Y(tps[i])])),style:'stroke:var(--g3);stroke-width:2.5;fill:none'},svg);ns.forEach((n,i)=>S('circle',{cx:b.X(n),cy:b.Y(tps[i]),r:3.5,style:'fill:var(--g3)'},svg));
  T(svg,a.X(256),a.Y(ms[8])-12,fmt(ms[8],1)+' ms',{'text-anchor':'middle',style:'font-weight:800;font-size:11.5px'});T(svg,a.X(1),a.Y(ms[0])-12,fmt(ms[0],1)+' ms',{'text-anchor':'start',style:'font-weight:800;font-size:11.5px'});
  T(svg,b.X(256),b.Y(tps[8])-12,f0(tps[8])+' tok/s',{'text-anchor':'end',style:'font-weight:800;font-size:11.5px'});
  stats($('#bt-stats'),[{v:f0(tps[0]),l:'tokens/s, one user alone'},{v:f0(tps[8]),l:'tokens/s total, 256 users',c:'good'},{v:f0(tps[8]/256),l:'tokens/s each, 256 users'},{v:'1,500',l:'tokens of context per request (assumed)'}]);})();

/* ---------- 09 run the notebook's code (Pyodide) ---------- */
(function(){const ta=$('#py-code'),out=$('#py-out'),st=$('#py-status');ta.value=PYCODE;let py=null,loading=null;
  function load(){if(loading)return loading;st.textContent='loading Python (about 10 MB, first run only)…';
    loading=new Promise((res,rej)=>{const s=document.createElement('script');s.src='https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js';s.onload=res;s.onerror=()=>rej(new Error('could not load Pyodide'));document.head.appendChild(s);})
      .then(()=>loadPyodide({indexURL:'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/'})).then(p=>{py=p;st.textContent='loading NumPy…';return p.loadPackage('numpy');}).then(()=>py);
    return loading;}
  $('#py-run').onclick=async()=>{$('#py-run').disabled=true;try{await load();st.textContent='running…';let buf='';py.setStdout({batched:s=>{buf+=s+'\n';}});py.setStderr({batched:s=>{buf+=s+'\n';}});
      const t0=performance.now();await py.runPythonAsync(ta.value);out.textContent=buf||'(no output)';st.textContent=`done in ${((performance.now()-t0)/1000).toFixed(1)} s · Python ${py.version}`;}
    catch(e){out.textContent=String(e.message||e);st.textContent=py?'error in the code (see output)':'Python could not load here. Use the Colab link above, or open this page from GitHub Pages.';loading=py?loading:null;}
    finally{$('#py-run').disabled=false;}};
  $('#py-reset').onclick=()=>{ta.value=PYCODE;};})();

/* ---------- code panels ---------- */
Object.entries(CODE).forEach(([k,v])=>{const el=document.getElementById('code-'+k);if(el)el.textContent=v;});
