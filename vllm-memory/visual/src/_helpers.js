
const NS="http://www.w3.org/2000/svg";
const $=q=>document.querySelector(q);
function S(tag,a,p){const e=document.createElementNS(NS,tag);if(a)for(const k in a)e.setAttribute(k,a[k]);if(p)p.appendChild(e);return e;}
function T(p,x,y,str,a){const t=S('text',Object.assign({x,y},a||{}),p);t.textContent=str;return t;}
function R(p,x,y,w,h,style,extra){return S('rect',Object.assign({x,y,width:Math.max(0,w),height:Math.max(0,h),style},extra||{}),p);}
function clear(e){while(e.firstChild)e.removeChild(e.firstChild);}
const RM=matchMedia('(prefers-reduced-motion: reduce)').matches;
const G=i=>`var(--g${i})`;
const PAL=i=>`hsl(${(i*47+205)%360} 70% 55%)`;
function seg(box,opts,init,cb){box.innerHTML='';opts.forEach(o=>{const b=document.createElement('button');b.type='button';b.className='btn';b.textContent=o.l;b.setAttribute('aria-pressed',String(o.v===init));b.onclick=()=>{[...box.children].forEach(c=>c.setAttribute('aria-pressed','false'));b.setAttribute('aria-pressed','true');cb(o.v);};box.appendChild(b);});}
function stats(box,arr){box.innerHTML=arr.map(s=>`<div class="stat ${s.c||''}"><b>${s.v}</b><span>${s.l}</span></div>`).join('');}
function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;}}
function fmt(n,d){return n.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});}


/* ---------- shared drawing kit ---------- */
function makeKit(svg){
  const c30=Math.cos(Math.PI/6);
  const A=d=>RM?'':`animation:pdIn .6s cubic-bezier(.2,.8,.2,1) ${d}s both;transform-box:fill-box;transform-origin:center;`;
  function iso(ox,oy,X,Y,Z,u,col,o){o=o||{};
    const P=(x,y,z)=>[ox+(x-y)*c30*u,oy+(x+y)*.5*u-z*u];
    const pts=a=>a.map(p=>P(...p).map(v=>v.toFixed(1)).join(',')).join(' ');
    const g=S('g',{style:o.an||''},svg);const segs=o.segs||[{x0:0,x1:X,col}];
    segs.forEach(sg=>[[[sg.x0,0,Z],[sg.x1,0,Z],[sg.x1,Y,Z],[sg.x0,Y,Z],.95],[[sg.x0,Y,0],[sg.x1,Y,0],[sg.x1,Y,Z],[sg.x0,Y,Z],.72]].forEach(f=>S('polygon',{points:pts(f.slice(0,4)),style:`fill:${sg.col};fill-opacity:${f[4]};stroke:${sg.col}`},g)));
    const lc=segs[segs.length-1].col;S('polygon',{points:pts([[X,0,0],[X,Y,0],[X,Y,Z],[X,0,Z]]),style:`fill:${lc};fill-opacity:.5;stroke:${lc}`},g);
    const Ln=(a,b)=>{const p=P(...a),q=P(...b);S('line',{x1:p[0],y1:p[1],x2:q[0],y2:q[1],style:'stroke:var(--surface);stroke-width:1;opacity:.8'},g);};
    for(let i=1;i<X;i++){Ln([i,0,Z],[i,Y,Z]);Ln([i,Y,0],[i,Y,Z]);}
    for(let j=1;j<Y;j++){Ln([0,j,Z],[X,j,Z]);Ln([X,j,0],[X,j,Z]);}
    for(let z=1;z<Z;z++){Ln([0,Y,z],[X,Y,z]);Ln([X,0,z],[X,Y,z]);}
    if(o.lab){const a=P(X/2,Y,0),b=P(X,Y/2,0),c=P(X,0,Z/2),l=o.lab;
      T(g,a[0]-6,a[1]+18,l.x,{'text-anchor':'end',style:'font-weight:700;fill:var(--muted)'});
      T(g,b[0]+6,b[1]+18,l.y,{style:'font-weight:700;fill:var(--muted)'});
      T(g,c[0]+8,c[1]+4,l.z,{style:'font-weight:700;fill:var(--muted)'});}
    const cx=ox+(X-Y)*c30*u/2,top=oy-Z*u;
    if(o.title)T(g,cx,top-12,o.title,{'text-anchor':'middle',style:'font-weight:700'});
    if(o.tag){const w=o.tag.length*7.5+14;R(g,cx-w/2,oy+(X+Y)*.5*u+6,w,18,`fill:${o.tagc||'var(--accent)'}`,{rx:9});T(g,cx,oy+(X+Y)*.5*u+19,o.tag,{'text-anchor':'middle',style:'fill:var(--on-fill);font-size:10.5px;font-weight:700'});}
    return g;}
  function arrow(x1,y1,x2,y2,lab,d,col){const g=S('g',{style:A(d||0)},svg);
    S('line',{x1,y1,x2,y2,'marker-end':'url(#arr)',style:`stroke:${col||'var(--muted)'};stroke-width:1.8`},g);
    if(lab)T(g,(x1+x2)/2,(y1+y2)/2-8,lab,{'text-anchor':'middle',style:`font-size:12px;font-weight:600;fill:${col||'var(--accent)'}`});return g;}
  function heat(x,y,n,cs,seed,col,d){const r=mulberry32(seed),g=S('g',{style:A(d||0)},svg);
    for(let i=0;i<n;i++)for(let j=0;j<n;j++)R(g,x+j*cs,y+i*cs,cs-1,cs-1,j>i?'fill:url(#hatch)':`fill:${col};opacity:${.25+.75*r()}`);return g;}
  function mat(x,y,w,h,col,lab,sub,d){const g=S('g',{style:A(d||0)},svg);R(g,x,y,w,h,`fill:${col};opacity:.85`,{rx:4});
    if(lab)T(g,x+w/2,y+h/2+4,lab,{'text-anchor':'middle',style:'fill:var(--on-fill);font-weight:700;font-size:12px'});
    if(sub)T(g,x+w/2,y+h+15,sub,{'text-anchor':'middle',style:'font-size:11px;fill:var(--muted)'});return g;}
  function pill(x,y,t,st,d){const w=Math.max(40,t.length*8.5+18),g=S('g',{style:A(d||0)},svg);R(g,x,y,w,26,st||'fill:var(--accent-soft);stroke:var(--accent)',{rx:7});T(g,x+w/2,y+18,t,{'text-anchor':'middle',style:'font-weight:600'});return w;}
  const big=(x,y,t)=>T(svg,x,y,t,{class:'big'});
  return{A,iso,arrow,heat,mat,pill,big};
}
const HCg=i=>`hsl(${[215,330,160,38][i%4]} 72% 55%)`;
function stager(o){ /* o: {svg,note,chips,prev,next,play,stages,after} */
  let k=0,timer=null;
  function render(){clear(o.svg);const st=o.stages[k];st.d();
    o.note.innerHTML=`<b class="mono" style="color:var(--accent)">${k+1}/${o.stages.length} · ${st.n}</b><div class="formula" style="margin:6px 0">${st.f}</div>${st.c}`;
    o.chips.innerHTML='';o.stages.forEach((s,i)=>{const b=document.createElement('button');b.type='button';b.className='chip';b.textContent=(i+1)+' '+s.n;b.setAttribute('aria-pressed',String(i===k));b.onclick=()=>{stop();k=i;render();};o.chips.appendChild(b);});
    o.prev.disabled=k===0;o.next.disabled=k===o.stages.length-1;if(o.after)o.after(k);}
  function stop(){clearInterval(timer);timer=null;o.play.textContent='Play all';}
  o.prev.onclick=()=>{stop();if(k>0){k--;render();}};o.next.onclick=()=>{stop();if(k<o.stages.length-1){k++;render();}};
  o.play.onclick=()=>{if(timer){stop();return;}if(k>=o.stages.length-1)k=0;else k++;render();o.play.textContent='Pause';
    timer=setInterval(()=>{if(k>=o.stages.length-1){stop();return;}k++;render();},RM?400:4500);};
  render();return{render};
}


