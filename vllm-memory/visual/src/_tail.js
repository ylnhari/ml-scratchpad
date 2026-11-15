/* layout: split each figure section into a text column and a figure column */
document.querySelectorAll('main section').forEach(sec=>{if(!sec.querySelector(':scope > .fig'))return;
  const t=document.createElement('div'),f=document.createElement('div');t.className='col-text';f.className='col-fig';
  [...sec.children].forEach(ch=>{(ch.classList.contains('fig')?f:t).appendChild(ch);});sec.append(t,f);sec.classList.add('split');});

/* nav highlight */
(function(){const links=[...document.querySelectorAll('nav.rail a')];
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){links.forEach(l=>l.classList.toggle('on',l.getAttribute('href')==='#'+e.target.id));}});},{rootMargin:'-40% 0px -55% 0px'});
  document.querySelectorAll('main section').forEach(s=>io.observe(s));})();
