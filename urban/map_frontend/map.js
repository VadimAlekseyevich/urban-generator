export default function(component) {
  const { data: d, parentElement: root, setStateValue } = component;
  const canvas = root.querySelector('canvas'), stage = root.querySelector('.stage');
  const ctx = canvas.getContext('2d'), raster = document.createElement('canvas');
  raster.width = d.width; raster.height = d.height;
  const source = raster.getContext('2d');
  const storageKey = 'urban-map-' + d.identity;
  let previous = null;
  try { previous = JSON.parse(sessionStorage.getItem(storageKey)); } catch (_) {}
  let view = previous || { scale: 1, x: 0, y: 0, mode: d.project ? 'project' : 'original', hidden: [], selected: null };
  let width = 0, height = 0, drag = null, ruler = false, points = [];
  const save = () => { try { sessionStorage.setItem(storageKey, JSON.stringify(view)); } catch (_) {} };
  const inMap = p => p.x >= 0 && p.x < d.width && p.y >= 0 && p.y < d.height;
  const point = event => { const r = canvas.getBoundingClientRect(); return { x: (event.clientX-r.left-view.x)/view.scale, y: (event.clientY-r.top-view.y)/view.scale }; };
  const current = () => view.mode === 'project' && d.project ? d.project : d.original;
  const card = () => {
    const o = d.objects[view.selected];
    root.querySelector('.object-title').textContent = o ? o.title : 'Выберите дом или школу на карте';
    root.querySelector('.object-details').textContent = o ? o.details : (d.project ? 'Данные соответствуют завершённому расчёту.' : 'Сначала сгенерируйте проект.');
  };
  function rebuild() {
    const pixels = source.createImageData(d.width, d.height), values = current();
    for (let i=0; i<values.length; i++) {
      let code = values[i];
      if (view.hidden.includes(code)) code = code >= 4 && code <= 6 ? d.original[i] : 255;
      const hex = d.palette[code].color;
      pixels.data.set([parseInt(hex.slice(1,3),16),parseInt(hex.slice(3,5),16),parseInt(hex.slice(5,7),16),255],i*4);
    }
    source.putImageData(pixels,0,0);
  }
  function draw() {
    const ratio = window.devicePixelRatio || 1;
    ctx.setTransform(ratio,0,0,ratio,0,0); ctx.clearRect(0,0,width,height);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(raster,view.x,view.y,d.width*view.scale,d.height*view.scale);
    ctx.strokeStyle='#dbeafe'; ctx.lineWidth=1;
    ctx.strokeRect(view.x,view.y,d.width*view.scale,d.height*view.scale);
    const o = d.objects[view.selected];
    if (o && view.mode === 'project' && !view.hidden.includes(o.kind==='house'?5:6)) {
      ctx.strokeStyle='#facc15'; ctx.lineWidth=3;
      ctx.strokeRect(view.x+o.column*view.scale,view.y+o.row*view.scale,o.side*view.scale,o.side*view.scale);
    }
    if (view.scale >= 7 && view.mode === 'project') {
      ctx.fillStyle='white'; ctx.font='bold 12px system-ui'; ctx.textAlign='left';
      for (const item of Object.values(d.objects)) if (item.kind==='school' && !view.hidden.includes(6)) {
        ctx.strokeStyle='#111827'; ctx.lineWidth=3;
        const x=view.x+(item.column+1)*view.scale+3,y=view.y+(item.row+1)*view.scale;
        ctx.strokeText(String(item.id),Math.min(width-20,x),y); ctx.fillText(String(item.id),Math.min(width-20,x),y);
      }
    }
    const target=120*d.step/view.scale, power=10**Math.floor(Math.log10(target));
    const metres=([5,2,1].find(n=>n*power<=target)||1)*power, length=metres/d.step*view.scale;
    ctx.fillStyle='#0b1220dd'; ctx.fillRect(12,height-54,Math.max(length+24,100),42);
    ctx.strokeStyle='white';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(24,height-24);ctx.lineTo(24+length,height-24);ctx.stroke();
    ctx.fillStyle='white';ctx.font='12px system-ui';ctx.fillText(`${metres.toLocaleString('ru')} м`,24,height-34);
    if (points.length) {
      ctx.strokeStyle='#fde047';ctx.fillStyle='#fde047';ctx.lineWidth=2;ctx.beginPath();
      points.forEach((p,i)=> { const x=view.x+p.x*view.scale,y=view.y+p.y*view.scale; i?ctx.lineTo(x,y):ctx.moveTo(x,y);ctx.fillRect(x-3,y-3,6,6); });ctx.stroke();
      if(points.length===2) {
        const m=Math.hypot(points[1].x-points[0].x,points[1].y-points[0].y)*d.step;
        root.querySelector('.position').textContent=`Линейка: ${m.toFixed(1)} м по прямой. Следующий щелчок — новое измерение.`;
      }
    }
    root.querySelector('.zoom').textContent=`${view.scale.toFixed(1)} px/яч. · ${d.width}×${d.height}`;
    root.querySelector('[data-action=mode]').textContent=view.mode==='project'?'Проект':'Исходник';
    canvas.dataset.scale=String(view.scale);canvas.dataset.offsetX=String(view.x);canvas.dataset.offsetY=String(view.y);
    canvas.dataset.columns=String(d.width);canvas.dataset.rows=String(d.height);
  }
  function fit() { view.scale=Math.max(.1,Math.min((width-32)/d.width,(height-32)/d.height));view.x=(width-d.width*view.scale)/2;view.y=(height-d.height*view.scale)/2;save();draw(); }
  function zoom(factor,x=width/2,y=height/2) {
    const next=Math.max(.1,Math.min(160,view.scale*factor)),ratio=next/view.scale;
    view.x=x-(x-view.x)*ratio;view.y=y-(y-view.y)*ratio;view.scale=next;save();draw();
  }
  function resize() {
    const newWidth=stage.clientWidth,newHeight=stage.clientHeight;
    if(width) {view.x+=(newWidth-width)/2;view.y+=(newHeight-height)/2;}
    width=newWidth;height=newHeight;
    canvas.width=Math.round(width*(window.devicePixelRatio||1));canvas.height=Math.round(height*(window.devicePixelRatio||1));
    if(!previous) {fit();previous=true;} else {save();draw();}
  }
  const layers=root.querySelector('.layers');
  for(const [code,legend] of Object.entries(d.palette)) {
    if (+code===255) continue;
    const label=document.createElement('label'),input=document.createElement('input'),swatch=document.createElement('i');
    input.type='checkbox';input.checked=!view.hidden.includes(+code);input.setAttribute('aria-label',legend.label);swatch.style.background=legend.color;
    label.append(input,swatch,document.createTextNode(legend.label));layers.append(label);
    input.onchange=()=> {view.hidden=input.checked?view.hidden.filter(c=>c!==+code):[...view.hidden,+code];save();rebuild();draw();};
  }
  root.querySelector('[data-action=zoom-in]').onclick=()=>zoom(1.5);
  root.querySelector('[data-action=zoom-out]').onclick=()=>zoom(1/1.5);
  root.querySelector('[data-action=fit]').onclick=fit;
  const mode=root.querySelector('[data-action=mode]');mode.disabled=!d.project;
  mode.onclick=()=> {view.mode=view.mode==='project'?'original':'project';save();rebuild();draw();};
  root.querySelector('[data-action=ruler]').onclick=event=> {ruler=!ruler;points=[];event.target.setAttribute('aria-pressed',String(ruler));root.querySelector('.position').textContent=ruler?'Выберите две точки для измерения расстояния по прямой.':'Щёлкните по дому или школе.';draw();};
  canvas.onwheel=event=> {event.preventDefault();const r=canvas.getBoundingClientRect();zoom(Math.exp(-event.deltaY*.0015),event.clientX-r.left,event.clientY-r.top);};
  canvas.onpointerdown=event=> {canvas.setPointerCapture(event.pointerId);drag={x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY};};
  canvas.onpointermove=event=> {
    if(drag) {view.x+=event.clientX-drag.x;view.y+=event.clientY-drag.y;drag.x=event.clientX;drag.y=event.clientY;draw();return;}
    const p=point(event);if(!inMap(p)||ruler) return;
    const col=Math.floor(p.x),row=Math.floor(p.y),t=d.transform;
    root.querySelector('.position').textContent=`Строка ${row}, столбец ${col} · ${d.palette[current()[row*d.width+col]].label} · X ${(t[0]*(col+.5)+t[2]).toFixed(1)} м, Y ${(t[4]*(row+.5)+t[5]).toFixed(1)} м`;
  };
  canvas.onpointerup=event=> {
    const moved=drag && Math.hypot(event.clientX-drag.startX,event.clientY-drag.startY)>4;drag=null;save();if(moved)return;
    const p=point(event);if(!inMap(p))return;
    if(ruler) {if(points.length===2)points=[];points.push(p);draw();return;}
    const row=Math.floor(p.y),column=Math.floor(p.x),code=current()[row*d.width+column];
    view.selected=view.mode==='project'&&!view.hidden.includes(code)?d.ids[row*d.width+column]:null;
    save();card();draw();setStateValue('selection',view.selected?{row,column}:null);
  };
  canvas.onpointercancel=()=>{drag=null;save();};
  canvas.onkeydown=event=> {if(event.key==='+'||event.key==='=')zoom(1.5);if(event.key==='-')zoom(1/1.5);if(event.key==='0')fit();};
  rebuild();card();const observer=new ResizeObserver(resize);observer.observe(stage);
  return ()=> {observer.disconnect();save();};
}
