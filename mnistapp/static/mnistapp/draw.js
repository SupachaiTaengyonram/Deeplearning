(function(){
function getPos(canvas, evt){
const rect = canvas.getBoundingClientRect();
const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
const clientY = evt.touches ? evt.touches[0].clientY : evt.clientY;
return { x: clientX - rect.left, y: clientY - rect.top };
}

window.initPad = function(canvas){
    const ctx = canvas.getContext('2d');
// White background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0,0,canvas.width,canvas.height);
// Black ink
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = 18;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

let drawing = false;

function start(e){ drawing = true; const p = getPos(canvas,e); ctx.beginPath(); ctx.moveTo(p.x, p.y); e.preventDefault(); }
function move(e){ if(!drawing) return; const p = getPos(canvas,e); ctx.lineTo(p.x,p.y); ctx.stroke(); e.preventDefault(); }
function end(e){ drawing = false; e.preventDefault(); }

canvas.addEventListener('mousedown', start);
canvas.addEventListener('mousemove', move);
canvas.addEventListener('mouseup', end);
canvas.addEventListener('mouseleave', end);

canvas.addEventListener('touchstart', start, {passive:false});
canvas.addEventListener('touchmove', move, {passive:false});
canvas.addEventListener('touchend', end, {passive:false});

// expose for htmx hx-vals
window.mnistPad = canvas;
}

window.clearPad = function(canvas){
const ctx = canvas.getContext('2d');
ctx.fillStyle = '#ffffff';
ctx.fillRect(0,0,canvas.width,canvas.height);
}
})();