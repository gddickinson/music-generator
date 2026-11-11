// Subtle animated background: slow drifting staff lines with a few note heads. Static under prefers-reduced-motion.
export function drawBackground(canvas) {
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const notes = Array.from({ length: 28 }, (_, i) => ({ x: (i * 0.137) % 1, line: (i * 7) % 11, s: 0.00004 + (i % 5) * 0.00001 }));
  const draw = t => {
    const w = canvas.width = innerWidth, h = canvas.height = innerHeight;
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(95,212,255,.18)';
    for (let k = 0; k < 5; k++) { const y = h * 0.3 + k * 14; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke(); }
    for (const n of notes) {
      const x = ((n.x + t * n.s) % 1) * w, y = h * 0.3 + n.line * 7 - 7;
      ctx.fillStyle = n.line % 3 ? 'rgba(255,179,71,.35)' : 'rgba(255,95,162,.35)';
      ctx.beginPath(); ctx.ellipse(x, y, 6, 4.5, -0.4, 0, Math.PI * 2); ctx.fill();
    }
  };
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) { draw(0); return; }
  const loop = t => { draw(t); setTimeout(() => requestAnimationFrame(loop), 80); };
  requestAnimationFrame(loop);
}
