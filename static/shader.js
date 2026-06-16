/**
 * shader.js — Subtle animated gradient background via Canvas
 * Keeps WebGL overhead low while still providing a dynamic feel.
 */
(function () {
  const canvas = document.getElementById("bg-canvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  let W, H, raf;

  const orbs = [
    { x: 0.15, y: 0.2,  r: 0.35, color: "rgba(0, 88, 190, 0.06)",  vx: 0.00012, vy: 0.00008 },
    { x: 0.75, y: 0.15, r: 0.30, color: "rgba(33, 112, 228, 0.05)", vx:-0.00010, vy: 0.00012 },
    { x: 0.5,  y: 0.7,  r: 0.40, color: "rgba(0, 108, 73, 0.04)",   vx: 0.00008, vy:-0.00010 },
    { x: 0.85, y: 0.6,  r: 0.25, color: "rgba(146, 71, 0, 0.03)",   vx:-0.00007, vy:-0.00009 },
  ];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }

  function draw(ts) {
    ctx.clearRect(0, 0, W, H);
    orbs.forEach(o => {
      o.x += o.vx;
      o.y += o.vy;
      if (o.x < -0.2 || o.x > 1.2) o.vx *= -1;
      if (o.y < -0.2 || o.y > 1.2) o.vy *= -1;

      const grd = ctx.createRadialGradient(
        o.x * W, o.y * H, 0,
        o.x * W, o.y * H, o.r * Math.max(W, H)
      );
      grd.addColorStop(0, o.color);
      grd.addColorStop(1, "transparent");
      ctx.fillStyle = grd;
      ctx.fillRect(0, 0, W, H);
    });
    raf = requestAnimationFrame(draw);
  }

  window.addEventListener("resize", resize);
  resize();
  raf = requestAnimationFrame(draw);
})();