/* ============================================================
   NEON GRID — vibeSpatial Docs Interactive Effects
   Shared across vibeProj, vibeSpatial, vibeSpatial-Raster.
   ============================================================ */
(function () {
  "use strict";

  /* ===========================
     PARTICLE CONSTELLATION
     Floating dots with proximity lines — hero section background
     =========================== */
  class ParticleField {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.particles = [];
      this.mouse = { x: -1000, y: -1000 };
      this.running = true;

      /* Config */
      this.PARTICLE_COUNT = 60;
      this.CONNECT_DIST = 120;
      this.MOUSE_DIST = 160;
      this.SPEED = 0.3;

      this._resize = this.resize.bind(this);
      this._mouseMove = this.onMouseMove.bind(this);
      this._mouseLeave = this.onMouseLeave.bind(this);

      window.addEventListener("resize", this._resize);
      this.canvas.parentElement.addEventListener("mousemove", this._mouseMove);
      this.canvas.parentElement.addEventListener("mouseleave", this._mouseLeave);

      this.resize();
      this.init();
      this.animate();
    }

    resize() {
      const rect = this.canvas.parentElement.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      this.w = rect.width;
      this.h = rect.height;
      this.canvas.width = this.w * dpr;
      this.canvas.height = this.h * dpr;
      this.canvas.style.width = this.w + "px";
      this.canvas.style.height = this.h + "px";
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    init() {
      this.particles = [];
      for (let i = 0; i < this.PARTICLE_COUNT; i++) {
        this.particles.push({
          x: Math.random() * this.w,
          y: Math.random() * this.h,
          vx: (Math.random() - 0.5) * this.SPEED,
          vy: (Math.random() - 0.5) * this.SPEED,
          r: Math.random() * 1.5 + 0.5,
          alpha: Math.random() * 0.4 + 0.2,
        });
      }
    }

    onMouseMove(e) {
      const rect = this.canvas.parentElement.getBoundingClientRect();
      this.mouse.x = e.clientX - rect.left;
      this.mouse.y = e.clientY - rect.top;
    }

    onMouseLeave() {
      this.mouse.x = -1000;
      this.mouse.y = -1000;
    }

    animate() {
      if (!this.running) return;
      requestAnimationFrame(() => this.animate());

      const { ctx, w, h, particles, mouse } = this;
      ctx.clearRect(0, 0, w, h);

      /* Update positions */
      for (const p of particles) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;
        p.x = Math.max(0, Math.min(w, p.x));
        p.y = Math.max(0, Math.min(h, p.y));
      }

      /* Draw connections */
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < this.CONNECT_DIST) {
            const alpha = (1 - dist / this.CONNECT_DIST) * 0.15;
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            ctx.strokeStyle = `rgba(255, 45, 111, ${alpha})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        }

        /* Mouse proximity glow */
        const mdx = particles[i].x - mouse.x;
        const mdy = particles[i].y - mouse.y;
        const mDist = Math.sqrt(mdx * mdx + mdy * mdy);
        if (mDist < this.MOUSE_DIST) {
          const alpha = (1 - mDist / this.MOUSE_DIST) * 0.5;
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(mouse.x, mouse.y);
          ctx.strokeStyle = `rgba(168, 85, 247, ${alpha})`;
          ctx.lineWidth = 0.8;
          ctx.stroke();
        }
      }

      /* Draw particles */
      for (const p of particles) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 45, 111, ${p.alpha})`;
        ctx.fill();
      }
    }

    destroy() {
      this.running = false;
      window.removeEventListener("resize", this._resize);
      this.canvas.parentElement.removeEventListener("mousemove", this._mouseMove);
      this.canvas.parentElement.removeEventListener("mouseleave", this._mouseLeave);
    }
  }

  /* ===========================
     GLITCH EFFECT
     Periodic glitch on hero title
     =========================== */
  class GlitchController {
    constructor(el) {
      this.el = el;
      this.running = true;
      this.schedule();
    }

    trigger() {
      this.el.classList.add("glitch-active");
      setTimeout(() => {
        this.el.classList.remove("glitch-active");
      }, 300);
    }

    schedule() {
      if (!this.running) return;
      /* Random interval between 3-8 seconds */
      const delay = 3000 + Math.random() * 5000;
      setTimeout(() => {
        if (!this.running) return;
        this.trigger();
        this.schedule();
      }, delay);
    }

    destroy() {
      this.running = false;
    }
  }

  /* ===========================
     GLOW ON SCROLL
     Reveal elements as they enter viewport
     =========================== */
  function initScrollReveal() {
    const els = document.querySelectorAll(".cp-reveal");
    if (!els.length) return;

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            observer.unobserve(entry.target);
          }
        }
      },
      { threshold: 0.15, rootMargin: "0px 0px -40px 0px" }
    );

    els.forEach((el) => observer.observe(el));
  }

  /* ===========================
     AMBIENT CURSOR GLOW
     Subtle radial glow follows mouse on hero
     =========================== */
  function initCursorGlow(heroEl) {
    if (!heroEl) return;

    const glow = document.createElement("div");
    glow.style.cssText = `
      position: absolute;
      width: 300px;
      height: 300px;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(0,229,255,0.06) 0%, transparent 70%);
      pointer-events: none;
      z-index: 1;
      transform: translate(-50%, -50%);
      transition: opacity 0.3s ease;
      opacity: 0;
    `;
    heroEl.appendChild(glow);

    heroEl.addEventListener("mousemove", (e) => {
      const rect = heroEl.getBoundingClientRect();
      glow.style.left = e.clientX - rect.left + "px";
      glow.style.top = e.clientY - rect.top + "px";
      glow.style.opacity = "1";
    });

    heroEl.addEventListener("mouseleave", () => {
      glow.style.opacity = "0";
    });
  }

  /* ===========================
     CODE BLOCK LANGUAGE LABELS
     Adds language badge to terminal header
     =========================== */
  function initCodeLabels() {
    document.querySelectorAll("div.highlight").forEach((block) => {
      /* Try to detect language from class */
      const pre = block.querySelector("pre");
      if (!pre) return;
      const classes = [...block.classList, ...(pre.classList || [])];
      let lang = "";
      for (const cls of classes) {
        const m = cls.match(/^highlight-(\w+)$/);
        if (m) { lang = m[1]; break; }
      }
      if (!lang) {
        /* Fallback: check parent wrapper */
        const wrapper = block.closest("[class*='highlight-']");
        if (wrapper) {
          const wm = [...wrapper.classList].find(c => c.startsWith("highlight-"));
          if (wm) lang = wm.replace("highlight-", "");
        }
      }
      if (lang) {
        const label = document.createElement("span");
        label.textContent = lang.toUpperCase();
        label.style.cssText = `
          position: absolute;
          top: 7px;
          right: 12px;
          font-family: 'Orbitron', monospace;
          font-size: 0.55rem;
          font-weight: 600;
          letter-spacing: 0.12em;
          color: rgba(0,229,255,0.45);
          z-index: 1;
          pointer-events: none;
        `;
        block.style.position = "relative";
        block.appendChild(label);
      }
    });
  }

  /* ===========================
     SIDEBAR ACTIVE PULSE
     Subtle glow animation on current page link
     =========================== */
  function initSidebarPulse() {
    const current = document.querySelector(".sidebar-tree .current-page > .reference");
    if (!current) return;
    current.style.animation = "sidebar-pulse 3s ease-in-out infinite";

    if (!document.getElementById("cp-sidebar-pulse-style")) {
      const style = document.createElement("style");
      style.id = "cp-sidebar-pulse-style";
      style.textContent = `
        @keyframes sidebar-pulse {
          0%, 100% { text-shadow: none; }
          50% { text-shadow: 0 0 8px rgba(0,229,255,0.3); }
        }
      `;
      document.head.appendChild(style);
    }
  }

  /* ===========================
     INITIALIZATION
     =========================== */
  function init() {
    /* Respect reduced motion */
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      /* Still init code labels and scroll reveal (no animation) */
      document.querySelectorAll(".cp-reveal").forEach((el) => el.classList.add("visible"));
      initCodeLabels();
      return;
    }

    /* Hero section */
    const hero = document.querySelector(".cp-hero");
    if (hero) {
      /* Particle canvas */
      const canvas = document.createElement("canvas");
      canvas.className = "cp-particles";
      canvas.style.cssText =
        "position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:1;";
      hero.insertBefore(canvas, hero.firstChild);
      new ParticleField(canvas);

      /* Glitch text */
      const title = hero.querySelector(".cp-hero-title[data-glitch]");
      if (title) new GlitchController(title);

      /* Cursor glow */
      initCursorGlow(hero);
    }

    /* Scroll reveal */
    initScrollReveal();

    /* Code labels */
    initCodeLabels();

    /* Sidebar pulse */
    initSidebarPulse();
  }

  /* Run on DOMContentLoaded or immediately if already loaded */
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
