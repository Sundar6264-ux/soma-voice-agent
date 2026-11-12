// Soma's AI browser client: microphone in (16 kHz PCM16 over the WebSocket), 24 kHz PCM16 frames out,
// captions, links panel, countdown. No inline script so a strict CSP can apply later.
(function () {
  const OUT_RATE = 24000, IN_RATE = 16000, JITTER_S = 0.12, LIMIT_S = 600;
  const $ = id => document.getElementById(id);
  const captions = $("captions"), stateEl = $("state"), startBtn = $("start"), clock = $("clock");
  const linksEl = $("links"), statsEl = $("stats"), typedForm = $("typed"), typedText = $("typed-text");

  let ws = null, ctx = null, mic = null, worklet = null, source = null;
  let current = null, started = 0, ticker = null;
  let nextAt = 0, playing = [], framesIn = 0, framesPlayed = 0;
  window.talkStats = { framesIn: 0, framesPlayed: 0, cleared: 0 };

  function line(cls, who, text) {
    const d = document.createElement("div");
    d.className = "msg " + cls;
    if (who) {
      const w = document.createElement("div"); w.className = "who"; w.textContent = who;
      const t = document.createElement("div"); t.className = "text"; t.textContent = text;
      d.append(w, t);
      captions.appendChild(d); captions.scrollTop = captions.scrollHeight;
      return t;
    }
    d.className = "msg sys"; d.textContent = text;
    captions.appendChild(d); captions.scrollTop = captions.scrollHeight;
    return d;
  }

  function setState(text) { stateEl.textContent = text; }

  function fmt(s) { s = Math.max(0, Math.round(s)); return Math.floor(s / 60) + ":" + String(s % 60).padStart(2, "0"); }

  function tick() {
    const left = LIMIT_S - (Date.now() - started) / 1000;
    clock.textContent = fmt(left);
    clock.classList.toggle("low", left <= 60);
  }

  // -- playback: schedule each 20 ms frame back to back, a small jitter buffer ahead of now --
  function playFrame(buf) {
    if (!ctx) return;
    const pcm = new Int16Array(buf);
    let silent = true;
    for (let i = 0; i < pcm.length; i += 16) { if (pcm[i] !== 0) { silent = false; break; } }
    framesIn++; window.talkStats.framesIn = framesIn;
    if (silent) return;  // idle silence frames keep the pacing clock, nothing to hear
    const ab = ctx.createBuffer(1, pcm.length, OUT_RATE);
    const data = ab.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) data[i] = pcm[i] / 32768;
    const src = ctx.createBufferSource();
    src.buffer = ab; src.connect(ctx.destination);
    const now = ctx.currentTime;
    if (nextAt < now + 0.005) nextAt = now + JITTER_S;
    src.start(nextAt);
    nextAt += ab.duration;
    playing.push(src);
    src.onended = () => { const i = playing.indexOf(src); if (i >= 0) playing.splice(i, 1); };
    framesPlayed++; window.talkStats.framesPlayed = framesPlayed;
    statsEl.textContent = "audio frames: " + framesPlayed;
  }

  function clearAudio() {
    for (const s of playing) { try { s.stop(); } catch (e) {} }
    playing = []; nextAt = 0;
    window.talkStats.cleared++;
  }

  function showLinks(links) {
    linksEl.querySelectorAll("a").forEach(a => a.remove());
    for (const l of links) {
      const a = document.createElement("a");
      a.href = l.url; a.target = "_blank"; a.rel = "noopener"; a.textContent = l.label + ": " + l.url;
      linksEl.appendChild(a);
    }
    linksEl.classList.add("show");
  }

  function onMessage(ev) {
    if (ev.data instanceof ArrayBuffer) { playFrame(ev.data); return; }
    let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
    if (m.type === "agent_text") {
      if (!current) current = line("agent", "Soma's AI", "");
      current.textContent += m.token;
      captions.scrollTop = captions.scrollHeight;
      if (m.last) current = null;
    } else if (m.type === "caller_text") {
      current = null;
      line("caller", "You", m.text);
    } else if (m.type === "clear") {
      clearAudio();
      current = null;
    } else if (m.type === "links") {
      showLinks(m.links || []);
    } else if (m.type === "time") {
      line("sys", "", "About " + fmt(m.seconds_left) + " left.");
    } else if (m.type === "busy") {
      line("sys", "", m.text);
      setState("Busy. Try again shortly.");
    } else if (m.type === "end") {
      line("sys", "", m.reason === "time_limit" ? "Time is up. Thanks for talking." : "Conversation ended.");
      setTimeout(stop, 1500);
    }
  }

  async function start() {
    startBtn.disabled = true;
    setState("Connecting...");
    try {
      ctx = new (window.AudioContext || window.webkitAudioContext)();
      await ctx.resume();
      mic = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 }, video: false });
      await ctx.audioWorklet.addModule("/static/talk-worklet.js");
      worklet = new AudioWorkletNode(ctx, "talk-capture", { numberOfInputs: 1, numberOfOutputs: 0 });
      source = ctx.createMediaStreamSource(mic);
      source.connect(worklet);
    } catch (e) {
      setState("Microphone not available: " + (e.message || e));
      startBtn.disabled = false;
      await teardownAudio();
      return;
    }
    ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/talk");
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      started = Date.now(); tick(); ticker = setInterval(tick, 1000);
      setState("Listening. Speak normally; you can interrupt.");
      startBtn.textContent = "Stop"; startBtn.classList.add("live"); startBtn.disabled = false;
      typedForm.hidden = false;
      worklet.port.onmessage = e => { if (ws && ws.readyState === 1) ws.send(e.data); };
    };
    ws.onmessage = onMessage;
    ws.onclose = e => {
      if (e.code === 4401) setState("Log in first (open /login?key=... then come back).");
      else if (e.code === 4429) setState("Soma's AI is busy with someone else. Try again shortly.");
      else if (started) setState("Conversation ended.");
      stop(true);
    };
    ws.onerror = () => setState("Connection problem.");
  }

  async function teardownAudio() {
    try { worklet && worklet.port && (worklet.port.onmessage = null); } catch (e) {}
    try { source && source.disconnect(); } catch (e) {}
    try { worklet && worklet.disconnect(); } catch (e) {}
    if (mic) mic.getTracks().forEach(t => t.stop());
    clearAudio();
    if (ctx) { try { await ctx.close(); } catch (e) {} }
    ctx = null; mic = null; worklet = null; source = null;
  }

  function stop(fromClose) {
    if (ticker) { clearInterval(ticker); ticker = null; }
    if (ws && !fromClose && ws.readyState === 1) { try { ws.send(JSON.stringify({ type: "hangup" })); } catch (e) {} }
    const sock = ws; ws = null;
    if (sock && sock.readyState < 2) sock.close();
    teardownAudio();
    startBtn.textContent = "Start"; startBtn.classList.remove("live"); startBtn.disabled = false;
    typedForm.hidden = true;
    current = null;
  }

  startBtn.onclick = () => { if (ws) stop(); else start(); };
  typedForm.onsubmit = e => {
    e.preventDefault();
    const text = typedText.value.trim();
    if (text && ws && ws.readyState === 1) { ws.send(JSON.stringify({ type: "text", text })); typedText.value = ""; }
  };
})();
