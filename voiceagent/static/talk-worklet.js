// Microphone capture: resamples the input to 16 kHz PCM16 and posts about 40 ms (640 samples)
// at a time to the page, which sends each chunk to the server as one binary WebSocket frame.
// Linear interpolation is enough for speech going to a speech recognizer.
class TalkCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.target = 16000;
    this.ratio = sampleRate / this.target;
    this.pos = 0;            // read position in input samples, carried between blocks
    this.prev = 0;           // last sample of the previous block, for interpolation across the edge
    this.out = new Int16Array(640);
    this.n = 0;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    const len = ch.length;
    while (this.pos < len) {
      const i = Math.floor(this.pos);
      const frac = this.pos - i;
      const a = i - 1 >= 0 ? ch[i - 1] : this.prev;
      const b = ch[i];
      const sample = a + (b - a) * frac;
      const s = Math.max(-1, Math.min(1, sample));
      this.out[this.n++] = s < 0 ? s * 32768 : s * 32767;
      if (this.n === this.out.length) {
        this.port.postMessage(this.out.buffer.slice(0));
        this.n = 0;
      }
      this.pos += this.ratio;
    }
    this.pos -= len;
    this.prev = ch[len - 1];
    return true;
  }
}

registerProcessor("talk-capture", TalkCapture);
