// End-to-end check of Soma's AI in a real Chromium: a WAV plays as the microphone, the page must
// caption the question, show a grounded answer and play audio frames.
//
//   NODE_PATH=/home/sundar/flockara-ui-tests/node_modules node scripts/talk_e2e.mjs [wav] [seconds]
//
// Reads ADMIN_KEY from .env (never printed). Costs a few cents: real AssemblyAI, Claude and Inworld.
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const wav = path.resolve(process.argv[2] || path.join(root, "logs/e2e/question.wav"));
const waitSeconds = Number(process.argv[3] || 45);
const env = Object.fromEntries(fs.readFileSync(path.join(root, ".env"), "utf8").split("\n")
  .filter(l => l.includes("=") && !l.startsWith("#")).map(l => { const i = l.indexOf("="); return [l.slice(0, i).trim(), l.slice(i + 1).trim()]; }));
const base = process.env.TALK_BASE || "http://127.0.0.1:8790";

const browser = await chromium.launch({
  args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream",
         `--use-file-for-fake-audio-capture=${wav}%noloop`, "--autoplay-policy=no-user-gesture-required"],
});
const page = await browser.newPage({ viewport: { width: 390, height: 780 } });
page.on("console", m => { if (m.type() === "error") console.log("console error:", m.text()); });
await page.goto(`${base}/login?key=${encodeURIComponent(env.ADMIN_KEY)}&next=/talk`);
if (!page.url().endsWith("/talk")) throw new Error("login did not land on /talk: " + page.url());
await page.click("#start");
await page.waitForSelector("#captions .agent", { timeout: 15000 });
console.log("greeting captioned at", new Date().toISOString());

const deadline = Date.now() + waitSeconds * 1000;
let callerSeen = false, answer = "";
while (Date.now() < deadline) {
  const caller = await page.$$eval("#captions .caller .text", els => els.map(e => e.textContent));
  const agents = await page.$$eval("#captions .agent .text", els => els.map(e => e.textContent));
  if (!callerSeen && caller.length) { callerSeen = true; console.log("caller caption:", JSON.stringify(caller[0])); }
  if (callerSeen && agents.length >= 2 && agents[agents.length - 1].length > 20) {
    answer = agents[agents.length - 1];
    await page.waitForTimeout(12000);  // let the answer finish streaming and playing
    answer = (await page.$$eval("#captions .agent .text", els => els.map(e => e.textContent))).slice(1).join(" ");
    break;
  }
  await page.waitForTimeout(500);
}
const stats = await page.evaluate(() => window.talkStats);
const clock = await page.textContent("#clock");
await page.screenshot({ path: path.join(root, "logs/e2e/talk.png") });
console.log("answer:", JSON.stringify(answer));
console.log("stats:", JSON.stringify(stats), "clock:", clock);
await page.click("#start");  // Stop
await page.waitForTimeout(800);
await browser.close();

const ok = callerSeen && answer.length > 20 && stats.framesPlayed > 0;
console.log(ok ? "E2E OK" : "E2E FAILED");
process.exit(ok ? 0 : 1);
