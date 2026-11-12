import { readFileSync } from "node:fs";
import { expect, it, vi } from "vitest";
const html = readFileSync("../spikes/listening/index.html", "utf8");
const script = readFileSync("../spikes/listening/app.js", "utf8");

it("starts comparison buffers on one clock, switches without restarting, and confirms before unblinding", async () => {
  document.body.innerHTML = new DOMParser().parseFromString(html, "text/html").body.innerHTML;
  const session = {
    manifest: { cases: [{ id: "c01", title: "Fixture", artist: "Test", duration: 8, start: 10, focus: 3.5,
      words: [], has_reference: false }] },
    state: { revision: 1, unblinded: false, answers: { c01: { finished: true, preference: "A", initial_preference: "A" } } },
  };
  const starts = [];
  class AudioMock {
    currentTime = 1;
    destination = {};
    async resume() {}
    async decodeAudioData() { return { duration: 8 }; }
    createBufferSource() {
      return { connect() {}, disconnect() {}, stop() {}, start(when) { starts.push(when); } };
    }
    createGain() { return { gain: { value: 0, setTargetAtTime() {} }, connect() {}, disconnect() {} }; }
  }
  const fetchMock = vi.fn(async (path) => ({ ok: true,
    async json() {
      if (path === "/api/session") return structuredClone(session);
      if (path === "/api/unblind") return { state: { ...session.state, unblinded: true },
        key: { cases: { c01: { group: "control", mapping: { A: "final", B: "before" } } } } };
      throw new Error(`Unexpected request ${path}`);
    },
    async arrayBuffer() { return new ArrayBuffer(0); },
  }));
  const dialog = document.getElementById("confirmation");
  dialog.showModal = () => { dialog.open = true; };
  HTMLElement.prototype.scrollIntoView = vi.fn();
  const AsyncFunction = Object.getPrototypeOf(async () => {}).constructor;
  await new AsyncFunction("document", "window", "fetch", "AudioContext", "requestAnimationFrame", script)(
    document, window, fetchMock, AudioMock, () => 0,
  );
  document.getElementById("play").click();
  await vi.waitFor(() => expect(starts).toHaveLength(4));
  expect(new Set(starts).size).toBe(1);
  (document.querySelector('[data-guide="A"]')).click();
  (document.querySelector('[data-guide="B"]')).click();
  expect(starts).toHaveLength(4);
  document.getElementById("unblind").click();
  expect(dialog.open).toBe(true);
  expect(fetchMock.mock.calls.some(([path]) => path === "/api/unblind")).toBe(false);
  dialog.returnValue = "cancel";
  dialog.dispatchEvent(new Event("close"));
  dialog.open = false;
  await Promise.resolve();
  expect(fetchMock.mock.calls.some(([path]) => path === "/api/unblind")).toBe(false);
  document.getElementById("unblind").click();
  dialog.returnValue = "continue";
  dialog.dispatchEvent(new Event("close"));
  await vi.waitFor(() => expect(document.getElementById("results").hidden).toBe(false));
  expect(document.getElementById("resultTable").textContent).toContain("A = New decoder");
});
