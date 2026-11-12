const $ = (id) => document.getElementById(id);
let manifest, state, assignmentKey, current = 0, dirty = false, loading = true, busy = false;
let context, buffers = {}, nodes = [], gains = {}, playing = false, offset = 0, startedAt = 0;
let source = "recording", guide = "off", loadEpoch = 0, playbackEpoch = 0, playingLoop = true;
const labels = {before:"Baseline", final:"New decoder", equal_good:"Both sound right", equal_bad:"Both sound wrong", unsure:"Unclear"};

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body),
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try { const data = await response.json(); message = typeof data.detail === "string" ? data.detail : message; } catch {}
    throw new Error(message);
  }
  return response.json();
}
function error(e) { $("error").textContent = e.message || String(e); $("error").hidden = false; }
function clearError() { $("error").hidden = true; }
function confirmAction(message, label) {
  const dialog=$("confirmation");
  if(dialog.open) return Promise.resolve(false);
  $("confirmationMessage").textContent=message; $("confirmationAccept").textContent=label;
  dialog.returnValue="cancel"; dialog.showModal();
  return new Promise(resolve=>dialog.addEventListener("close",()=>resolve(dialog.returnValue==="continue"),{once:true}));
}
function stamp(s) { return `${Math.floor(s/60)}:${Math.floor(s%60).toString().padStart(2,"0")}`; }
function item() { return manifest.cases[current]; }
function answer() { return state.answers[item().id]; }
function position() {
  if (!playing) return offset;
  const p = Math.max(0,context.currentTime-startedAt);
  return playingLoop ? p % item().duration : Math.min(p, item().duration);
}
function pause() {
  if (playing) offset = position();
  playing = false; playbackEpoch++;
  for (const node of nodes) { try { node.stop(); } catch {} node.disconnect(); }
  for (const gain of Object.values(gains)) gain.disconnect();
  nodes = []; gains = {};
  $("play").textContent = loading ? "Loading audio…" : "Play";
}
async function play() {
  if (loading || playing) return;
  const epoch = ++playbackEpoch;
  await context.resume();
  if (epoch !== playbackEpoch) return;
  if (offset >= item().duration-.01) offset = 0;
  const when = context.currentTime+.025;
  startedAt = when-offset;
  playingLoop = $("loop").checked;
  for (const [name, buffer] of Object.entries(buffers)) {
    const node = context.createBufferSource(), gain = context.createGain();
    node.buffer = buffer; node.loop = playingLoop;
    gain.gain.value = 0;
    node.connect(gain); gain.connect(context.destination);
    node.start(when, offset);
    nodes.push(node); gains[name] = gain;
  }
  playing = true; $("play").textContent = "Pause"; mix();
}
function mix() {
  for (const button of document.querySelectorAll("[data-source]")) button.setAttribute("aria-pressed", String(button.dataset.source === source));
  for (const button of document.querySelectorAll("[data-guide]")) button.setAttribute("aria-pressed", String(button.dataset.guide === guide));
  $("solo").disabled = guide === "off";
  for (const [name, gain] of Object.entries(gains)) {
    let level = 0;
    if (name === source && (guide === "off" || !$("solo").checked)) level = +$("sourceVolume").value;
    if (name === guide) level = +$("guideVolume").value;
    gain.gain.setTargetAtTime(level, context.currentTime, .012);
  }
}
async function audioBuffer(label, caseId) {
  const response = await fetch(`/audio/${caseId}/${label}`);
  if (!response.ok) throw new Error(`Could not load ${label} audio (${response.status}). Reload to retry.`);
  return context.decodeAudioData(await response.arrayBuffer());
}
function draw() {
  if (manifest) {
    const p = Math.max(0, position()), duration = item().duration;
    $("seek").value = String(p);
    $("clock").textContent = `${stamp(p)} / ${stamp(duration)}`;
    const c = $("wave"), g = c.getContext("2d"), w = c.width, h = c.height;
    g.clearRect(0,0,w,h);
    const left = Math.max(0,item().focus-.65)/duration*w, right = Math.min(duration,item().focus+.65)/duration*w;
    g.fillStyle = "#a8c68920"; g.fillRect(left,0,right-left,h);
    const buffer = buffers.recording;
    if (buffer) {
      const y = buffer.getChannelData(0), step = Math.max(1,Math.floor(y.length/400));
      g.strokeStyle = "#a0baa0"; g.lineWidth = 1.5; g.beginPath();
      for (let i=0;i<400;i++) {
        let peak = 0;
        for (let j=i*step;j<Math.min(y.length,(i+1)*step);j+=3) peak = Math.max(peak,Math.abs(y[j]));
        const x=i/400*w, a=Math.max(1,Math.min(.48,peak)*h);
        g.moveTo(x,h/2-a); g.lineTo(x,h/2+a);
      }
      g.stroke();
    }
    g.strokeStyle="#e2edc9"; g.lineWidth=2; g.beginPath(); g.moveTo(p/duration*w,0); g.lineTo(p/duration*w,h); g.stroke();
    if (playing && !$("loop").checked && p >= duration) { pause(); offset=0; }
  }
  requestAnimationFrame(draw);
}
function progress() {
  const completed = manifest.cases.filter(c => state.answers[c.id]?.finished).length;
  $("progress").textContent = `${completed} / ${manifest.cases.length}`;
  $("progressBar").max = manifest.cases.length; $("progressBar").value = completed;
  $("unblind").disabled = completed !== manifest.cases.length || busy;
  $("unblind").textContent = state.unblinded ? "Show revealed results" : "Reveal versions & results";
  $("caseList").replaceChildren();
  manifest.cases.forEach((c,i) => {
    const button = document.createElement("button"), a = state.answers[c.id];
    button.textContent = `${String(i+1).padStart(2,"0")}${a?.finished ? " ✓" : a ? " ◐" : ""}`;
    button.setAttribute("aria-label",`Excerpt ${i+1}${a?.finished ? ", finished" : ""}`);
    button.setAttribute("aria-current",String(i===current)); button.classList.toggle("done",!!a?.finished);
    button.disabled=busy; button.addEventListener("click",() => changeCase(i)); $("caseList").append(button);
  });
  $("previous").disabled = current===0 || busy; $("next").disabled = current===manifest.cases.length-1 || busy;
}
function form() {
  const a = answer();
  for (const radio of document.querySelectorAll("[name=preference]")) radio.checked=radio.value===a?.preference;
  $("confidence").value=a?.confidence || "medium";
  for (const box of document.querySelectorAll("[name=issue]")) box.checked=(a?.issues || []).includes(box.value);
  $("note").value=a?.note || ""; $("referenceVerdict").value=a?.reference || "not_assessed";
  $("referenceStep").hidden=!a; $("referenceControls").hidden=true;
  $("referenceUnavailable").hidden=item().has_reference;
  $("loadReference").hidden=!item().has_reference;
  $("loadReference").disabled=false; $("loadReference").textContent="Listen to the reference";
  $("choiceSaved").textContent=a ? (a.reference_exposed ? "First choice preserved before the reference was opened." : "Saved before hearing the reference.") : "";
  $("finish").textContent=current===manifest.cases.length-1 ? "Finish this excerpt" : "Save & next excerpt";
}
function markDirty() { dirty=true; $("saveStatus").textContent="Unsaved choice"; $("loadReference").disabled=true; }
async function changeCase(i, force=false) {
  if (busy || (!force && dirty && !await confirmAction("Leave this excerpt without saving your latest changes?", "Leave excerpt"))) return;
  const moving=!!buffers.recording;
  pause(); const epoch=++loadEpoch;
  current=i; offset=0; buffers={}; loading=true; source="recording"; guide="off"; dirty=false;
  clearError(); form(); progress(); mix();
  $("caseNumber").textContent=`EXCERPT ${String(i+1).padStart(2,"0")} / ${manifest.cases.length}`;
  $("title").textContent=item().title;
  $("subtitle").textContent=`${item().artist} · ${stamp(item().start)}–${stamp(item().start+item().duration)} in the song`;
  if(moving) $("title").scrollIntoView({block:"start"});
  $("words").textContent=item().words.map(w=>w.text).join(" ");
  $("seek").max=item().duration; $("play").disabled=true; $("play").textContent="Loading audio…";
  $("saveStatus").textContent=state.unblinded ? "Labels revealed · further edits are recorded" : "Saved locally · labels hidden";
  try {
    context ||= new AudioContext();
    const entries=await Promise.all(["recording","vocals","A","B"].map(async label=>[label,await audioBuffer(label,item().id)]));
    if (epoch!==loadEpoch) return;
    buffers=Object.fromEntries(entries); loading=false; $("play").disabled=false; $("play").textContent="Play";
  } catch(e) { if(epoch===loadEpoch) error(e); }
}
function verdict(finished) {
  const preference=document.querySelector("[name=preference]:checked")?.value;
  if (!preference) throw new Error("Choose A, B, both, or ‘I can’t tell’ before saving.");
  return {revision:state.revision, preference, confidence:$("confidence").value,
    issues:[...document.querySelectorAll("[name=issue]:checked")].map(x=>x.value), note:$("note").value,
    reference:$("referenceVerdict").value, finished};
}
async function save(finished=false) {
  if (busy) return;
  clearError();
  try {
    const v=verdict(finished); busy=true; progress(); $("saveChoice").disabled=true; $("finish").disabled=true;
    state=await api(`/api/answers/${item().id}`,v); dirty=false;
    $("saveStatus").textContent="Saved on this computer";
    $("referenceStep").hidden=false; $("loadReference").disabled=false;
    $("choiceSaved").textContent="Choice saved. You can check the reference or finish this excerpt.";
    busy=false;
    if(finished) {
      const next=manifest.cases.findIndex((c,i)=>i>current && !state.answers[c.id]?.finished);
      if(next>=0) await changeCase(next,true);
      else {
        const earlier=manifest.cases.findIndex(c=>!state.answers[c.id]?.finished);
        if(earlier>=0) await changeCase(earlier,true);
        else { pause(); $("saveStatus").textContent="All excerpts saved. You can reveal the versions now."; }
      }
    }
  } catch(e) { error(e); }
  finally { busy=false; $("saveChoice").disabled=false; $("finish").disabled=false; progress(); }
}
async function loadReference() {
  if(busy || dirty) return;
  clearError(); busy=true; progress(); $("loadReference").disabled=true;
  try {
    const result=await api(`/api/reference/${item().id}`,{}); state=result.state;
    const names=result.has_revised ? ["R","R2"] : ["R"];
    const entries=await Promise.all(names.map(async name=>[name,await audioBuffer(name,item().id)]));
    pause(); Object.assign(buffers,Object.fromEntries(entries));
    $("referenceControls").hidden=false; $("revised").hidden=!result.has_revised;
    $("loadReference").hidden=true; guide="R"; mix();
    $("choiceSaved").textContent="First choice preserved. Changes after opening the reference are recorded separately.";
    await play();
  } catch(e) { error(e); $("loadReference").disabled=false; }
  finally { busy=false; progress(); }
}
function results() {
  const groups=new Map();
  for(const c of manifest.cases) {
    const a=state.answers[c.id], k=assignmentKey.cases[c.id];
    const row=groups.get(k.group) || {before:0,final:0,equal_good:0,equal_bad:0,unsure:0};
    const first=a.initial_preference;
    row[k.mapping[first] || first]++; groups.set(k.group,row);
  }
  const table=document.createElement("table"), header=document.createElement("tr");
  for(const title of ["Selection group","Baseline","New","Both right","Both wrong","Unclear"]) {
    const cell=document.createElement("th"); cell.textContent=title; header.append(cell);
  }
  table.append(header);
  for(const [group,counts] of groups) {
    const row=document.createElement("tr");
    for(const value of [group.replaceAll("_"," "),...Object.values(counts)]) {const td=document.createElement("td");td.textContent=value;row.append(td);}
    table.append(row);
  }
  $("resultTable").replaceChildren(table);
  const p=document.createElement("p");p.className="small";
  p.textContent="Counts use your first saved A/B choices. The downloaded file includes assignments, reference assessments, changes and sampling provenance.";
  $("resultTable").append(p);
  const list=document.createElement("p");list.className="small";
  list.textContent=manifest.cases.map((c,i)=>`${i+1}: A = ${labels[assignmentKey.cases[c.id].mapping.A]}`).join(" · ");
  $("resultTable").append(list); $("results").hidden=false;
}
$("play").addEventListener("click",()=>playing ? pause() : play().catch(error));
$("restart").addEventListener("click",()=>{const was=playing;pause();offset=0;if(was)play().catch(error);});
$("seek").addEventListener("input",()=>{const next=+$("seek").value,was=playing;pause();offset=next;if(was)play().catch(error);});
$("loop").addEventListener("change",()=>{const was=playing;pause();if(was)play().catch(error);});
for(const button of document.querySelectorAll("[data-source]")) button.addEventListener("click",()=>{source=button.dataset.source;mix();});
for(const button of document.querySelectorAll("[data-guide]")) button.addEventListener("click",()=>{guide=button.dataset.guide;mix();});
for(const id of ["solo","sourceVolume","guideVolume"]) $(id).addEventListener("input",mix);
$("verdictForm").addEventListener("input",markDirty);
$("verdictForm").addEventListener("submit",event=>{event.preventDefault();save(true);});
$("saveChoice").addEventListener("click",()=>save(false));
$("loadReference").addEventListener("click",loadReference);
$("referencePlay").addEventListener("click",()=>playing ? pause() : play().catch(error));
$("previous").addEventListener("click",()=>changeCase(current-1));
$("next").addEventListener("click",()=>changeCase(current+1));
$("unblind").addEventListener("click",async()=>{
  if(dirty) {error(new Error("Save your latest changes before revealing results."));return;}
  if(!state.unblinded && !await confirmAction("Reveal which version is which? Future edits will no longer be blinded. Your initial answers will be preserved.", "Reveal versions")) return;
  try {const result=await api("/api/unblind",{});state=result.state;assignmentKey=result.key;results();progress();$("saveStatus").textContent="Labels revealed · first choices preserved";$("results").scrollIntoView({behavior:"smooth"});}
  catch(e){error(e);}
});
window.addEventListener("beforeunload",event=>{if(dirty){event.preventDefault();event.returnValue="";}});
document.addEventListener("keydown",event=>{
  if(event.target.closest("input,textarea,select,button") || event.ctrlKey || event.metaKey || event.altKey || !manifest) return;
  if(event.code==="Space") {event.preventDefault();playing ? pause() : play().catch(error);}
  else if(["a","b","r"].includes(event.key.toLowerCase())) {guide={a:"A",b:"B",r:"off"}[event.key.toLowerCase()];mix();}
});
try {
  const session=await api("/api/session");manifest=session.manifest;state=session.state;
  const first=manifest.cases.findIndex(c=>!state.answers[c.id]?.finished);
  await changeCase(Math.max(0,first),true);
  if(state.unblinded) {assignmentKey=(await api("/api/unblind",{})).key;results();}
} catch(e) {error(e);$("saveStatus").textContent="Could not open the local review";}
requestAnimationFrame(draw);
