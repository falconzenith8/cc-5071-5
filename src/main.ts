import RAPIER from '@dimforge/rapier3d-compat';
import { SCENARIOS, ScenarioId } from './core/animator';
import { Simulator } from './core/simulator';
import { HandoffSnapshot, describeSubset } from './core/physics';
import { BONE_IDS, anchorErrors } from './core/rig';
import { CharacterView } from './view/CharacterView';
import { SceneView } from './view/SceneView';
import { parseReplay } from './core/replay';

const $ = <T extends HTMLElement>(sel: string): T => {
  const el = document.querySelector(sel);
  if (!el) throw new Error(`missing element ${sel}`);
  return el as T;
};

await RAPIER.init();

const canvas = $('#stage') as HTMLCanvasElement;
const sceneView = new SceneView(canvas);
const character = new CharacterView();
sceneView.scene.add(character.group);

const sim = new Simulator({ scenario: 'walk' });
sceneView.setScenario(sim.scenario);

let lastSnapshot: HandoffSnapshot | null = null;
let frozen = false;
let failureBone: string | null = null;

// ---------------------------------------------------------------- UI rendering

const metric = (label: string, value: string, ok: boolean): string =>
  `<div class="metric ${ok ? 'ok' : 'bad'}"><div class="metric-label">${label}</div><div class="metric-value">${value}</div></div>`;

const renderResult = (snap: HandoffSnapshot): void => {
  const fail = !snap.converged;
  const result = $('#result');
  result.classList.toggle('failed', fail);
  const rows = snap.bones.map((b) => {
    const isBoundary = Object.keys(snap.boundary).includes(b.id);
    return `<tr class="${b.owner === 'physics' ? 'phys' : ''} ${isBoundary ? 'boundary' : ''}">
      <td>${b.label}${isBoundary ? ' <span class="tag">边界</span>' : ''}</td>
      <td>${b.owner === 'physics' ? '受力' : '动画'}</td>
      <td class="${b.jointError > 0.01 ? 'bad' : ''}">${(b.jointError * 1000).toFixed(2)}</td>
      <td>${(b.posDelta * 1000).toFixed(1)}</td>
      <td class="${b.posResidual > 0.03 ? 'bad' : ''}">${(b.posResidual * 1000).toFixed(1)}</td>
      <td>${b.animSpeed.toFixed(2)}</td>
      <td>${b.inheritSpeed.toFixed(2)}</td>
      <td>${b.physSpeed.toFixed(2)}</td>
      <td>${b.hingeDeg === null ? '—' : b.hingeDeg.toFixed(1)}</td>
      <td>${b.groundContact ? `<span class="yes">是</span>${b.groundNormalY !== null ? ` ${b.groundNormalY.toFixed(2)}` : ''}` : '否'}</td>
    </tr>`;
  }).join('');

  result.innerHTML = `
    <div class="result-head">
      <span class="badge ${fail ? 'bad' : 'good'}">${fail ? '交接偏差' : '交接通过'}</span>
      <span class="muted">t=${snap.animTime.toFixed(3)}s · ${SCENARIOS[snap.scenario].label} · 步长 ${(snap.dt * 1000).toFixed(2)}ms</span>
    </div>
    <div class="metrics">
      ${metric('关节锚点误差', `${(snap.maxJointError * 1000).toFixed(2)} mm`, snap.maxJointError < 0.01)}
      ${metric('姿态残差', `${(snap.maxPosResidual * 1000).toFixed(1)} mm`, snap.maxPosResidual < 0.03)}
      ${metric('单步位移', `${(snap.maxPosDelta * 1000).toFixed(1)} mm`, true)}
      ${metric('单步转角', `${snap.maxRotDeltaDeg.toFixed(2)}°`, true)}
      ${metric('速度写入误差', `${(snap.maxInheritError * 1000).toFixed(3)} mm/s`, snap.maxInheritError < 1e-6)}
      ${metric('根速度 动画→物理', snap.rootOwner === 'physics' ? `${snap.rootSpeedBefore.toFixed(2)} → ${snap.rootSpeedAfter.toFixed(2)} m/s` : `${snap.rootSpeedBefore.toFixed(2)} m/s（动画持有）`, snap.rootOwner === 'physics' ? Math.abs(snap.rootSpeedAfter - snap.rootSpeedBefore) < 0.3 : true)}
      ${metric('落地接触', `左 ${snap.contacts.L ? '有' : '无'} · 右 ${snap.contacts.R ? '有' : '无'}`, true)}
    </div>
    ${snap.failedChecks.length ? `<ul class="fail-list">${snap.failedChecks.map((f) => `<li>${f}</li>`).join('')}</ul>` : ''}
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>骨骼</th><th>归属</th><th>关节 mm</th><th>单步 mm</th><th>残差 mm</th>
          <th>动画 m/s</th><th>写入 m/s</th><th>物理 m/s</th><th>铰链°</th><th>接地</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <p class="note">${snap.note}</p>
    <p class="note muted">单步位移含预期的一步动画位移；残差是去掉预期位移后的真实交接误差。</p>
  `;
};

const clearResult = (): void => {
  $('#result').innerHTML = `<p class="empty">尚未触发交接。点击「受击并接手」或等待自动触发。</p>`;
};

const updateHud = (): void => {
  $('#mode').textContent = sim.mode === 'animating' ? '动画控制' : '受力姿态';
  $('#mode').dataset.mode = sim.mode;
  $('#t-value').textContent = sim.time.toFixed(2);
  const physCount = BONE_IDS.filter((id) => sim.ownership.get(id) === 'physics').length;
  $('#owner-value').textContent = `${physCount} / ${BONE_IDS.length}`;
  const audit = sim.handoff.auditOwnership();
  const auditEl = $('#audit');
  auditEl.textContent = audit.ok ? '所有权审计通过' : `所有权冲突：${audit.conflicts.join(',')}`;
  auditEl.className = audit.ok ? 'audit ok' : 'audit bad';
  $('#message').textContent = sim.lastMessage;
  for (const el of document.querySelectorAll<HTMLElement>('[data-scenario]')) {
    el.classList.toggle('active', el.dataset.scenario === sim.scenario);
  }
  for (const el of document.querySelectorAll<HTMLElement>('[data-subset]')) {
    el.classList.toggle('active', el.dataset.subset === sim.subsetKind);
  }
  $('#speed-label').textContent = `${sim.speed.toFixed(2)} m/s`;
  ($('#speed') as HTMLInputElement).value = sim.speed.toFixed(2);
  $('#auto-handoff').classList.toggle('on', sim.autoHandoff);
  ($('#auto-handoff') as HTMLInputElement).checked = sim.autoHandoff;
  $('#replay-state').textContent = sim.replaying
    ? `回放中 ${sim.replayIndex}/${sim.replay?.inputs.length ?? 0}`
    : sim.recording
      ? '录制中'
      : sim.replay
        ? '已载入回放'
        : '未录制';
};

const setFrozen = (value: boolean): void => {
  frozen = value;
  $('#freeze').classList.toggle('on', value);
  $('#freeze').textContent = value ? '继续' : '暂停';
};

// ---------------------------------------------------------------- controls

const triggerStrike = (): void => {
  if (sim.mode === 'physics') {
    sim.message('已处于受力姿态；先「恢复动画」再触发下一次受击。');
    updateHud();
    return;
  }
  const snap = sim.strike();
  lastSnapshot = snap;
  failureBone = null;
  if (snap) {
    renderResult(snap);
    if (!snap.converged) {
      const worst = [...snap.bones].filter((b) => b.owner === 'physics').sort((a, b) => b.posResidual - a.posResidual)[0];
      failureBone = worst?.id ?? null;
      showFailure(snap, failureBone);
    } else {
      $('#failure').classList.remove('show');
    }
  }
  updateHud();
};

const showFailure = (snap: HandoffSnapshot, worst: string | null): void => {
  const box = $('#failure');
  box.classList.add('show');
  box.innerHTML = `<strong>交接失败</strong>
    <span>${snap.failedChecks.join('；')}</span>
    ${worst ? `<span>最差骨骼：${worst}（残差 ${(snap.bones.find((b) => b.id === worst)!.posResidual * 1000).toFixed(1)} mm）。</span>` : ''}
    <button id="retry">用同一录制输入重演</button>`;
  $('#retry').addEventListener('click', () => {
    box.classList.remove('show');
    sim.retryRecorded();
    lastSnapshot = null;
    clearResult();
    updateHud();
  });
};

const reset = (): void => {
  sim.reset();
  lastSnapshot = null;
  failureBone = null;
  character.clearTrail();
  clearResult();
  $('#failure').classList.remove('show');
  updateHud();
};

const wire = (): void => {
  for (const el of document.querySelectorAll<HTMLButtonElement>('[data-scenario]')) {
    el.addEventListener('click', () => {
      sim.setScenario(el.dataset.scenario as ScenarioId);
      sceneView.setScenario(sim.scenario);
      character.clearTrail();
      lastSnapshot = null;
      clearResult();
      $('#failure').classList.remove('show');
      updateHud();
    });
  }
  for (const el of document.querySelectorAll<HTMLButtonElement>('[data-subset]')) {
    el.addEventListener('click', () => {
      sim.setSubset(el.dataset.subset === 'upper' ? 'upper' : 'all');
      lastSnapshot = null;
      clearResult();
      updateHud();
    });
  }
  ($('#speed') as HTMLInputElement).addEventListener('input', (e) => {
    sim.setSpeed(Number((e.target as HTMLInputElement).value));
    updateHud();
  });
  $('#strike').addEventListener('click', triggerStrike);
  $('#restore').addEventListener('click', () => {
    sim.restoreManual();
    lastSnapshot = null;
    failureBone = null;
    clearResult();
    $('#failure').classList.remove('show');
    updateHud();
  });
  $('#reset').addEventListener('click', reset);
  $('#freeze').addEventListener('click', () => setFrozen(!frozen));
  ($('#auto-handoff') as HTMLInputElement).addEventListener('change', (e) => {
    sim.autoHandoff = (e.target as HTMLInputElement).checked;
    updateHud();
  });
  $('#record').addEventListener('click', () => {
    sim.beginRecording();
    reset();
    updateHud();
  });
  $('#export').addEventListener('click', () => {
    const text = sim.exportReplay();
    if (!text) {
      sim.message('没有可导出的回放：先录制或载入。');
      updateHud();
      return;
    }
    download('pose-handoff-replay.json', text);
    updateHud();
  });
  ($('#import') as HTMLInputElement).addEventListener('change', async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    try {
      const replay = parseReplay(await file.text());
      sim.loadReplay(replay);
      sceneView.setScenario(sim.scenario);
      lastSnapshot = null;
      clearResult();
      updateHud();
    } catch (err) {
      sim.message(`回放载入失败：${(err as Error).message}`);
      updateHud();
    }
  });
  $('#replay').addEventListener('click', () => {
    if (!sim.replay) {
      sim.message('请先载入回放文件。');
      updateHud();
      return;
    }
    sim.replayLoaded();
    sceneView.setScenario(sim.scenario);
    lastSnapshot = null;
    clearResult();
    updateHud();
  });
  window.addEventListener('resize', () => sceneView.resize());
};

const download = (name: string, text: string): void => {
  const blob = new Blob([text], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
};

// ---------------------------------------------------------------- loop

let last = performance.now();
let accumulator = 0;
const FIXED = 1 / 120;
const FOLLOW_Z = 2.35;
const FOLLOW_Y = 1.32;

const frame = (now: number): void => {
  const raw = Math.min((now - last) / 1000, 0.1);
  last = now;

  if (!frozen) {
    accumulator += raw;
    let guard = 0;
    while (accumulator >= FIXED && guard < 8) {
      sim.tick(FIXED);
      accumulator -= FIXED;
      guard++;
    }
    if (guard === 8) accumulator = 0;
  }

  // The automatic trigger and the manual button both route through the same
  // snapshot; render whenever a new one appears so the result panel never goes stale.
  if (sim.lastSnapshot !== lastSnapshot) {
    lastSnapshot = sim.lastSnapshot;
    if (lastSnapshot) {
      renderResult(lastSnapshot);
      if (!lastSnapshot.converged) {
        const worst = [...lastSnapshot.bones]
          .filter((b) => b.owner === 'physics')
          .sort((a, b) => b.posResidual - a.posResidual)[0];
        failureBone = worst?.id ?? null;
        showFailure(lastSnapshot, failureBone);
      }
    }
  }

  const pose = sim.pose;
  character.setPose(pose, sim.ownership);
  character.showJointErrors(pose, anchorErrors(pose), 0.008);
  character.pushTrail(pose.pelvis.pos);
  sceneView.showContacts(sim.handoff.contactPoints());
  $('#scenario-desc').textContent = SCENARIOS[sim.scenario].description;
  $('#subset-desc').textContent = describeSubset(sim.subset);

  const root = pose.pelvis.pos;
  sceneView.setCamera(
    { x: root.x - 2.05, y: FOLLOW_Y, z: FOLLOW_Z },
    { x: root.x + 0.25, y: root.y - 0.02, z: root.z },
  );
  sceneView.render();
  updateHud();
  requestAnimationFrame(frame);
};

wire();
sceneView.resize();
clearResult();
updateHud();
requestAnimationFrame(frame);

// Expose a minimal harness for the end-to-end test; no gameplay logic lives here.
(window as unknown as { __lab: unknown }).__lab = {
  sim,
  snapshot: () => lastSnapshot,
  failureBone: () => failureBone,
};
