
import { Animator, SCENARIOS, ScenarioId } from './animator';
import { DEFAULT_SUBSET, PoseHandoff, HandoffSnapshot, UPPER_BODY_SUBSET } from './physics';
import { BONE_IDS, BoneId, Pose } from './rig';
import { Vec3 } from './math';
import { Replay, emptyReplay, serializeReplay } from './replay';

export type Mode = 'animating' | 'physics';
export type SubsetKind = 'all' | 'upper';

export interface SimulatorOptions {
  scenario?: ScenarioId;
  speed?: number;
  subset?: SubsetKind;
}

/**
 * Drives the drill. All controls talk to this object; the view only mirrors it.
 * The handoff core (PoseHandoff) owns the bodies, this owns the mode machine.
 */
export class Simulator {
  animator: Animator;
  handoff: PoseHandoff;
  scenario: ScenarioId;
  speed: number;
  subset: BoneId[];
  mode: Mode = 'animating';
  time = 0;
  autoHandoff = true;
  lastSnapshot: HandoffSnapshot | null = null;
  autoRestoreAt: number | null = null;
  recording: Replay | null = null;
  replay: Replay | null = null;
  replayIndex = 0;
  replaying = false;
  /** Deterministic input log of the current run; this is what "replay" consumes. */
  inputLog: { time: number; action: 'impact' | 'restore'; subset: BoneId[] }[] = [];
  /** Simulation time of the last handoff, for the assist and for trap detection. */
  handoffAt = -1;
  /** Root trace for the current run, recorded so a replay can be compared. */
  trace: { t: number; root: Vec3; handoff: boolean }[] = [];
  /** The impulse used by the current run's handoff, so retries replay it verbatim. */
  lastImpact: { bone: BoneId; impulse: Vec3 } | null = null;
  lastMessage = '';

  constructor(opts: SimulatorOptions = {}) {
    this.scenario = opts.scenario ?? 'walk';
    this.speed = opts.speed ?? SCENARIOS[this.scenario].suggestedSpeed;
    this.subset = opts.subset === 'upper' ? [...UPPER_BODY_SUBSET] : [...DEFAULT_SUBSET];
    this.animator = new Animator(this.scenario, this.speed);
    this.handoff = new PoseHandoff(this.scenario, this.animator);
    this.handoff.build(this.animator.sample(0), this.subset);
  }

  dispose(): void {
    this.handoff.dispose();
  }

  get pose(): Pose {
    return this.mode === 'physics' ? this.handoff.poseFromWorld() : this.animator.sample(this.time).pose;
  }

  get ownership(): Map<BoneId, 'animation' | 'physics'> {
    return this.handoff.getOwnership();
  }

  /** Restart the drill from t=0 at the configured scenario/speed. */
  reset(): void {
    this.animator.configure(this.scenario, this.speed);
    this.time = 0;
    this.mode = 'animating';
    this.lastSnapshot = null;
    this.autoRestoreAt = null;
    this.handoffAt = -1;
    this.trace = [];
    this.replayIndex = 0;
    this.inputLog = [];
    this.lastImpact = null;
    this.handoff.build(this.animator.sample(0), this.subset);
    this.message('已回到动画控制，t=0。');
  }

  setScenario(scenario: ScenarioId): void {
    this.scenario = scenario;
    this.speed = SCENARIOS[scenario].suggestedSpeed;
    this.reset();
  }

  setSpeed(speed: number): void {
    this.speed = speed;
    if (this.mode === 'animating') this.reset();
  }

  setSubset(kind: SubsetKind): void {
    this.subset = kind === 'upper' ? [...UPPER_BODY_SUBSET] : [...DEFAULT_SUBSET];
    this.reset();
  }

  get subsetKind(): SubsetKind {
    return this.subset.length === BONE_IDS.length ? 'all' : 'upper';
  }

  /** Manual "受击" control: hand off at the current animation frame, then apply the impact. */
  strike(): HandoffSnapshot | null {
    const snap = this.handoffNow({ bone: 'chest', impulse: { x: 55, y: 8, z: -4 } });
    if (snap) this.recordInput('impact');
    return snap;
  }

  /** Manual restore control (records the input so the run can be replayed). */
  restoreManual(): void {
    this.restore(!this.replaying);
  }

  /**
   * Failure feedback path: re-run the exact recorded inputs from t=0 through the
   * same handoff core. Used when a handoff fails so the failing frame is
   * reproducible instead of being retried with different parameters.
   */
  retryRecorded(): HandoffSnapshot | null {
    if (!this.inputLog.length) return null;
    const impact = this.lastImpact;
    const log = this.inputLog.map((e) => ({ ...e }));
    this.reset();
    this.autoHandoff = false;
    this.replaying = true;
    this.replay = {
      version: 1,
      scenario: this.scenario,
      seed: 1,
      speed: this.speed,
      subset: [...this.subset],
      handoffTime: SCENARIOS[this.scenario].handoffTime,
      impact: impact ? { ...impact, atTime: log[0]?.time ?? 0 } : null,
      dt: this.handoff.world.timestep,
      inputs: log.map((e) => ({ time: e.time, action: e.action })),
      trace: [],
      createdAt: new Date().toISOString(),
    };
    this.replayIndex = 0;
    this.message('失败反馈：从 t=0 用同一输入重演。');
    return null;
  }

  /** Automatic trigger at the scenario's handoff time. */
  tick(dt: number): void {
    this.time += dt;

    // Replays feed the recorded inputs back at the same simulated times.
    if (this.replaying && this.replay) {
      const inputs = this.replay.inputs;
      while (this.replayIndex < inputs.length && inputs[this.replayIndex].time <= this.time) {
        const input = inputs[this.replayIndex++];
        if (input.action === 'impact') {
          this.handoffNow(this.replay.impact ? { bone: this.replay.impact.bone, impulse: this.replay.impact.impulse } : null);
        }
        if (input.action === 'restore') this.restore(true);
      }
    }

    const sample = this.animator.sample(this.time);
    this.handoff.driveAnimation(sample);
    if (this.mode === 'physics') {
      this.handoff.applyStabilityAssist(sample, this.time - this.handoffAt);
    }
    this.handoff.stepWorldOnly();

    if (this.mode === 'physics') {
      this.trace.push({ t: this.time, root: this.handoff.poseFromWorld().pelvis.pos, handoff: false });
      if (this.autoRestoreAt !== null && this.time >= this.autoRestoreAt) {
        this.autoRestoreAt = null;
        this.restore(true);
      }
    } else {
      this.trace.push({ t: this.time, root: sample.pose.pelvis.pos, handoff: false });
      if (this.autoHandoff && this.time >= SCENARIOS[this.scenario].handoffTime) {
        const snap = this.handoffNow(null);
        if (snap) this.recordInput('impact');
      }
    }

    if (this.recording && this.mode === 'physics' && this.time - this.recording.handoffTime > 1.6) {
      this.finishRecording();
    }
  }

  private recordInput(action: 'impact' | 'restore'): void {
    this.inputLog.push({ time: this.time, action, subset: [...this.subset] });
    if (this.recording) this.recording.inputs.push({ time: this.time, action });
  }

  /** Core handoff entry point. No stance reset: sample is the current clip frame. */
  handoffNow(impact: { bone: BoneId; impulse: Vec3 } | null): HandoffSnapshot | null {
    if (this.mode === 'physics') return this.lastSnapshot;
    const sample = this.animator.sample(this.time);
    const snapshot = this.handoff.handoff(sample, { subset: this.subset, impact });
    this.lastSnapshot = snapshot;
    this.mode = 'physics';
    this.handoffAt = this.time;
    this.lastImpact = impact ? { bone: impact.bone, impulse: { ...impact.impulse } } : null;
    this.trace.push({ t: this.time, root: sample.pose.pelvis.pos, handoff: true });
    if (this.recording) {
      this.recording.handoffTime = this.time;
      this.recording.impact = impact ? { ...impact, atTime: this.time } : null;
    }
    const rootSpeed = snapshot.rootOwner === 'physics'
      ? `根速度 ${snapshot.rootSpeedBefore.toFixed(2)}→${snapshot.rootSpeedAfter.toFixed(2)} m/s`
      : `根速度 ${snapshot.rootSpeedBefore.toFixed(2)} m/s（骨盆仍由动画持有）`;
    this.message(
      snapshot.converged
        ? `交接通过：关节误差 ${(snapshot.maxJointError * 1000).toFixed(2)} mm，残差 ${(snapshot.maxPosResidual * 1000).toFixed(1)} mm，${rootSpeed}。`
        : `交接存在偏差：${snapshot.failedChecks.join('；')}`,
    );
    return snapshot;
  }

  /** Restore animation control, rebuilding the chain at the clip's current pose. */
  restore(record = false): void {
    this.mode = 'animating';
    this.handoff.restoreAnimation();
    this.handoff.build(this.animator.sample(this.time), this.subset);
    this.handoffAt = -1;
    if (record) this.recordInput('restore');
    this.message('已恢复动画控制。');
  }

  /**
   * Replays a recording from t=0. Inputs are applied at the same simulated times,
   * so the run is reproducible frame-for-frame on the same build.
   */
  replayFromStart(): void {
    if (!this.replay) return;
    this.loadReplay(this.replay, true);
  }

  beginRecording(): void {
    this.recording = emptyReplay(this.scenario, this.speed, this.subset);
    this.message('录制中：触发受击后自动停止。');
  }

  finishRecording(): Replay | null {
    if (!this.recording) return null;
    this.recording.trace = this.trace.slice();
    this.recording.dt = this.handoff.world.timestep;
    const out = this.recording;
    this.recording = null;
    this.lastMessage = `已生成回放：${out.trace.length} 帧，${out.inputs.length} 个输入。`;
    return out;
  }

  exportReplay(): string | null {
    const replay = this.recording ? this.finishRecording() : this.replay;
    return replay ? serializeReplay(replay) : null;
  }

  loadReplay(replay: Replay, start = false): void {
    this.replay = replay;
    this.scenario = replay.scenario;
    this.speed = replay.speed;
    this.subset = [...replay.subset];
    this.animator.configure(replay.scenario, replay.speed);
    this.time = 0;
    this.mode = 'animating';
    this.lastSnapshot = null;
    this.trace = [];
    this.replayIndex = 0;
    this.replaying = start;
    this.autoHandoff = false;
    this.handoff.build(this.animator.sample(0), this.subset);
    this.message(
      start
        ? `开始回放：${replay.scenario}，${replay.inputs.length} 个输入。`
        : `已载入回放：${replay.scenario}，${replay.inputs.length} 个输入；点「回放」开始。`,
    );
  }

  replayLoaded(): void {
    if (!this.replay) return;
    const replay = this.replay;
    this.loadReplay(replay, true);
  }

  stopReplay(): void {
    this.replaying = false;
    this.autoHandoff = true;
  }

  message(text: string): void {
    this.lastMessage = text;
  }
}
