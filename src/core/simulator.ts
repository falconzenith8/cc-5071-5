
import { Animator, AnimSample, SCENARIOS, ScenarioId } from './animator';
import { DEFAULT_SUBSET, ObstacleSpec, PoseHandoff, HandoffSnapshot, UPPER_BODY_SUBSET } from './physics';
import { BONE_IDS, BoneId, Pose } from './rig';
import { Vec3 } from './math';
import { Replay, emptyReplay, serializeReplay } from './replay';
import { RecoveryController, RecoveryState, planRecovery } from './recovery';
import { FallenOrientation, buildFallenPose } from './recoveryPoses';

export type Mode = 'animating' | 'physics' | 'falling' | 'recovering';
export type SubsetKind = 'all' | 'upper';
export type EnvironmentKind = 'clear' | 'wall' | 'ceiling';

export interface SimulatorOptions {
  scenario?: ScenarioId;
  speed?: number;
  subset?: SubsetKind;
  environment?: EnvironmentKind;
}

/**
 * Drives the drill. All controls talk to this object; the view only mirrors it.
 * The handoff core (PoseHandoff) owns the bodies, this owns the mode machine.
 */
export const ENVIRONMENT_OBSTACLES: Record<Exclude<EnvironmentKind, 'clear'>, (root: Vec3) => ObstacleSpec[]> = {
  // Low slab 0.5 m above the fallen pelvis: lying and tucking fit, standing
  // headroom does not, so an in-place rise must be reported unrecoverable.
  ceiling: (root) => [
    {
      id: 'low-ceiling',
      label: '低顶',
      kind: 'ceiling',
      center: { x: root.x + 0.25, y: root.y + 0.5, z: root.z },
      halfExtents: { x: 1.1, y: 0.06, z: 0.85 },
    },
  ],
  // Wall just beyond the head in the facing (+X) direction, close enough that
  // the planner's forward probe reports no room to unfold the rising torso.
  // The anchor passed at drill time is the fallen head position (see fallDown).
  wall: (root) => [
    {
      id: 'retaining-wall',
      label: '挡墙',
      kind: 'wall',
      center: { x: root.x + 0.05, y: 0.45, z: root.z },
      halfExtents: { x: 0.06, y: 0.6, z: 0.7 },
    },
  ],
};

export class Simulator {
  animator: Animator;
  handoff: PoseHandoff;
  scenario: ScenarioId;
  speed: number;
  subset: BoneId[];
  environment: EnvironmentKind;
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
  /** Contact-aware get-up state machine; null until a recovery is requested. */
  recovery: RecoveryController | null = null;
  /** Snapshot of the terminal recovery state, retained after control returns. */
  private recoverySnapshot: RecoveryState | null = null;

  constructor(opts: SimulatorOptions = {}) {
    this.scenario = opts.scenario ?? 'walk';
    this.speed = opts.speed ?? SCENARIOS[this.scenario].suggestedSpeed;
    this.subset = opts.subset === 'upper' ? [...UPPER_BODY_SUBSET] : [...DEFAULT_SUBSET];
    this.environment = opts.environment ?? 'clear';
    this.animator = new Animator(this.scenario, this.speed);
    this.handoff = new PoseHandoff(this.scenario, this.animator, this.environmentObstacles());
    this.handoff.build(this.animator.sample(0), this.subset);
  }

  private environmentObstacles(): ObstacleSpec[] {
    if (this.environment === 'clear') return [];
    const root = this.animator.sample(SCENARIOS[this.scenario].handoffTime).pose.pelvis.pos;
    return ENVIRONMENT_OBSTACLES[this.environment](root);
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
    this.recovery = null;
    this.recoverySnapshot = null;
    this.handoff.setObstacles(this.environmentObstacles());
    this.handoff.build(this.animator.sample(0), this.subset);
    this.message('已回到动画控制，t=0。');
  }

  setScenario(scenario: ScenarioId): void {
    this.scenario = scenario;
    this.speed = SCENARIOS[scenario].suggestedSpeed;
    this.reset();
  }

  setEnvironment(environment: EnvironmentKind): void {
    this.environment = environment;
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

    if (this.mode === 'recovering' && this.recovery) {
      // The get-up drives the dynamic chain itself; the clip only supplies the
      // final blend target and stays rebased onto the body's current root.
      this.recovery.step(sample.pose);
      const state = this.recovery.getState();
      this.lastMessage = state.message;
      if (state.status === 'complete') {
        this.completeRecovery(sample);
      }
      this.handoff.stepWorldOnly();
      this.trace.push({ t: this.time, root: this.handoff.poseFromWorld().pelvis.pos, handoff: false });
      this.finishRecordingIfDue();
      return;
    }

    if (this.mode === 'falling') {
      // Guided fall steps the world internally; just record the trace.
      this.trace.push({ t: this.time, root: this.handoff.poseFromWorld().pelvis.pos, handoff: false });
      this.finishRecordingIfDue();
      return;
    }

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

    this.finishRecordingIfDue();
  }

  private finishRecordingIfDue(): void {
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
    this.recovery = null;
    this.recoverySnapshot = null;
    this.handoff.restoreAnimation();
    this.handoff.resetDamping();
    this.handoff.build(this.animator.sample(this.time), this.subset);
    this.handoffAt = -1;
    if (record) this.recordInput('restore');
    this.message('已恢复动画控制。');
  }

  /**
   * Guided-fall entry for the contact drill: the current chain is moved through
   * a deterministic fallen configuration (supine/prone/side) while every frame
   * runs the collision pipeline, so the contacts the planner reads are real.
   */
  fallDown(orientation: Exclude<FallenOrientation, 'upright'>): void {
    if (this.mode === 'recovering' || this.mode === 'falling') return;
    const current = this.handoff.poseFromWorld().pelvis.pos;
    const target = buildFallenPose(this.scenario, orientation, current, 0);
    this.mode = 'falling';
    this.recovery = null;
    this.recoverySnapshot = null;
    // 36 fixed steps (~0.3 s) of kinematic motion through live collision.
    this.handoff.guidedFall(target, 36);
    // Let the body settle dynamically for a short window so manifolds stabilize.
    this.handoff.beginDynamicRecovery();
    for (let i = 0; i < 24; i++) this.handoff.stepWorldOnly();
    // Place wall/ceiling constraints around the settled fallen body. The wall
    // sits just beyond the actual head position (not a fixed root offset, which
    // drifts as the chain settles); the ceiling is 0.5 m above the lying pelvis.
    if (this.environment !== 'clear') {
      const fallen = this.handoff.poseFromWorld();
      const anchor =
        this.environment === 'wall'
          ? { ...fallen.head.pos, z: fallen.pelvis.pos.z }
          : fallen.pelvis.pos;
      this.handoff.setObstacles(ENVIRONMENT_OBSTACLES[this.environment](anchor));
    }
    this.mode = 'physics';
    this.handoffAt = this.time;
    this.message(`受控倒地完成（${orientation}）：接触与净空已就绪，可启动接触感知恢复。`);
  }

  /** Analyze the current fallen state without committing to a get-up. */
  analyzeRecovery() {
    return planRecovery(this.handoff, this.scenario);
  }

  /** Start the contact-aware get-up. Returns the plan, including any blocking reason. */
  beginRecovery() {
    if (this.mode === 'recovering') return this.recovery!.getPlan();
    this.recovery = new RecoveryController(this.handoff, this.scenario);
    const plan = this.recovery.start(() => undefined);
    if (plan.feasible) {
      this.mode = 'recovering';
      this.message(this.recovery.getState().message);
    } else {
      this.mode = 'physics';
      this.message(this.recovery.getState().message);
    }
    return plan;
  }

  /** Re-evaluate a paused/failed recovery against the current scene. */
  retryRecovery() {
    if (!this.recovery) return null;
    const plan = this.recovery.retry();
    if (plan?.feasible) this.mode = 'recovering';
    this.message(this.recovery.getState().message);
    return plan;
  }

  getRecoveryState(): RecoveryState | null {
    if (this.recovery) return this.recovery.getState();
    return this.recoverySnapshot;
  }

  /**
   * Final handoff after the blend: rebuild the chain at the clip pose rebased
   * onto the recovered root, then flip ownership back to animation. No
   * teleport — the clip root is offset to the body's actual position.
   */
  private completeRecovery(sample: AnimSample): void {
    const current = this.handoff.poseFromWorld();
    const delta: Vec3 = {
      x: current.pelvis.pos.x - sample.pose.pelvis.pos.x,
      y: current.pelvis.pos.y - sample.pose.pelvis.pos.y,
      z: current.pelvis.pos.z - sample.pose.pelvis.pos.z,
    };
    const rebased = {} as Pose;
    for (const id of BONE_IDS) {
      rebased[id] = {
        pos: {
          x: sample.pose[id].pos.x + delta.x,
          y: sample.pose[id].pos.y + delta.y,
          z: sample.pose[id].pos.z + delta.z,
        },
        rot: sample.pose[id].rot,
      };
    }
    // build() only consumes `.pose`, but keep the sample shape for the API.
    const rebasedSample: AnimSample = { ...sample, pose: rebased };
    this.recoverySnapshot = this.recovery!.getState();
    this.handoff.restoreAnimation();
    this.handoff.resetDamping();
    this.handoff.build(rebasedSample, this.subset);
    this.mode = 'animating';
    this.recovery = null;
    this.handoffAt = -1;
    this.message('接触感知起身完成：控制已平滑交回动画，根位置沿用实际恢复位置。');
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
