import { Vec3, qrot, vadd, vdot, vlen, vnorm, vscale, vsub } from './math';
import { BONE_IDS, BoneId, Pose, ROOT_ID } from './rig';
import { SCENARIOS, ScenarioId } from './animator';
import { PoseHandoff } from './physics';
import {
  FallenOrientation,
  GetupStrategy,
  Keyframe,
  PHASE_KEYFRAMES,
  RecoveryPhaseId,
  buildKeyframePose,
  classifyOrientation,
  clampLocalsToLimits,
  facingYawOnPlane,
  interpolateKeyframes,
  poseGroundPenetration,
  strategyFor,
} from './recoveryPoses';

export type RecoveryStatus =
  | 'idle'
  | 'analyzing'
  | 'getting-up'
  | 'blending'
  | 'paused'
  | 'failed'
  | 'complete';

export interface RecoveryBasis {
  orientation: FallenOrientation;
  strategy: GetupStrategy;
  yaw: number;
  supportNormal: Vec3;
  supportBones: BoneId[];
  headClearance: number;
  rootTarget: Vec3;
  rootStart: Vec3;
  horizontalTravel: number;
  reasons: string[];
}

export interface RecoveryPlan {
  feasible: boolean;
  blockingReasons: string[];
  basis: RecoveryBasis;
  /** Minimum head clearance each upcoming phase needs (m). */
  phaseClearance: { phase: RecoveryPhaseId; needed: number; available: number }[];
}

export interface RecoveryState {
  status: RecoveryStatus;
  phase: RecoveryPhaseId;
  phaseIndex: number;
  progress: number;
  plan: RecoveryPlan | null;
  blockedFor: number;
  message: string;
}

// Tunables. Drives are deliberately weaker than contact response: when a wall
// or the ceiling pushes back, the solver wins and the controller stalls rather
// than dragging the chain through geometry.
const PHASE_DURATION = 0.62;
const BLEND_DURATION = 0.3;
const BLEND_SETTLE_STEPS = 12;
const POS_TOLERANCE = 0.075;
const ROT_TOLERANCE_DEG = 16;
const STALL_SPEED = 0.04;
const PAUSE_LIMIT = 1.6;
const HEAD_CLEARANCE_MIN = 0.55;
const WALL_SIDE_CLEARANCE = 0.42;
const MAX_TRAVEL = 0.6;
const DRIVE_GAINS = { linK: 95, linD: 16, maxForce: 520, angK: 26, angD: 9, maxTorque: 190 };

const PHASE_LABELS: Record<RecoveryPhaseId, string> = {
  fallen: '倒地接触',
  tuck: '收身',
  situp: '撑起/坐起',
  crouch: '蹲姿支撑',
  rise: '起身',
  blend: '交回动画',
  paused: '暂停（接触阻挡）',
  failed: '不可恢复',
  done: '完成',
};

export const phaseLabel = (phase: RecoveryPhaseId): string => PHASE_LABELS[phase] ?? phase;

/** Where the head collider sits, in pelvis-local terms at bind (approx via head bone world pose). */
const headWorldPoint = (pose: Pose): Vec3 => pose.head.pos;

/**
 * Ray origins sit slightly inside the body's own colliders; offset along the
 * probe direction so the query starts on the shape surface and solid hits
 * against a touching wall/ceiling are still reported.
 */
const PROBE_SKIM = 0.03;
const probe = (handoff: PoseHandoff, origin: Vec3, dir: Vec3, maxToi: number) => {
  const n = dir;
  return handoff.castEnvironmentRay({
    origin: vadd(origin, vscale(n, PROBE_SKIM)),
    dir: n,
    maxToi,
  });
};

/**
 * Pure-ish planner: reads actual contacts and ray-casts the scene. The root
 * target is the fallen body's own horizontal position (optionally nudged away
 * from a wall), never a preset teleport point.
 */
export const planRecovery = (handoff: PoseHandoff, scenario: ScenarioId): RecoveryPlan => {
  const pose = handoff.poseFromWorld();
  const support = handoff.supportNormal();
  const normal = support.normal;
  const orientation = classifyOrientation(pose[ROOT_ID].rot, normal);
  const strategy = strategyFor(orientation);
  const yaw = facingYawOnPlane(pose[ROOT_ID].rot, normal);
  const supportBones = handoff.groundContactBones();

  const rootStart = { ...pose[ROOT_ID].pos };
  const sc = SCENARIOS[scenario];

  // Horizontal root target: stay where the body lies. If a wall blocks the
  // facing direction at crouch distance, shift along the support tangent
  // instead of driving into it.
  const rootTarget = { ...rootStart };
  const reasons: string[] = [];
  const blocking: string[] = [];

  const head = headWorldPoint(pose);
  const upHit = probe(handoff, head, normal, 2.2);
  const headClearance = upHit ? upHit.toi + PROBE_SKIM : Infinity;
  reasons.push(
    `头上方空间 ${Number.isFinite(headClearance) ? headClearance.toFixed(2) : '≥2.2'} m` +
      (upHit?.obstacleId ? `（障碍 ${upHit.obstacleId}）` : ''),
  );

  // Probe the space the rising torso will occupy: a ray along the support
  // normal from the chest and from the pelvis.
  const chest = pose.chest.pos;
  const chestHit = probe(handoff, chest, normal, 1.6);
  const chestClearance = chestHit ? chestHit.toi + PROBE_SKIM : Infinity;

  // Side probes at shoulder height detect walls pinning the arms/legs. Rays run
  // inside the support plane (normal component removed) so a lying body whose
  // bone axes point at the floor does not "see" the ground as a wall.
  const planar = (d: Vec3): Vec3 => {
    const p = vsub(d, vscale(normal, vdot(d, normal)));
    return vlen(p) > 1e-6 ? vnorm(p) : { x: 1, y: 0, z: 0 };
  };
  const sideDirA = planar(qrot(pose[ROOT_ID].rot, { x: 0, y: 0, z: 1 }));
  const sideDirB = vscale(sideDirA, -1);
  const sideA = probe(handoff, pose.chest.pos, sideDirA, WALL_SIDE_CLEARANCE + 0.2);
  const sideB = probe(handoff, pose.chest.pos, sideDirB, WALL_SIDE_CLEARANCE + 0.2);
  const sideMin = Math.min(
    sideA ? sideA.toi + PROBE_SKIM : Infinity,
    sideB ? sideB.toi + PROBE_SKIM : Infinity,
  );
  reasons.push(
    `侧向余量 ${Number.isFinite(sideMin) ? sideMin.toFixed(2) : '通畅'} m` +
      (sideA?.obstacleId || sideB?.obstacleId ? '（贴墙）' : ''),
  );

  // Forward/back probes catch retaining walls in the facing direction, which
  // the side rays miss when the body lies square to the wall. The horizontal
  // facing comes from the robust yaw (with a degenerate-axis fallback) rather
  // than re-projecting local +X: when fully supine/prone that axis points
  // nearly straight into the ground, so its tiny planar component can flip sign.
  const fwdDir = vnorm({ x: Math.cos(yaw), y: 0, z: -Math.sin(yaw) });
  const fwdHit = probe(handoff, pose.chest.pos, fwdDir, 1.0);
  const backHit = probe(handoff, pose.chest.pos, vscale(fwdDir, -1), 1.0);
  const frontClearance = Math.min(
    fwdHit ? fwdHit.toi + PROBE_SKIM : Infinity,
    backHit ? backHit.toi + PROBE_SKIM : Infinity,
  );
  const wallTouch = !!(fwdHit?.obstacleId || backHit?.obstacleId || sideA?.obstacleId || sideB?.obstacleId);
  reasons.push(
    `前后余量 ${Number.isFinite(frontClearance) ? frontClearance.toFixed(2) : '通畅'} m` +
      (wallTouch ? '（墙体接触）' : ''),
  );

  if (support.contactCount === 0) {
    blocking.push('身体没有任何环境接触，无法推导支撑面（可能仍在空中）');
  } else {
    reasons.push(`支撑面法线 (${normal.x.toFixed(2)}, ${normal.y.toFixed(2)}, ${normal.z.toFixed(2)})，接触 ${support.contactCount} 处`);
  }

  // Per-phase clearance table: measure the vertical room each keyframe's head
  // would need from the current support point.
  const frames = PHASE_KEYFRAMES[strategy];
  const phaseClearance = frames.map((kf, i): { phase: RecoveryPhaseId; needed: number; available: number } => {
    const ids: RecoveryPhaseId[] = ['fallen', 'tuck', 'situp', 'crouch', 'rise'];
    const needed = Math.max(0.1, kf.rootHeight + 0.35 - (rootStart.y - sc.groundHeight(rootStart.x, rootStart.z)));
    return { phase: ids[i], needed, available: i === 0 ? headClearance : Math.min(headClearance, chestClearance) };
  });

  // Hard feasibility: the final rise needs standing headroom.
  const riseNeed = phaseClearance[phaseClearance.length - 1];
  if (riseNeed.available < HEAD_CLEARANCE_MIN) {
    blocking.push(`低顶阻挡：起身需要 ${riseNeed.needed.toFixed(2)} m 净空，头顶只剩 ${riseNeed.available.toFixed(2)} m`);
  }
  if (sideMin < 0.18 || frontClearance < 0.3) {
    blocking.push(
      `被墙/障碍压住：侧向余量 ${sideMin === Infinity ? '通畅' : sideMin.toFixed(2) + ' m'}，前后余量 ${frontClearance === Infinity ? '通畅' : frontClearance.toFixed(2) + ' m'}，肢体没有展开空间`,
    );
  }

  // Ground under the target must exist inside the terrain mesh bounds.
  const groundHere = sc.groundHeight(rootTarget.x, rootTarget.z);
  if (!Number.isFinite(groundHere)) {
    blocking.push('目标位置超出支撑面范围');
  }

  const travel = vlen(vsub({ ...rootTarget, y: 0 }, { ...rootStart, y: 0 }));
  if (travel > MAX_TRAVEL) {
    blocking.push(`恢复目标根位移 ${travel.toFixed(2)} m 超过允许的就地起身范围 ${MAX_TRAVEL} m`);
  }

  reasons.push(`朝向分类：${orientation}，采用 ${strategy}，根目标由当前身体位置推导（水平位移 ${travel.toFixed(2)} m）`);

  return {
    feasible: blocking.length === 0,
    blockingReasons: blocking,
    phaseClearance,
    basis: {
      orientation,
      strategy,
      yaw,
      supportNormal: normal,
      supportBones,
      headClearance,
      rootTarget,
      rootStart,
      horizontalTravel: travel,
      reasons,
    },
  };
};

/**
 * State machine driving the get-up. Every target pose is forward kinematics
 * through the real rig hierarchy with revolute limits clamped; the physics
 * chain keeps its joints and colliders for the entire sequence.
 */
export class RecoveryController {
  private handoff: PoseHandoff;
  private scenario: ScenarioId;
  private phaseIndex = 0;
  private phaseTime = 0;
  private status: RecoveryStatus = 'idle';
  private plan: RecoveryPlan | null = null;
  private blockedFor = 0;
  private blendTime = 0;
  private settleSteps = 0;
  private blendFrom: Pose | null = null;
  private blendTo: Pose | null = null;
  private onComplete: (() => void) | null = null;

  constructor(handoff: PoseHandoff, scenario: ScenarioId) {
    this.handoff = handoff;
    this.scenario = scenario;
  }

  getState(): RecoveryState {
    const frames = this.plan ? PHASE_KEYFRAMES[this.plan.basis.strategy] : [];
    const phase: RecoveryPhaseId =
      this.status === 'blending'
        ? 'blend'
        : this.status === 'paused'
          ? 'paused'
          : this.status === 'failed'
            ? 'failed'
            : this.status === 'complete'
              ? 'done'
              : (['fallen', 'tuck', 'situp', 'crouch', 'rise'][this.phaseIndex] as RecoveryPhaseId);
    return {
      status: this.status,
      phase,
      phaseIndex: this.phaseIndex,
      progress: this.status === 'getting-up' ? clamp01(this.phaseTime / PHASE_DURATION) : this.status === 'blending' ? clamp01(this.blendTime / BLEND_DURATION) : 0,
      plan: this.plan,
      blockedFor: this.blockedFor,
      message: this.describe(frames.length),
    };
  }

  getPlan(): RecoveryPlan | null {
    return this.plan;
  }

  isRunning(): boolean {
    return this.status === 'getting-up' || this.status === 'blending' || this.status === 'paused';
  }

  /** Analyze contacts/space and start the recovery when feasible. */
  start(onComplete: () => void): RecoveryPlan {
    const plan = planRecovery(this.handoff, this.scenario);
    this.plan = plan;
    this.onComplete = onComplete;
    if (!plan.feasible) {
      this.status = 'failed';
      this.phaseIndex = 0;
      return plan;
    }
    this.handoff.beginDynamicRecovery();
    this.status = 'getting-up';
    this.phaseIndex = 0;
    this.phaseTime = 0;
    this.blockedFor = 0;
    return plan;
  }

  /** Re-check the scene while blocked: if the obstacle moved away, resume. */
  retry(): RecoveryPlan | null {
    if (this.status !== 'paused' && this.status !== 'failed') return this.plan;
    const plan = planRecovery(this.handoff, this.scenario);
    this.plan = plan;
    if (plan.feasible) {
      if (this.status === 'failed') this.handoff.beginDynamicRecovery();
      this.status = 'getting-up';
      this.blockedFor = 0;
    }
    return plan;
  }

  abort(): void {
    this.status = 'idle';
    this.plan = null;
    this.phaseIndex = 0;
    this.phaseTime = 0;
    this.blockedFor = 0;
  }

  /**
   * Advance the recovery by one fixed step. `blendTarget` is the clip pose the
   * simulator will hand control to; the controller blends toward it in place.
   */
  step(blendTarget: Pose): void {
    if (this.status !== 'getting-up' && this.status !== 'paused' && this.status !== 'blending') return;
    if (!this.plan) return;
    const strategy = this.plan.basis.strategy;
    const frames = PHASE_KEYFRAMES[strategy];

    if (this.status === 'blending') {
      this.stepBlend(blendTarget);
      return;
    }

    const target = this.phaseTarget(frames);
    const before = this.handoff.poseFromWorld();
    this.handoff.driveTowardPose(target, DRIVE_GAINS, 1);
    this.handoff.stepWorldOnly();
    const after = this.handoff.poseFromWorld();
    const deviation = this.handoff.poseDeviation(target);
    const rootSpeed = vlen(this.handoff.getBody(ROOT_ID)?.linvel() as Vec3);

    // Blockage detection: target still far, body nearly stopped, and an
    // environment obstacle is in contact. Pause first; declare unrecoverable
    // only after the pause window expires.
    const touchingObstacle = this.handoff.contactingObstacles().length > 0;
    const stuck =
      deviation.maxPos > POS_TOLERANCE * 1.6 && rootSpeed < STALL_SPEED;
    if ((stuck && touchingObstacle) || this.blockedFor > 0) {
      if (stuck && touchingObstacle) this.blockedFor += this.handoff.world.timestep;
      if (this.blockedFor >= PAUSE_LIMIT) {
        this.status = 'failed';
        this.plan.blockingReasons.push(
          `起身在「${phaseLabel((['fallen', 'tuck', 'situp', 'crouch', 'rise'][this.phaseIndex] as RecoveryPhaseId))}」阶段被接触阻挡超过 ${PAUSE_LIMIT.toFixed(1)} s，判定不可恢复`,
        );
        return;
      }
      if (this.status !== 'paused') this.status = 'paused';
      // Keep gentle drive while paused so the chain doesn't collapse, but do
      // not advance phase time: the get-up freezes against the obstacle.
      this.phaseTime = Math.max(0, this.phaseTime - this.handoff.world.timestep);
      return;
    }
    if (this.status === 'paused') this.status = 'getting-up';

    this.phaseTime += this.handoff.world.timestep;
    const reached = deviation.maxPos < POS_TOLERANCE && deviation.maxRotDeg < ROT_TOLERANCE_DEG;
    if (reached || this.phaseTime >= PHASE_DURATION) {
      this.phaseIndex += 1;
      this.phaseTime = 0;
      if (this.phaseIndex >= frames.length) {
        this.status = 'blending';
        this.blendTime = 0;
        this.settleSteps = 0;
        this.blendFrom = after;
        this.blendTo = this.standingTargetPose(after);
      }
    }
    void before;
  }

  /**
   * Final standing pose authored through the rig FK at the body's current
   * horizontal position. The root height comes from the support surface under
   * that position, so the blend never pulls the body toward a distant clip root.
   */
  private standingTargetPose(current: Pose): Pose {
    const basis = this.plan!.basis;
    const frames = PHASE_KEYFRAMES[basis.strategy];
    const upright = frames[frames.length - 1];
    const rootXY = { ...current[ROOT_ID].pos };
    return buildKeyframePose(this.scenario, upright, basis.yaw, rootXY, basis.supportNormal);
  }

  private stepBlend(_clipPose: Pose): void {
    if (!this.blendFrom || !this.blendTo) {
      this.finish();
      return;
    }
    this.blendTime += this.handoff.world.timestep;
    const u = clamp01(this.blendTime / BLEND_DURATION);
    const target = this.lerpPose(this.blendFrom, this.blendTo, u);
    // Ramp to full drive weight and hold it for the last few steps so the
    // final dynamic frame settles on the standing target; contacts still
    // constrain every intermediate step and can stall the blend if blocked.
    const weight = Math.min(1, 0.7 + 0.8 * u);
    this.handoff.driveTowardPose(target, DRIVE_GAINS, weight);
    this.handoff.stepWorldOnly();
    if (u >= 1) {
      // Hold the standing target briefly so the pelvis settles before control
      // is handed back, instead of flipping ownership at the top of a transient.
      this.handoff.driveTowardPose(this.blendTo, DRIVE_GAINS, 1);
      this.handoff.stepWorldOnly();
      this.settleSteps += 1;
      if (this.settleSteps >= BLEND_SETTLE_STEPS) this.finish();
    }
  }

  private finish(): void {
    this.status = 'complete';
    const cb = this.onComplete;
    this.onComplete = null;
    cb?.();
  }

  private lerpPose(a: Pose, b: Pose, t: number): Pose {
    const out = {} as Pose;
    for (const id of BONE_IDS) {
      out[id] = {
        pos: {
          x: a[id].pos.x + (b[id].pos.x - a[id].pos.x) * t,
          y: a[id].pos.y + (b[id].pos.y - a[id].pos.y) * t,
          z: a[id].pos.z + (b[id].pos.z - a[id].pos.z) * t,
        },
        rot: this.handoff.slerpRotation(a[id].rot, b[id].rot, t),
      };
    }
    return out;
  }

  /** Author the current phase target through the real FK hierarchy. */
  private phaseTarget(frames: Keyframe[]): Pose {
    const basis = this.plan!.basis;
    const i = Math.min(this.phaseIndex, frames.length - 1);
    const kf: Keyframe =
      i === 0
        ? frames[0]
        : interpolateKeyframes(frames[i - 1], frames[i], clamp01(this.phaseTime / PHASE_DURATION));
    const clamped: Keyframe = { ...kf, locals: clampLocalsToLimits(kf.locals) };
    let pose = buildKeyframePose(this.scenario, clamped, basis.yaw, basis.rootTarget, basis.supportNormal);
    // Never embed the target into the terrain: lift the whole authored pose if
    // any shape would penetrate, keeping the hierarchy intact.
    const penetration = poseGroundPenetration(this.scenario, pose);
    if (penetration < 0) {
      const lifted = {} as Pose;
      for (const id of BONE_IDS) {
        lifted[id] = { pos: { ...pose[id].pos, y: pose[id].pos.y - penetration + 0.002 }, rot: pose[id].rot };
      }
      pose = lifted;
    }
    return pose;
  }

  private describe(frameCount: number): string {
    if (!this.plan) return '尚未开始接触感知恢复。';
    const b = this.plan.basis;
    const head = Number.isFinite(b.headClearance) ? `${b.headClearance.toFixed(2)} m` : '充足';
    const base = `依据：${b.orientation} → ${b.strategy}；支撑法线 ${b.supportNormal.y.toFixed(2)}；头顶净空 ${head}；根就地推导，水平位移 ${b.horizontalTravel.toFixed(2)} m。`;
    if (this.status === 'failed') return `${base} 不可恢复：${this.plan.blockingReasons.join('；')}`;
    if (this.status === 'paused') return `${base} 阶段「${phaseLabel((['fallen', 'tuck', 'situp', 'crouch', 'rise'][this.phaseIndex] as RecoveryPhaseId))}」被接触阻挡，已暂停 ${this.blockedFor.toFixed(1)} s。`;
    if (this.status === 'blending') return `${base} 正在把控制平滑交回动画（${(clamp01(this.blendTime / BLEND_DURATION) * 100).toFixed(0)}%）。`;
    if (this.status === 'complete') return `${base} 起身完成，控制已交回动画。`;
    if (this.status === 'getting-up') {
      const phase = ['fallen', 'tuck', 'situp', 'crouch', 'rise'][Math.min(this.phaseIndex, frameCount - 1)] as RecoveryPhaseId;
      return `${base} 阶段「${phaseLabel(phase)}」 ${(clamp01(this.phaseTime / PHASE_DURATION) * 100).toFixed(0)}%。`;
    }
    return base;
  }
}

const clamp01 = (x: number): number => (x < 0 ? 0 : x > 1 ? 1 : x);

export { PHASE_DURATION, BLEND_DURATION };
