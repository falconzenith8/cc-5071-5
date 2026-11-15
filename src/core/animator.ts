import {
  Quat,
  Vec3,
  clamp,
  qAngularVelocity,
  qfromAxisAngle,
  qmul,
  qnorm,
  smoothstep,
  vadd,
  vdist,
  vlen,
  vnorm,
  vscale,
  vsub,
} from './math';
import {
  BoneId,
  FOOT_CLEARANCE,
  PELVIS_HEIGHT,
  Pose,
  ROOT_ID,
  computePose,
  footCornersWorld,
} from './rig';

export type ScenarioId = 'walk' | 'turn' | 'slope';

export interface Scenario {
  id: ScenarioId;
  label: string;
  description: string;
  groundHeight: (x: number, z: number) => number;
  groundNormal: (x: number, z: number) => Vec3;
  handoffTime: number;
  suggestedSpeed: number;
  start: Vec3;
  camera: { pos: Vec3; target: Vec3 };
}

export const SLOPE_ANGLE = (20 * Math.PI) / 180;
export const SLOPE_TOE_X = 1.1;

/** Shared analytic terrain. The renderer and the physics trimesh both sample this. */
export const slopeHeight = (x: number): number => {
  const d = x - SLOPE_TOE_X;
  if (d <= 0) return 0;
  const toe = smoothstep(clamp(d / 0.6, 0, 1));
  return Math.tan(SLOPE_ANGLE) * d * toe;
};

export const groundHeightFor = (scenario: ScenarioId, x: number, _z: number): number =>
  scenario === 'slope' ? slopeHeight(x) : 0;

export const groundNormalFor = (scenario: ScenarioId, x: number, _z: number): Vec3 => {
  if (scenario !== 'slope' || x <= SLOPE_TOE_X) return { x: 0, y: 1, z: 0 };
  const h0 = slopeHeight(x - 1e-3);
  const h1 = slopeHeight(x + 1e-3);
  return vnorm({ x: -(h1 - h0) / 2e-3, y: 1, z: 0 });
};

export const SCENARIOS: Record<ScenarioId, Scenario> = {
  walk: {
    id: 'walk', label: '行走中',
    description: '沿 +X 匀速行走，在左脚支撑相中触发受击。',
    groundHeight: (x, z) => groundHeightFor('walk', x, z),
    groundNormal: (x, z) => groundNormalFor('walk', x, z),
    handoffTime: 0.86,
    suggestedSpeed: 1.45,
    start: { x: 0, y: 0, z: 0 },
    camera: { pos: { x: -1.15, y: 1.55, z: 2.35 }, target: { x: 1.05, y: 0.9, z: 0 } },
  },
  turn: {
    id: 'turn', label: '转身中',
    description: '行走中左转 120°，在角速度峰值附近触发受击，检查角动量继承。',
    groundHeight: (x, z) => groundHeightFor('turn', x, z),
    groundNormal: (x, z) => groundNormalFor('turn', x, z),
    handoffTime: 1.15,
    suggestedSpeed: 1.15,
    start: { x: 0, y: 0, z: 0 },
    camera: { pos: { x: -1.0, y: 1.6, z: 2.5 }, target: { x: 1.0, y: 0.9, z: -0.35 } },
  },
  slope: {
    id: 'slope', label: '斜面上',
    description: '沿 20° 斜坡上行，在坡中触发受击，检查法向姿态与脚部接触。',
    groundHeight: (x, z) => groundHeightFor('slope', x, z),
    groundNormal: (x, z) => groundNormalFor('slope', x, z),
    handoffTime: 1.05,
    suggestedSpeed: 1.2,
    start: { x: 0.35, y: 0, z: 0 },
    camera: { pos: { x: -0.7, y: 1.9, z: 2.6 }, target: { x: 1.75, y: 1.15, z: 0 } },
  },
};

export interface AnimSample {
  pose: Pose;
  rootLinVel: Vec3;
  rootAngVel: Vec3;
  boneLinVel: Record<BoneId, Vec3>;
  boneAngVel: Record<BoneId, Vec3>;
  animTime: number;
  footContact: { L: boolean; R: boolean };
  yaw: number;
  yawRate: number;
}

const WALK_CYCLE = 1.05;
const CLIP_DURATION = 5.0;
const TABLE_DT = 1 / 240;
const TURN_WINDOW = 2.2;

interface PathSample {
  x: number;
  z: number;
  yaw: number;
}

/** Deterministic procedural gait: no external assets, so every run is exactly replayable. */
export class Animator {
  scenario: ScenarioId;
  speed: number;
  private path: PathSample[] = [];

  constructor(scenario: ScenarioId, speed?: number) {
    this.scenario = scenario;
    this.speed = speed ?? SCENARIOS[scenario].suggestedSpeed;
    this.buildPath();
  }

  configure(scenario: ScenarioId, speed = this.speed): void {
    this.scenario = scenario;
    this.speed = speed;
    this.buildPath();
  }

  private buildPath(): void {
    const start = SCENARIOS[this.scenario].start;
    const n = Math.ceil(CLIP_DURATION / TABLE_DT) + 1;
    const path: PathSample[] = [];
    let x = start.x;
    let z = start.z;
    for (let i = 0; i < n; i++) {
      const t = i * TABLE_DT;
      const yaw = this.scenario === 'turn'
        ? smoothstep(clamp(t / TURN_WINDOW, 0, 1)) * ((120 * Math.PI) / 180)
        : 0;
      path.push({ x, z, yaw });
      // Heading: +X rotates toward -Z for a left turn.
      const step = this.speed * TABLE_DT;
      x += Math.cos(yaw) * step;
      z -= Math.sin(yaw) * step;
    }
    this.path = path;
  }

  private pathAt(t: number): PathSample {
    const f = clamp(t / TABLE_DT, 0, this.path.length - 1);
    const i = Math.floor(f);
    const j = Math.min(i + 1, this.path.length - 1);
    const u = f - i;
    const a = this.path[i];
    const b = this.path[j];
    return { x: a.x + (b.x - a.x) * u, z: a.z + (b.z - a.z) * u, yaw: a.yaw + (b.yaw - a.yaw) * u };
  }

  yawRateAt(t: number, h = 1 / 240): number {
    return (this.pathAt(t + h).yaw - this.pathAt(t - h).yaw) / (2 * h);
  }

  /** Pose + finite-difference world velocities. Pure function of time. */
  sample(t: number, h = 1 / 240): AnimSample {
    const now = this.poseAt(t);
    const before = this.poseAt(t - h);
    const after = this.poseAt(t + h);
    const boneLinVel = {} as Record<BoneId, Vec3>;
    const boneAngVel = {} as Record<BoneId, Vec3>;
    for (const id of Object.keys(now.pose) as BoneId[]) {
      boneLinVel[id] = vscale(vsub(after.pose[id].pos, before.pose[id].pos), 1 / (2 * h));
      boneAngVel[id] = qAngularVelocity(before.pose[id].rot, after.pose[id].rot, 2 * h);
    }
    return {
      pose: now.pose,
      rootLinVel: boneLinVel[ROOT_ID],
      rootAngVel: boneAngVel[ROOT_ID],
      boneLinVel,
      boneAngVel,
      animTime: t,
      footContact: this.footContact(now.pose),
      yaw: now.yaw,
      yawRate: this.yawRateAt(t),
    };
  }

  private footContact(pose: Pose): { L: boolean; R: boolean } {
    const touching = (id: BoneId): boolean =>
      footCornersWorld(pose, id).some(
        (c) => c.y - SCENARIOS[this.scenario].groundHeight(c.x, c.z) < FOOT_CLEARANCE * 3,
      );
    return { L: touching('footL'), R: touching('footR') };
  }

  /** Pure pose function used by both rendering and the velocity estimate. */
  poseAt(t: number): { pose: Pose; yaw: number; rootPos: Vec3 } {
    const phase = (t / WALK_CYCLE) * Math.PI * 2;
    const swing = Math.sin(phase) * 0.52;
    const kneeL = 0.16 + 0.62 * Math.max(0, -Math.cos(phase)) + 0.2 * Math.abs(Math.sin(phase));
    const kneeR = 0.16 + 0.62 * Math.max(0, Math.cos(phase)) + 0.2 * Math.abs(Math.sin(phase));

    const p = this.pathAt(t);
    const sc = SCENARIOS[this.scenario];
    const gy = sc.groundHeight(p.x, p.z);
    const normal = sc.groundNormal(p.x, p.z);
    const bob = 0.022 * Math.cos(phase * 2);
    const rootPos = vadd(
      { x: p.x, y: gy + PELVIS_HEIGHT + bob, z: p.z },
      { x: 0, y: 0, z: 0 },
    );

    const yawQuat = qfromAxisAngle({ x: 0, y: 1, z: 0 }, p.yaw);
    // Lean the rig toward the terrain normal so the animated pose already matches the slope.
    const lean = this.scenario === 'slope' ? Math.atan2(normal.x, normal.y) * -1 : 0;
    const rootRot = qnorm(qmul(yawQuat, qfromAxisAngle({ x: 0, y: 0, z: 1 }, lean)));

    const footPitch = this.scenario === 'slope' ? -SLOPE_ANGLE : 0;
    const local: Partial<Record<BoneId, Quat>> = {
      upperLegL: qfromAxisAngle({ x: 1, y: 0, z: 0 }, swing),
      upperLegR: qfromAxisAngle({ x: 1, y: 0, z: 0 }, -swing),
      lowerLegL: qfromAxisAngle({ x: 1, y: 0, z: 0 }, -kneeL),
      lowerLegR: qfromAxisAngle({ x: 1, y: 0, z: 0 }, -kneeR),
      upperArmL: qfromAxisAngle({ x: 1, y: 0, z: 0 }, -swing * 0.8),
      upperArmR: qfromAxisAngle({ x: 1, y: 0, z: 0 }, swing * 0.8),
      lowerArmL: qfromAxisAngle({ x: 1, y: 0, z: 0 }, -0.32 - 0.16 * Math.sin(phase)),
      lowerArmR: qfromAxisAngle({ x: 1, y: 0, z: 0 }, -0.32 + 0.16 * Math.sin(phase)),
      spine: qfromAxisAngle({ x: 1, y: 0, z: 0 }, -0.035),
      chest: qfromAxisAngle({ x: 1, y: 0, z: 0 }, -0.03),
      neck: qfromAxisAngle({ x: 1, y: 0, z: 0 }, 0.05),
      head: qfromAxisAngle({ x: 1, y: 0, z: 0 }, 0.02),
      footL: qfromAxisAngle({ x: 1, y: 0, z: 0 }, footPitch),
      footR: qfromAxisAngle({ x: 1, y: 0, z: 0 }, footPitch),
    };

    let pose = computePose(rootPos, rootRot, local);
    pose = this.plantFeet(pose);
    return { pose, yaw: p.yaw, rootPos };
  }

  /**
   * Lower or raise the rig so the lowest foot corner sits exactly at the ground
   * clearance. This is the animator's foot IK substitute: the support foot is
   * always planted, which makes contact state deterministic at the handoff.
   */
  private plantFeet(pose: Pose): Pose {
    let minClear = Infinity;
    for (const id of ['footL', 'footR'] as BoneId[]) {
      for (const c of footCornersWorld(pose, id)) {
        minClear = Math.min(minClear, c.y - SCENARIOS[this.scenario].groundHeight(c.x, c.z));
      }
    }
    if (!Number.isFinite(minClear)) return pose;
    const correction = FOOT_CLEARANCE - minClear;
    if (Math.abs(correction) < 1e-9) return pose;
    const out = {} as Pose;
    for (const id of Object.keys(pose) as BoneId[]) {
      out[id] = { pos: { ...pose[id].pos, y: pose[id].pos.y + correction }, rot: pose[id].rot };
    }
    return out;
  }
}

export const estimateRootSpeed = (a: Pose, b: Pose, dt: number): number => vdist(a[ROOT_ID].pos, b[ROOT_ID].pos) / dt;
export const vectorLength = (v: Vec3): number => vlen(v);
