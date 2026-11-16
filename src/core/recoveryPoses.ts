import {
  Quat,
  Vec3,
  clamp,
  qfromAxisAngle,
  qmul,
  qnorm,
  qrot,
  smoothstep,
  vcross,
  vdot,
  vlen,
  vnorm,
  vscale,
  vsub,
} from './math';
import { BONES, BONE_IDS, BoneId, Pose, ROOT_ID, computePose, lowestPoint } from './rig';
import { SCENARIOS, ScenarioId } from './animator';

export type FallenOrientation = 'supine' | 'prone' | 'side' | 'upright';
export type GetupStrategy = 'supine-roll' | 'prone-push' | 'side-roll';
export type RecoveryPhaseId =
  | 'fallen'
  | 'tuck'
  | 'situp'
  | 'crouch'
  | 'rise'
  | 'blend'
  | 'paused'
  | 'failed'
  | 'done';

/** Local joint rotations authored per get-up phase. Everything else stays at bind. */
export interface Keyframe {
  /**
   * Root rotation relative to the chosen world yaw frame. The character faces
   * local +X, so the pitch that lays it down rotates about local Z (rotating
   * about X would leave the forward axis untouched); roll about local X handles
   * the side-lying case.
   */
  rootEuler: { yaw: number; pitch: number; roll: number };
  /** Root height above the local support surface (m). */
  rootHeight: number;
  locals: Partial<Record<BoneId, Quat>>;
}

const AX = (a: number): Quat => qfromAxisAngle({ x: 1, y: 0, z: 0 }, a);

/**
 * Compose the root rotation: world yaw (Y), then pitch about the body's local
 * Z (forward +X tilts up/down), then roll about local X (side lying).
 */
const rootRotation = (yaw: number, pitch: number, roll: number): Quat =>
  qnorm(
    qmul(
      qfromAxisAngle({ x: 0, y: 1, z: 0 }, yaw),
      qmul(
        qfromAxisAngle({ x: 0, y: 0, z: 1 }, pitch),
        qfromAxisAngle({ x: 1, y: 0, z: 0 }, roll),
      ),
    ),
  );

/**
 * Phase keyframes. Root pitch is measured from upright: -90deg is lying on the
 * back (supine), +90deg is face down (prone). Knee/elbow hinges use the same
 * local-X convention as the rig and stay inside the BONES joint limits.
 */
export const PHASE_KEYFRAMES: Record<GetupStrategy, Keyframe[]> = {
  'supine-roll': [
    {
      rootEuler: { yaw: 0, pitch: -Math.PI / 2, roll: 0 },
      rootHeight: 0.14,
      locals: {
        upperLegL: AX(0.15), upperLegR: AX(-0.15),
        lowerLegL: AX(-0.25), lowerLegR: AX(-0.25),
        upperArmL: AX(-0.5), upperArmR: AX(0.5),
        lowerArmL: AX(-0.9), lowerArmR: AX(-0.9),
      },
    },
    {
      // Tuck: knees and arms fold in, preparing the sit-up.
      rootEuler: { yaw: 0, pitch: -Math.PI / 2 + 0.35, roll: 0 },
      rootHeight: 0.16,
      locals: {
        upperLegL: AX(0.75), upperLegR: AX(0.55),
        lowerLegL: AX(-1.55), lowerLegR: AX(-1.45),
        upperArmL: AX(-1.05), upperArmR: AX(0.85),
        lowerArmL: AX(-1.7), lowerArmR: AX(-1.5),
      },
    },
    {
      // Sit-up: torso comes up over the folded legs, hands brace behind/side.
      rootEuler: { yaw: 0, pitch: -0.55, roll: 0 },
      rootHeight: 0.42,
      locals: {
        upperLegL: AX(1.15), upperLegR: AX(1.0),
        lowerLegL: AX(-2.05), lowerLegR: AX(-1.95),
        upperArmL: AX(-0.35), upperArmR: AX(0.25),
        lowerArmL: AX(-1.15), lowerArmR: AX(-1.05),
        spine: AX(0.25), chest: AX(0.2),
      },
    },
    {
      // Deep crouch: feet under the pelvis, arms forward for balance.
      rootEuler: { yaw: 0, pitch: 0.12, roll: 0 },
      rootHeight: 0.5,
      locals: {
        upperLegL: AX(1.05), upperLegR: AX(0.95),
        lowerLegL: AX(-2.15), lowerLegR: AX(-2.05),
        upperArmL: AX(0.55), upperArmR: AX(-0.45),
        lowerArmL: AX(-0.55), lowerArmR: AX(-0.45),
        spine: AX(0.12), chest: AX(0.08),
      },
    },
    {
      // Upright standing pose; the clip blend takes over from here.
      rootEuler: { yaw: 0, pitch: 0, roll: 0 },
      rootHeight: 0.94,
      locals: {
        upperLegL: AX(0.04), upperLegR: AX(-0.04),
        lowerLegL: AX(-0.08), lowerLegR: AX(-0.08),
        upperArmL: AX(0.08), upperArmR: AX(-0.08),
        lowerArmL: AX(-0.2), lowerArmR: AX(-0.2),
      },
    },
  ],
  'prone-push': [
    {
      rootEuler: { yaw: 0, pitch: Math.PI / 2, roll: 0 },
      rootHeight: 0.14,
      locals: {
        upperLegL: AX(-0.1), upperLegR: AX(0.1),
        lowerLegL: AX(-0.2), lowerLegR: AX(-0.2),
        upperArmL: AX(0.55), upperArmR: AX(-0.45),
        lowerArmL: AX(-1.0), lowerArmR: AX(-1.0),
      },
    },
    {
      // Push-up plank: arms extend, hips stay low.
      rootEuler: { yaw: 0, pitch: Math.PI / 2 - 0.5, roll: 0 },
      rootHeight: 0.28,
      locals: {
        upperLegL: AX(0.05), upperLegR: AX(-0.05),
        lowerLegL: AX(-0.35), lowerLegR: AX(-0.35),
        upperArmL: AX(1.25), upperArmR: AX(-1.15),
        lowerArmL: AX(-0.35), lowerArmR: AX(-0.35),
        spine: AX(-0.12), chest: AX(-0.1),
      },
    },
    {
      // Four-point crawl: hips rise, legs fold underneath.
      rootEuler: { yaw: 0, pitch: 0.75, roll: 0 },
      rootHeight: 0.46,
      locals: {
        upperLegL: AX(0.9), upperLegR: AX(0.8),
        lowerLegL: AX(-1.7), lowerLegR: AX(-1.6),
        upperArmL: AX(0.85), upperArmR: AX(-0.75),
        lowerArmL: AX(-0.7), lowerArmR: AX(-0.6),
        spine: AX(0.1),
      },
    },
    {
      rootEuler: { yaw: 0, pitch: 0.15, roll: 0 },
      rootHeight: 0.52,
      locals: {
        upperLegL: AX(1.0), upperLegR: AX(0.9),
        lowerLegL: AX(-2.1), lowerLegR: AX(-2.0),
        upperArmL: AX(0.5), upperArmR: AX(-0.4),
        lowerArmL: AX(-0.5), lowerArmR: AX(-0.4),
        spine: AX(0.1),
      },
    },
    {
      rootEuler: { yaw: 0, pitch: 0, roll: 0 },
      rootHeight: 0.94,
      locals: {
        upperLegL: AX(0.04), upperLegR: AX(-0.04),
        lowerLegL: AX(-0.08), lowerLegR: AX(-0.08),
        upperArmL: AX(0.08), upperArmR: AX(-0.08),
        lowerArmL: AX(-0.2), lowerArmR: AX(-0.2),
      },
    },
  ],
  'side-roll': [
    {
      rootEuler: { yaw: 0, pitch: 0, roll: Math.PI / 2 },
      rootHeight: 0.16,
      locals: {
        upperLegL: AX(0.4), upperLegR: AX(0.3),
        lowerLegL: AX(-0.9), lowerLegR: AX(-0.8),
        upperArmL: AX(-0.7), upperArmR: AX(0.6),
        lowerArmL: AX(-1.2), lowerArmR: AX(-1.1),
      },
    },
    {
      // Roll toward supine while tucking, then reuse the supine chain.
      rootEuler: { yaw: 0, pitch: -Math.PI / 2 + 0.3, roll: 0.25 },
      rootHeight: 0.17,
      locals: {
        upperLegL: AX(0.75), upperLegR: AX(0.55),
        lowerLegL: AX(-1.55), lowerLegR: AX(-1.45),
        upperArmL: AX(-1.05), upperArmR: AX(0.85),
        lowerArmL: AX(-1.7), lowerArmR: AX(-1.5),
      },
    },
    {
      rootEuler: { yaw: 0, pitch: -0.55, roll: 0 },
      rootHeight: 0.42,
      locals: {
        upperLegL: AX(1.15), upperLegR: AX(1.0),
        lowerLegL: AX(-2.05), lowerLegR: AX(-1.95),
        upperArmL: AX(-0.35), upperArmR: AX(0.25),
        lowerArmL: AX(-1.15), lowerArmR: AX(-1.05),
        spine: AX(0.25), chest: AX(0.2),
      },
    },
    {
      rootEuler: { yaw: 0, pitch: 0.12, roll: 0 },
      rootHeight: 0.5,
      locals: {
        upperLegL: AX(1.05), upperLegR: AX(0.95),
        lowerLegL: AX(-2.15), lowerLegR: AX(-2.05),
        upperArmL: AX(0.55), upperArmR: AX(-0.45),
        lowerArmL: AX(-0.55), lowerArmR: AX(-0.45),
        spine: AX(0.12), chest: AX(0.08),
      },
    },
    {
      rootEuler: { yaw: 0, pitch: 0, roll: 0 },
      rootHeight: 0.94,
      locals: {
        upperLegL: AX(0.04), upperLegR: AX(-0.04),
        lowerLegL: AX(-0.08), lowerLegR: AX(-0.08),
        upperArmL: AX(0.08), upperArmR: AX(-0.08),
        lowerArmL: AX(-0.2), lowerArmR: AX(-0.2),
      },
    },
  ],
};

/** Ordered phase ids matching the keyframe arrays above. */
export const STRATEGY_PHASES: Record<GetupStrategy, RecoveryPhaseId[]> = {
  'supine-roll': ['fallen', 'tuck', 'situp', 'crouch', 'rise'],
  'prone-push': ['fallen', 'tuck', 'situp', 'crouch', 'rise'],
  'side-roll': ['fallen', 'tuck', 'situp', 'crouch', 'rise'],
};

/**
 * Interpolate between two keyframes. Euler components are interpolated on the
 * shortest wrapped path so pitch crosses -90deg -> 0 without spinning.
 */
export const interpolateKeyframes = (a: Keyframe, b: Keyframe, u: number): Keyframe => {
  const t = smoothstep(u);
  const wrap = (x: number): number => {
    while (x > Math.PI) x -= Math.PI * 2;
    while (x < -Math.PI) x += Math.PI * 2;
    return x;
  };
  const locals = {} as Partial<Record<BoneId, Quat>>;
  const ids = new Set<BoneId>([...(Object.keys(a.locals) as BoneId[]), ...(Object.keys(b.locals) as BoneId[])]);
  for (const id of ids) {
    const qa = a.locals[id] ?? { x: 0, y: 0, z: 0, w: 1 };
    const qb = b.locals[id] ?? { x: 0, y: 0, z: 0, w: 1 };
    let cos = vdot(qa, qb);
    const qb2 = cos < 0 ? { x: -qb.x, y: -qb.y, z: -qb.z, w: -qb.w } : qb;
    cos = Math.abs(cos);
    if (cos > 0.9995) {
      locals[id] = qnorm({
        x: qa.x + (qb2.x - qa.x) * t,
        y: qa.y + (qb2.y - qa.y) * t,
        z: qa.z + (qb2.z - qa.z) * t,
        w: qa.w + (qb2.w - qa.w) * t,
      });
    } else {
      const theta = Math.acos(clamp(cos, -1, 1));
      const sin = Math.sin(theta);
      const wa = Math.sin((1 - t) * theta) / sin;
      const wb = Math.sin(t * theta) / sin;
      locals[id] = qnorm({
        x: qa.x * wa + qb2.x * wb,
        y: qa.y * wa + qb2.y * wb,
        z: qa.z * wa + qb2.z * wb,
        w: qa.w * wa + qb2.w * wb,
      });
    }
  }
  return {
    rootEuler: {
      yaw: a.rootEuler.yaw + wrap(b.rootEuler.yaw - a.rootEuler.yaw) * t,
      pitch: a.rootEuler.pitch + wrap(b.rootEuler.pitch - a.rootEuler.pitch) * t,
      roll: a.rootEuler.roll + wrap(b.rootEuler.roll - a.rootEuler.roll) * t,
    },
    rootHeight: a.rootHeight + (b.rootHeight - a.rootHeight) * t,
    locals,
  };
};

/**
 * Build a world-space get-up pose for a keyframe. The yaw frame and support
 * height are derived from the actual body/scene, never from a preset point.
 */
export const buildKeyframePose = (
  scenario: ScenarioId,
  kf: Keyframe,
  yaw: number,
  rootXY: Vec3,
  supportNormal: Vec3,
): Pose => {
  const sc = SCENARIOS[scenario];
  const groundY = sc.groundHeight(rootXY.x, rootXY.z);
  // On slopes the whole keyframe tilts about the body's local Z so feet and
  // hands approach the inclined surface instead of driving through it.
  const slopePitch = Math.atan2(supportNormal.x, supportNormal.y) * -1;
  const rootRot = rootRotation(yaw + kf.rootEuler.yaw, kf.rootEuler.pitch + slopePitch, kf.rootEuler.roll);
  const rootPos: Vec3 = { x: rootXY.x, y: groundY + kf.rootHeight, z: rootXY.z };
  return computePose(rootPos, rootRot, kf.locals);
};

/**
 * Lowest point of the authored pose relative to the terrain: the recovery
 * controller uses this to keep the target from embedding into the ground and
 * to detect when a keyframe needs more room than the scene provides.
 */
export const poseGroundPenetration = (scenario: ScenarioId, pose: Pose): number => {
  const sc = SCENARIOS[scenario];
  let worst = 0;
  for (const id of BONE_IDS) {
    const low = lowestPoint(pose, id);
    const ground = sc.groundHeight(pose[id].pos.x, pose[id].pos.z);
    worst = Math.min(worst, low - ground);
  }
  return worst;
};

/** Clamp every authored local rotation back into the rig's revolute limits. */
export const clampLocalsToLimits = (locals: Partial<Record<BoneId, Quat>>): Partial<Record<BoneId, Quat>> => {
  const out: Partial<Record<BoneId, Quat>> = { ...locals };
  for (const id of BONE_IDS) {
    const def = BONES[id];
    if (def.joint !== 'revolute' || !def.jointLimits || !out[id]) continue;
    const q = out[id]!;
    // Revolute joints in this rig rotate only about local X.
    const angle = 2 * Math.atan2(q.x, q.w);
    const clamped = clamp(angle, def.jointLimits[0], def.jointLimits[1]);
    out[id] = AX(clamped);
  }
  return out;
};

/**
 * Yaw of the fallen pelvis: direction the character's chest faces projected
 * onto the support plane. Used both for orientation classification and to
 * keep the get-up aligned with the actual facing.
 */
export const facingYawOnPlane = (pelvisRot: Quat, normal: Vec3): number => {
  // A horizontal axis is only usable for facing when it keeps most of its
  // length after the normal component is removed. A body lying flat still has
  // a tiny planar residual (~0.08) from settling, so the degenerate threshold
  // can't be near zero or the facing flips toward that noise.
  const DEGENERATE = 0.35;
  const project = (local: Vec3): Vec3 => {
    const world = qrot(pelvisRot, local);
    const planar = vsub(world, vscale(normal, vdot(world, normal)));
    return vlen(planar) > DEGENERATE ? vnorm(planar) : { x: 0, y: 0, z: 0 };
  };
  const f = project({ x: 1, y: 0, z: 0 });
  if (vlen(f) < 1e-6) {
    const side = project({ x: 0, y: 0, z: 1 });
    if (vlen(side) < 1e-6) return 0;
    return Math.atan2(-side.z, side.x) + Math.PI / 2;
  }
  return Math.atan2(-f.z, f.x);
};

/**
 * Classify how the body lies from the pelvis orientation against the support
 * normal. At full pitch (lying flat) the spine axis is horizontal, so the
 * chest-facing axis (+X) projected on the normal decides supine vs prone;
 * only a genuinely sideways roll (both axes horizontal) is "side".
 */
export const classifyOrientation = (pelvisRot: Quat, normal: Vec3): FallenOrientation => {
  const up = qrot(pelvisRot, { x: 0, y: 1, z: 0 });
  const spine = vdot(up, normal);
  if (spine > 0.55) return 'upright';
  const fwd = qrot(pelvisRot, { x: 1, y: 0, z: 0 });
  const face = vdot(fwd, normal);
  if (face < -0.45) return 'supine';
  if (face > 0.45) return 'prone';
  return 'side';
};

export const strategyFor = (orientation: FallenOrientation): GetupStrategy => {
  switch (orientation) {
    case 'prone':
      return 'prone-push';
    case 'side':
      return 'side-roll';
    default:
      return 'supine-roll';
  }
};

/**
 * Deterministic fallen pose used by tests and the guided-fall drill entry.
 * The body is laid on the support surface at the requested facing; ground
 * clamping is applied so contacts exist on the first solver step.
 */
export const buildFallenPose = (
  scenario: ScenarioId,
  orientation: Exclude<FallenOrientation, 'upright'>,
  rootXY: Vec3,
  yaw: number,
): Pose => {
  const kf0 = PHASE_KEYFRAMES[strategyFor(orientation)][0];
  const sc = SCENARIOS[scenario];
  const normal = sc.groundNormal(rootXY.x, rootXY.z);
  let pose = buildKeyframePose(scenario, kf0, yaw, rootXY, normal);
  // Drop the whole rig until its lowest shape rests on the terrain.
  const penetration = poseGroundPenetration(scenario, pose);
  if (penetration < 0) {
    const lifted = {} as Pose;
    for (const id of BONE_IDS) {
      lifted[id] = { pos: { ...pose[id].pos, y: pose[id].pos.y - penetration + 0.004 }, rot: pose[id].rot };
    }
    pose = lifted;
  }
  return pose;
};

/** Horizontal direction from one point to another, normalised. */
export const horizontalDir = (from: Vec3, to: Vec3): Vec3 => {
  const d = vsub(to, from);
  d.y = 0;
  return vlen(d) > 1e-9 ? vnorm(d) : { x: 1, y: 0, z: 0 };
};

/** Tangent on the support plane perpendicular to a horizontal direction. */
export const planeTangent = (dir: Vec3, normal: Vec3): Vec3 =>
  vnorm(vcross(normal, dir));

export { ROOT_ID };
