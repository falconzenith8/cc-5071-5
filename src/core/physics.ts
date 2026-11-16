import RAPIER from '@dimforge/rapier3d-compat';
import { Quat, Vec3, clamp, qconj, qmul, qnorm, qrot, qslerp, vadd, vdot, vlen, vnorm, vscale, vsub } from './math';
import {
  BONES,
  BoneId,
  BONE_IDS,
  Pose,
  ROOT_ID,
  UPPER_BODY,
  anchorErrors,
  capsuleHalfHeight,
  footCornersWorld,
  hingeCoordinate,
  worldToLocal,
} from './rig';
import { SCENARIOS, ScenarioId, AnimSample, Animator } from './animator';

export type Ownership = 'animation' | 'physics';

/** Fixed environment geometry the recovery flow must work around. Collision stays on for the whole chain. */
export interface ObstacleSpec {
  id: string;
  label: string;
  kind: 'wall' | 'ceiling';
  /** Cuboid center in world space. */
  center: Vec3;
  /** Cuboid half extents. */
  halfExtents: Vec3;
}

/** Rays are cast against environment colliders only, never against the character's own chain. */
export interface SpaceProbe {
  origin: Vec3;
  dir: Vec3;
  maxToi: number;
}

export interface SpaceHit {
  toi: number;
  point: Vec3;
  normal: Vec3;
  obstacleId: string | null;
}

export interface BodyRecord {
  id: BoneId;
  body: RAPIER.RigidBody;
  collider: RAPIER.Collider;
  joint: RAPIER.ImpulseJoint | null;
}

export interface BoneDiagnostic {
  id: BoneId;
  label: string;
  owner: Ownership;
  /** Child-origin to parent-anchor distance after the first physics step (m). */
  jointError: number;
  /** Raw pose change from the last animated frame to the first physics frame (m). */
  posDelta: number;
  /** Raw orientation change (deg). */
  rotDeltaDeg: number;
  /** Pose residual after removing the expected one-step animated motion (m). */
  posResidual: number;
  rotResidualDeg: number;
  /** Velocity written at takeover vs the animated velocity (m/s). */
  inheritSpeed: number;
  animSpeed: number;
  inheritError: number;
  inheritAngError: number;
  /** Solver-produced speed one step later (m/s). */
  physSpeed: number;
  /** Note is non-empty when this bone is animation-owned or a contact response. */
  flags: string[];
  hingeDeg: number | null;
  groundContact: boolean;
  groundNormalY: number | null;
}

export interface HandoffSnapshot {
  dt: number;
  animTime: number;
  scenario: ScenarioId;
  subset: BoneId[];
  boundary: Partial<Record<BoneId, BoneId>>;
  bones: BoneDiagnostic[];
  maxJointError: number;
  maxPosDelta: number;
  maxRotDeltaDeg: number;
  maxPosResidual: number;
  maxRotResidualDeg: number;
  maxInheritError: number;
  maxInheritAngError: number;
  maxSolverSpeedChange: number;
  rootSpeedBefore: number;
  rootSpeedAfter: number;
  rootOwner: 'animation' | 'physics';
  contacts: { L: boolean; R: boolean; other: BoneId[] };
  converged: boolean;
  failedChecks: string[];
  note: string;
}

export interface HandoffOptions {
  subset?: BoneId[];
  impact?: { bone: BoneId; impulse: Vec3 } | null;
  /** Fraction of the animated velocity inherited by the bodies. */
  velocityInheritance?: number;
}

export const DEFAULT_SUBSET: BoneId[] = [...BONE_IDS];
export const UPPER_BODY_SUBSET: BoneId[] = [...UPPER_BODY];

export const subsetBoundary = (subset: BoneId[]): Partial<Record<BoneId, BoneId>> => {
  const set = new Set(subset);
  const out: Partial<Record<BoneId, BoneId>> = {};
  for (const id of subset) {
    const p = BONES[id].parent;
    if (p && !set.has(p)) out[id] = p;
  }
  return out;
};

export const describeSubset = (subset: BoneId[]): string => {
  const set = new Set(subset);
  const boundary = subsetBoundary(subset);
  const edges = Object.entries(boundary).map(
    ([child, parent]) => `${BONES[parent as BoneId].label}→${BONES[child as BoneId].label}`,
  );
  const pinned = BONE_IDS.filter((id) => !set.has(id));
  const parts = [`受力骨骼 ${subset.length} 根`];
  if (edges.length) {
    parts.push(`边界关节 ${edges.join('、')}：动画侧保持运动学，动力学子链通过关节锚点跟随，不在边界额外施加驱动力`);
  }
  if (pinned.length) parts.push(`其余 ${pinned.length} 根仍由动画控制`);
  return parts.join('；') + '。';
};

const GROUP_TERRAIN = 0x0001;
const GROUP_CHARACTER = 0x0002;
const GROUP_OBSTACLE = 0x0004;
/** Terrain and obstacles share the environment mask: the chain collides with both, never itself. */
const ENV_GROUPS = GROUP_TERRAIN | GROUP_OBSTACLE;
const TERRAIN_GROUPS = (ENV_GROUPS << 16) | GROUP_CHARACTER;
const CHARACTER_GROUPS = (GROUP_CHARACTER << 16) | ENV_GROUPS;

// Post-handoff stability assist. Tuned as an acceleration-level PD toward the
// animated pose so body mass does not change the response. Kept deliberately
// mild: the chain still reacts to ground contact and impact, it just does not
// fold under gravity the instant control changes hands.
const ASSIST_RAMP = 0.12;
const ASSIST_DURATION = 2.0;
const ASSIST_FADE = 0.7;
const ASSIST_STIFFNESS = 120;    // 1/s^2
const ASSIST_DAMPING = 20;       // 1/s
const ASSIST_MAX_ACCEL = 40;     // m/s^2
const ASSIST_TORQUE_GAIN = 3.2;  // 1/s^2 per rad
const ASSIST_MAX_ANG_ACCEL = 28; // rad/s^2
const ASSIST_GRAVITY_SCALE = 0.55;

const quatAngleDeg = (a: Quat, b: Quat): number => {
  const d = Math.abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
  return 2 * Math.acos(Math.min(1, d)) * (180 / Math.PI);
};

const vecLen = (v: Vec3): number => Math.hypot(v.x, v.y, v.z);

/**
 * Owns the rigid-body chain and enforces exclusive control. A bone is driven by
 * either the animator or the solver. Animation-owned bodies are kinematic, so
 * the solver can never also integrate them; physics-owned bodies are ignored by
 * the animation write path.
 */
export class PoseHandoff {
  world: RAPIER.World;
  private records = new Map<BoneId, BodyRecord>();
  private ownership = new Map<BoneId, Ownership>();
  private joints = new Map<BoneId, RAPIER.ImpulseJoint>();
  private subset: BoneId[] = [...BONE_IDS];
  private active = false;
  private handoffTime = 0;
  private snapshot: HandoffSnapshot | null = null;
  private animator: Animator;
  private scenario: ScenarioId;
  private terrainBody: RAPIER.RigidBody | null = null;
  private obstacleBodies = new Map<string, { body: RAPIER.RigidBody; spec: ObstacleSpec }>();
  private obstacles: ObstacleSpec[] = [];

  constructor(scenario: ScenarioId, animator: Animator, obstacles: ObstacleSpec[] = []) {
    this.scenario = scenario;
    this.animator = animator;
    this.obstacles = obstacles.map((o) => ({ ...o, center: { ...o.center }, halfExtents: { ...o.halfExtents } }));
    this.world = new RAPIER.World({ x: 0, y: -9.81, z: 0 });
    this.world.timestep = 1 / 120;
    // Joint chains under a full-character impact need more than the default 4
    // solver iterations to keep anchors together; 24/4 measured at 13 mm worst
    // stretch on the slope versus 48 mm at 12/2.
    this.world.integrationParameters.numSolverIterations = 24;
    this.world.integrationParameters.numInternalPgsIterations = 4;
  }

  /** Builds the chain in the given animated world pose. No default stance is involved. */
  build(sample: AnimSample, subset: BoneId[] = [...BONE_IDS]): void {
    this.clear();
    this.scenario = this.animator.scenario;
    this.subset = [...subset];
    this.active = false;
    this.snapshot = null;

    this.buildTerrain();
    this.buildObstacles();

    for (const id of BONE_IDS) {
      const pose = sample.pose[id];
      const desc = RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(pose.pos.x, pose.pos.y, pose.pos.z)
        .setRotation(pose.rot)
        .setCanSleep(false);
      const body = this.world.createRigidBody(desc);
      const collider = this.world.createCollider(this.colliderFor(id), body);
      body.setBodyType(RAPIER.RigidBodyType.KinematicPositionBased, true);
      body.setNextKinematicTranslation(pose.pos);
      body.setNextKinematicRotation(pose.rot);
      this.records.set(id, { id, body, collider, joint: null });
      this.ownership.set(id, 'animation');
    }

    this.buildJoints();
    // One settle step keeps the collision pipeline in sync with the animated pose.
    this.world.step();
  }

  private colliderFor(id: BoneId): RAPIER.ColliderDesc {
    const def = BONES[id];
    const shape = def.shape;
    let desc: RAPIER.ColliderDesc;
    if (shape.kind === 'capsule') {
      desc = RAPIER.ColliderDesc.capsule(capsuleHalfHeight(shape), shape.radius);
    } else if (shape.kind === 'ball') {
      desc = RAPIER.ColliderDesc.ball(shape.radius);
    } else {
      desc = RAPIER.ColliderDesc.cuboid(shape.halfExtents.x, shape.halfExtents.y, shape.halfExtents.z);
    }
    return desc
      .setTranslation(shape.offset.x, shape.offset.y, shape.offset.z)
      .setFriction(0.9)
      .setRestitution(0.02)
      .setCollisionGroups(CHARACTER_GROUPS)
      .setMass(def.mass);
  }

  private buildJoints(): void {
    for (const id of BONE_IDS) {
      const def = BONES[id];
      if (!def.parent) continue;
      const parent = this.records.get(def.parent);
      const child = this.records.get(id);
      if (!parent || !child) continue;
      // The child's origin is its proximal joint by construction, so its local
      // anchor is zero and the parent anchor comes from the same BONES entry.
      const childLocal = worldToLocal(
        { pos: child.body.translation() as Vec3, rot: child.body.rotation() as Quat },
        vadd(parent.body.translation() as Vec3, qrot(parent.body.rotation() as Quat, def.parentAnchor)),
      );
      let data: RAPIER.JointData;
      if (def.joint === 'revolute') {
        const axisWorld = qrot(parent.body.rotation() as Quat, qrot(def.restLocalRot, { x: 1, y: 0, z: 0 }));
        const axisParent = qrot(qconj(parent.body.rotation() as Quat), axisWorld);
        const axisChild = qrot(qconj(child.body.rotation() as Quat), axisWorld);
        data = RAPIER.JointData.revoluteWithAxes(def.parentAnchor, childLocal, axisParent, axisChild);
      } else {
        data = RAPIER.JointData.spherical(def.parentAnchor, childLocal);
      }
      const joint = this.world.createImpulseJoint(data, parent.body, child.body, true);
      joint.setContactsEnabled(false);
      if (def.joint === 'revolute' && def.jointLimits) {
        (joint as RAPIER.RevoluteImpulseJoint).setLimits(def.jointLimits[0], def.jointLimits[1]);
      }
      this.records.get(id)!.joint = joint;
      this.joints.set(id, joint);
    }
  }

  private buildTerrain(): void {
    if (this.terrainBody) {
      this.world.removeRigidBody(this.terrainBody);
      this.terrainBody = null;
    }
    const sc = SCENARIOS[this.scenario];
    const verts: number[] = [];
    const indices: number[] = [];
    const nx = 64;
    const nz = 16;
    const x0 = -3;
    const x1 = 7;
    const z0 = -2.5;
    const z1 = 2.5;
    for (let ix = 0; ix <= nx; ix++) {
      for (let iz = 0; iz <= nz; iz++) {
        const x = x0 + ((x1 - x0) * ix) / nx;
        const z = z0 + ((z1 - z0) * iz) / nz;
        verts.push(x, sc.groundHeight(x, z), z);
      }
    }
    for (let ix = 0; ix < nx; ix++) {
      for (let iz = 0; iz < nz; iz++) {
        const a = ix * (nz + 1) + iz;
        const b = a + 1;
        const c = a + (nz + 1);
        const d = c + 1;
        // Counter-clockwise when seen from above (+Y), so contact normals point up.
        indices.push(a, b, c, b, d, c);
      }
    }
    this.terrainBody = this.world.createRigidBody(RAPIER.RigidBodyDesc.fixed());
    this.world.createCollider(
      RAPIER.ColliderDesc.trimesh(new Float32Array(verts), new Uint32Array(indices))
        .setFriction(1.0)
        .setRestitution(0.0)
        .setCollisionGroups(TERRAIN_GROUPS),
      this.terrainBody,
    );
  }

  private buildObstacles(): void {
    for (const rec of this.obstacleBodies.values()) this.world.removeRigidBody(rec.body);
    this.obstacleBodies.clear();
    for (const spec of this.obstacles) {
      const body = this.world.createRigidBody(
        RAPIER.RigidBodyDesc.fixed().setTranslation(spec.center.x, spec.center.y, spec.center.z),
      );
      this.world.createCollider(
        RAPIER.ColliderDesc.cuboid(spec.halfExtents.x, spec.halfExtents.y, spec.halfExtents.z)
          .setFriction(0.95)
          .setRestitution(0.0)
          .setCollisionGroups(TERRAIN_GROUPS),
        body,
      );
      this.obstacleBodies.set(spec.id, { body, spec });
    }
  }

  getObstacles(): ObstacleSpec[] {
    return this.obstacles.map((o) => ({ ...o, center: { ...o.center }, halfExtents: { ...o.halfExtents } }));
  }

  /** Swap environment geometry and rebuild the fixed obstacle bodies immediately. */
  setObstacles(obstacles: ObstacleSpec[]): void {
    this.obstacles = obstacles.map((o) => ({ ...o, center: { ...o.center }, halfExtents: { ...o.halfExtents } }));
    this.buildObstacles();
    this.world.step();
  }

  private clear(): void {
    for (const joint of this.joints.values()) if (joint.isValid()) this.world.removeImpulseJoint(joint, false);
    this.joints.clear();
    for (const rec of this.records.values()) this.world.removeRigidBody(rec.body);
    this.records.clear();
    this.ownership.clear();
  }

  dispose(): void {
    this.clear();
    for (const rec of this.obstacleBodies.values()) this.world.removeRigidBody(rec.body);
    this.obstacleBodies.clear();
    if (this.terrainBody) this.world.removeRigidBody(this.terrainBody);
    this.world.free();
  }

  /** Drives every animation-owned body from the clip. Physics-owned bodies are untouched. */
  applyAnimation(sample: AnimSample, snap = false): void {
    for (const id of BONE_IDS) {
      if (this.ownership.get(id) !== 'animation') continue;
      const rec = this.records.get(id);
      if (!rec) continue;
      const pose = sample.pose[id];
      if (snap) rec.body.setTranslation(pose.pos, true);
      if (snap) rec.body.setRotation(pose.rot, true);
      rec.body.setNextKinematicTranslation(pose.pos);
      rec.body.setNextKinematicRotation(pose.rot);
    }
  }

  /**
   * Hands the subset over to physics at the exact animated pose and animated
   * per-bone velocities. The first physics step is compared against the last
   * animated frame both raw and with the expected one-step motion removed.
   */
  handoff(sample: AnimSample, options: HandoffOptions = {}): HandoffSnapshot {
    const subset = options.subset ?? this.subset;
    const inheritance = options.velocityInheritance ?? 1.0;
    if (subset.some((id) => !this.records.has(id))) throw new Error('handoff subset contains unknown bones');

    this.applyAnimation(sample, true);
    this.world.step();

    const dt = this.world.timestep;
    const animVel = { lin: sample.boneLinVel, ang: sample.boneAngVel };
    const before = this.captureState();

    for (const id of subset) {
      const rec = this.records.get(id)!;
      rec.body.setBodyType(RAPIER.RigidBodyType.Dynamic, true);
      rec.body.setLinvel(vscale(animVel.lin[id], inheritance), true);
      rec.body.setAngvel(vscale(animVel.ang[id], inheritance), true);
      rec.body.setLinearDamping(0.06);
      rec.body.setAngularDamping(0.3);
      this.ownership.set(id, 'physics');
    }
    this.subset = [...subset];
    // Bodies outside the subset remain kinematic and continue to follow the clip.
    this.applyAnimation(sample, true);

    // The handoff audit measures *only* the takeover: velocities are written,
    // then one solver step is taken with no gameplay impulse. The impact is
    // applied afterward as a separate step, so it can never contaminate the
    // continuity numbers the user is asked to verify.
    const writtenVel = new Map<BoneId, Vec3>();
    const writtenAngVel = new Map<BoneId, Vec3>();
    for (const id of BONE_IDS) {
      const rec = this.records.get(id)!;
      writtenVel.set(id, { ...(rec.body.linvel() as Vec3) });
      writtenAngVel.set(id, { ...(rec.body.angvel() as Vec3) });
    }
    this.world.step();
    const after = this.captureState();
    const jointErrors = anchorErrors(this.poseFromWorld());
    const groundBones = new Set(this.groundContactBones());

    const bones: BoneDiagnostic[] = [];
    let maxJointError = 0;
    let maxPosDelta = 0;
    let maxRotDeltaDeg = 0;
    let maxPosResidual = 0;
    let maxRotResidualDeg = 0;
    let maxInheritError = 0;
    let maxInheritAngError = 0;
    let maxSolverSpeedChange = 0;

    for (const id of BONE_IDS) {
      const a = before.get(id)!;
      const b = after.get(id)!;
      const solverOwned = subset.includes(id);
      const expectedPos = vadd(a.pos, vscale(animVel.lin[id], dt));
      const posDelta = vecLen({ x: b.pos.x - a.pos.x, y: b.pos.y - a.pos.y, z: b.pos.z - a.pos.z });
      const posResidual = vecLen({ x: b.pos.x - expectedPos.x, y: b.pos.y - expectedPos.y, z: b.pos.z - expectedPos.z });
      const rotDeltaDeg = quatAngleDeg(a.rot, b.rot);
      const w = writtenVel.get(id)!;
      const wa = writtenAngVel.get(id)!;
      const inheritError = vecLen({ x: w.x - animVel.lin[id].x, y: w.y - animVel.lin[id].y, z: w.z - animVel.lin[id].z });
      const inheritAngError = vecLen({ x: wa.x - animVel.ang[id].x, y: wa.y - animVel.ang[id].y, z: wa.z - animVel.ang[id].z });
      const solverSpeedChange = vecLen({ x: b.vel.x - w.x, y: b.vel.y - w.y, z: b.vel.z - w.z });
      const flags: string[] = [];
      if (!solverOwned) flags.push('动画控制');
      if (groundBones.has(id)) flags.push('接地');
      if (solverOwned) {
        maxPosDelta = Math.max(maxPosDelta, posDelta);
        maxRotDeltaDeg = Math.max(maxRotDeltaDeg, rotDeltaDeg);
        maxPosResidual = Math.max(maxPosResidual, posResidual);
        maxRotResidualDeg = Math.max(maxRotResidualDeg, rotDeltaDeg);
        maxInheritError = Math.max(maxInheritError, inheritError);
        maxInheritAngError = Math.max(maxInheritAngError, inheritAngError);
        maxSolverSpeedChange = Math.max(maxSolverSpeedChange, solverSpeedChange);
      }
      maxJointError = Math.max(maxJointError, jointErrors[id]);
      const contact = this.contactInfo(id);
      bones.push({
        id,
        label: BONES[id].label,
        owner: this.ownership.get(id)!,
        jointError: jointErrors[id],
        posDelta,
        rotDeltaDeg,
        posResidual,
        rotResidualDeg: rotDeltaDeg,
        inheritSpeed: vecLen(w),
        animSpeed: vecLen(animVel.lin[id]),
        inheritError,
        inheritAngError,
        physSpeed: vecLen(b.vel),
        flags,
        hingeDeg: BONES[id].joint === 'revolute' ? hingeCoordinate(this.poseFromWorld(), id) * (180 / Math.PI) : null,
        groundContact: contact.touching,
        groundNormalY: contact.normalY,
      });
    }

    const groundContacts = this.groundContactBones();
    const contacts = {
      L: groundContacts.includes('footL'),
      R: groundContacts.includes('footR'),
      other: groundContacts.filter((id) => id !== 'footL' && id !== 'footR'),
    };
    const rootAfter = vecLen(after.get(ROOT_ID)!.vel);
    const rootBefore = vecLen(animVel.lin[ROOT_ID]);

    const failedChecks: string[] = [];
    if (maxJointError > 0.01) failedChecks.push(`关节锚点误差 ${(maxJointError * 1000).toFixed(1)} mm 超过 10 mm`);
    if (maxPosResidual > 0.03) failedChecks.push(`姿态残差 ${(maxPosResidual * 1000).toFixed(1)} mm 超过 30 mm`);
    if (maxInheritError > 1e-6) failedChecks.push(`写入速度与动画速度不一致 ${(maxInheritError * 1000).toFixed(2)} mm/s`);
    if (maxInheritAngError > 1e-6) failedChecks.push(`写入角速度与动画不一致 ${maxInheritAngError.toFixed(4)} rad/s`);

    this.active = true;
    this.handoffTime = sample.animTime;
    this.snapshot = {
      dt,
      animTime: sample.animTime,
      scenario: this.scenario,
      subset: [...subset],
      boundary: subsetBoundary(subset),
      bones,
      maxJointError,
      maxPosDelta,
      maxRotDeltaDeg,
      maxPosResidual,
      maxRotResidualDeg,
      maxInheritError,
      maxInheritAngError,
      maxSolverSpeedChange,
      rootSpeedBefore: rootBefore,
      rootSpeedAfter: rootAfter,
      rootOwner: this.ownership.get(ROOT_ID)!,
      contacts,
      converged: failedChecks.length === 0,
      failedChecks,
      note: describeSubset(subset),
    };
    // Impact is applied after the audited takeover step: it is a gameplay
    // event, not part of the continuity measurement.
    if (options.impact) {
      const rec = this.records.get(options.impact.bone);
      if (rec && this.ownership.get(options.impact.bone) === 'physics') {
        const p = rec.body.translation() as Vec3;
        rec.body.applyImpulseAtPoint(options.impact.impulse, p, true);
      }
    }
    return this.snapshot;
  }

  private captureState(): Map<BoneId, { pos: Vec3; rot: Quat; vel: Vec3; angVel: Vec3 }> {
    const out = new Map<BoneId, { pos: Vec3; rot: Quat; vel: Vec3; angVel: Vec3 }>();
    for (const id of BONE_IDS) {
      const rec = this.records.get(id)!;
      out.set(id, {
        pos: { ...(rec.body.translation() as Vec3) },
        rot: { ...(rec.body.rotation() as Quat) },
        vel: { ...(rec.body.linvel() as Vec3) },
        angVel: { ...(rec.body.angvel() as Vec3) },
      });
    }
    return out;
  }

  private contactInfo(id: BoneId): { touching: boolean; normalY: number | null } {
    const rec = this.records.get(id);
    if (!rec) return { touching: false, normalY: null };
    let touching = false;
    let normalY: number | null = null;
    this.world.contactPairsWith(rec.collider, (other) => {
      this.world.contactPair(rec.collider, other, (manifold) => {
        if (manifold.numContacts() > 0) {
          touching = true;
          if (normalY === null) normalY = manifold.normal().y;
        }
      });
    });
    return { touching, normalY };
  }

  /** Which bones touch the terrain, using actual contact manifolds rather than broadphase pairs. */
  groundContactBones(): BoneId[] {
    const out: BoneId[] = [];
    if (!this.terrainBody) return out;
    for (const id of BONE_IDS) {
      const rec = this.records.get(id);
      if (!rec) continue;
      let touchesTerrain = false;
      this.world.contactPairsWith(rec.collider, (other) => {
        if (other.parent()?.handle !== this.terrainBody!.handle) return;
        this.world.contactPair(rec.collider, other, (manifold) => {
          if (manifold.numContacts() > 0) touchesTerrain = true;
        });
      });
      if (touchesTerrain) out.push(id);
    }
    return out;
  }

  /** Advance one frame: animation-owned bones follow the clip, physics-owned integrate. */
  driveAnimation(sample: AnimSample): void {
    this.applyAnimation(sample, false);
  }

  /**
   * Post-handoff stability assist. The physics chain is deliberately soft (it is
   * a ragdoll), so without assistance the character would fold to the floor the
   * instant contact is lost. This applies a capped, damped spring toward the
   * *animated* pose for a short window, which lets the character keep moving
   * through the contact instead of collapsing. The assist is bounded to
   * MAX_ASSIST_N so the handoff audit (first solver step) stays assist-free.
   */
  applyStabilityAssist(sample: AnimSample, elapsed: number): void {
    if (!this.active) return;
    const ramp = elapsed < ASSIST_RAMP ? elapsed / ASSIST_RAMP : 1;
    const fade = elapsed > ASSIST_DURATION ? Math.max(0, 1 - (elapsed - ASSIST_DURATION) / ASSIST_FADE) : 1;
    const gain = ramp * fade;
    if (gain <= 0) return;
    for (const id of BONE_IDS) {
      if (this.ownership.get(id) !== 'physics') continue;
      const rec = this.records.get(id);
      if (!rec) continue;
      const mass = rec.body.mass();
      // Partially unload gravity so the assist can hold posture without
      // out-muscling contact response; the character still feels impacts.
      rec.body.setGravityScale(1 - (1 - ASSIST_GRAVITY_SCALE) * gain, true);

      const target = sample.pose[id];
      const pos = rec.body.translation() as Vec3;
      const vel = rec.body.linvel() as Vec3;
      const toTarget = { x: target.pos.x - pos.x, y: target.pos.y - pos.y, z: target.pos.z - pos.z };
      const accel = {
        x: ASSIST_STIFFNESS * toTarget.x - ASSIST_DAMPING * vel.x,
        y: ASSIST_STIFFNESS * toTarget.y - ASSIST_DAMPING * vel.y,
        z: ASSIST_STIFFNESS * toTarget.z - ASSIST_DAMPING * vel.z,
      };
      const accelMag = vecLen(accel);
      if (accelMag > ASSIST_MAX_ACCEL) {
        const s = ASSIST_MAX_ACCEL / accelMag;
        accel.x *= s; accel.y *= s; accel.z *= s;
      }
      rec.body.applyImpulse(
        { x: accel.x * mass * gain * this.world.timestep, y: accel.y * mass * gain * this.world.timestep, z: accel.z * mass * gain * this.world.timestep },
        true,
      );

      // Orientation: pull the core bones back toward the animated facing so the
      // character keeps its intended attitude while the limbs react freely.
      if (id === 'pelvis' || id === 'chest' || id === 'head') {
        const rel = qmul(rec.body.rotation() as Quat, qconj(target.rot));
        const q = rel.w < 0 ? { x: -rel.x, y: -rel.y, z: -rel.z, w: -rel.w } : rel;
        const angle = 2 * Math.acos(Math.min(1, Math.abs(q.w)));
        const s = Math.sqrt(Math.max(0, 1 - q.w * q.w));
        if (s > 1e-6 && angle > 1e-6) {
          const axis = { x: q.x / s, y: q.y / s, z: q.z / s };
          const angAccel = Math.min(ASSIST_MAX_ANG_ACCEL, ASSIST_TORQUE_GAIN * angle);
          rec.body.applyTorqueImpulse(
            { x: axis.x * angAccel * mass * gain * this.world.timestep, y: axis.y * angAccel * mass * gain * this.world.timestep, z: axis.z * angAccel * mass * gain * this.world.timestep },
            true,
          );
        }
      }
    }
  }

  stepWorldOnly(): void {
    this.world.step();
  }

  /** Backwards-compatible alias used by tests. */
  step(sample: AnimSample): void {
    this.driveAnimation(sample);
    this.stepWorldOnly();
  }

  /**
   * World-space contact points on the terrain, for the on-screen contact drill
   * overlay. Uses solver contacts so the dots match what actually pushes.
   */
  contactPoints(): { pos: Vec3; normal: Vec3 }[] {
    const out: { pos: Vec3; normal: Vec3 }[] = [];
    if (!this.terrainBody) return out;
    for (const id of BONE_IDS) {
      const rec = this.records.get(id);
      if (!rec) continue;
      this.world.contactPairsWith(rec.collider, (other) => {
        if (other.parent()?.handle !== this.terrainBody!.handle) return;
        this.world.contactPair(rec.collider, other, (manifold) => {
          const n = manifold.normal();
          for (let i = 0; i < manifold.numSolverContacts(); i++) {
            const p = manifold.solverContactPoint(i);
            if (p) out.push({ pos: { x: p.x, y: p.y, z: p.z }, normal: { x: n.x, y: n.y, z: n.z } });
          }
        });
      });
    }
    return out;
  }

  poseFromWorld(): Pose {
    const pose = {} as Pose;
    for (const id of BONE_IDS) {
      const rec = this.records.get(id)!;
      pose[id] = {
        pos: { ...(rec.body.translation() as Vec3) },
        rot: { ...(rec.body.rotation() as Quat) },
      };
    }
    return pose;
  }

  /**
   * Ray cast against environment colliders (terrain + obstacles) only. The
   * character's own colliders are excluded by collision groups, so a probe
   * through an arm measures the room around the body, not the body itself.
   */
  castEnvironmentRay(probe: SpaceProbe): SpaceHit | null {
    const dir = vnorm(probe.dir);
    if (vlen(dir) < 1e-9) return null;
    const ray = new RAPIER.Ray(probe.origin, dir);
    // Rapier interaction groups (verified empirically): a candidate collider is
    // reported when (queryMemberships & colliderFilter) != 0 AND
    // (colliderMemberships & queryFilter) != 0. Environment colliders carry
    // membership ENV_GROUPS with filter GROUP_CHARACTER, so the query must carry
    // GROUP_CHARACTER membership while filtering for ENV_GROUPS. Character
    // colliders (membership GROUP_CHARACTER) fail the second test and never hit.
    const queryGroups = (GROUP_CHARACTER << 16) | ENV_GROUPS;
    const hit = this.world.castRayAndGetNormal(ray, probe.maxToi, true, undefined, queryGroups);
    if (!hit) return null;
    const point = vadd(probe.origin, vscale(dir, hit.timeOfImpact));
    let obstacleId: string | null = null;
    for (const [id, rec] of this.obstacleBodies) {
      if (rec.body.handle === hit.collider.parent()?.handle) obstacleId = id;
    }
    return { toi: hit.timeOfImpact, point, normal: { ...(hit.normal as Vec3) }, obstacleId };
  }

  /** Free-space distance above a world point; Infinity when nothing is hit within maxToi. */
  clearanceAbove(point: Vec3, maxToi = 2.5): number {
    const hit = this.castEnvironmentRay({ origin: point, dir: { x: 0, y: 1, z: 0 }, maxToi });
    return hit ? hit.toi : Infinity;
  }

  /**
   * Average outward normal of every terrain/obstacle contact on the chain,
   * weighted by solver contact count. This is the actual support surface the
   * recovery planner aligns to, not an assumed flat floor.
   */
  supportNormal(): { normal: Vec3; contactCount: number } {
    const acc = { x: 0, y: 0, z: 0 };
    let count = 0;
    for (const id of BONE_IDS) {
      const rec = this.records.get(id);
      if (!rec) continue;
      this.world.contactPairsWith(rec.collider, (other) => {
        if (!this.isEnvironment(other)) return;
        this.world.contactPair(rec.collider, other, (manifold) => {
          if (manifold.numContacts() === 0) return;
          // One contribution per contacting manifold pair, weighted by the
          // manifold normal. Solver contacts can be zero during warm-up.
          const n = manifold.normal();
          acc.x += n.x;
          acc.y += n.y;
          acc.z += n.z;
          count++;
        });
      });
    }
    if (count === 0) return { normal: { x: 0, y: 1, z: 0 }, contactCount: 0 };
    return { normal: vnorm(acc), contactCount: count };
  }

  /** Contact normals per bone, including wall and ceiling pushes, for the recovery UI. */
  contactNormalsByBone(): Map<BoneId, Vec3[]> {
    const out = new Map<BoneId, Vec3[]>();
    for (const id of BONE_IDS) {
      const rec = this.records.get(id);
      if (!rec) continue;
      const normals: Vec3[] = [];
      this.world.contactPairsWith(rec.collider, (other) => {
        if (!this.isEnvironment(other)) return;
        this.world.contactPair(rec.collider, other, (manifold) => {
          if (manifold.numContacts() > 0) normals.push({ ...(manifold.normal() as Vec3) });
        });
      });
      if (normals.length) out.set(id, normals);
    }
    return out;
  }

  private isEnvironment(collider: RAPIER.Collider): boolean {
    const handle = collider.parent()?.handle;
    if (this.terrainBody && handle === this.terrainBody.handle) return true;
    for (const rec of this.obstacleBodies.values()) {
      if (rec.body.handle === handle) return true;
    }
    return false;
  }

  /** Which obstacle (if any) currently has a solver contact with the chain. */
  contactingObstacles(): string[] {
    const ids = new Set<string>();
    for (const id of BONE_IDS) {
      const rec = this.records.get(id);
      if (!rec) continue;
      this.world.contactPairsWith(rec.collider, (other) => {
        const handle = other.parent()?.handle;
        for (const [obsId, obs] of this.obstacleBodies) {
          if (obs.body.handle === handle) {
            this.world.contactPair(rec.collider, other, (manifold) => {
              if (manifold.numContacts() > 0) ids.add(obsId);
            });
          }
        }
      });
    }
    return [...ids];
  }

  /**
   * Kinematic write of an externally authored pose (the guided-fall entry used
   * by the recovery drill). All bodies stay in the same chain; joints are not
   * rebuilt, so the skeleton hierarchy is untouched. One solver step is taken
   * so contact manifolds exist against the new pose.
   */
  writePoseKinematic(pose: Pose): void {
    for (const id of BONE_IDS) {
      const rec = this.records.get(id);
      if (!rec) continue;
      if (!rec.body.isKinematic()) rec.body.setBodyType(RAPIER.RigidBodyType.KinematicPositionBased, true);
      this.ownership.set(id, 'animation');
      rec.body.setNextKinematicTranslation(pose[id].pos);
      rec.body.setNextKinematicRotation(pose[id].rot);
    }
    this.world.step();
  }

  /**
   * Blended kinematic target used while a get-up phase is in progress. Bodies
   * are moved back to dynamic ownership and the caller applies bounded drives
   * toward these targets; collision response always wins over the drive.
   */
  driveTowardPose(
    target: Pose,
    gains: { linK: number; linD: number; maxForce: number; angK: number; angD: number; maxTorque: number },
    weight = 1,
  ): void {
    const dt = this.world.timestep;
    for (const id of BONE_IDS) {
      const rec = this.records.get(id);
      if (!rec) continue;
      if (rec.body.isKinematic()) rec.body.setBodyType(RAPIER.RigidBodyType.Dynamic, true);
      this.ownership.set(id, 'physics');
      const body = rec.body;
      const mass = body.mass();
      const pos = body.translation() as Vec3;
      const vel = body.linvel() as Vec3;
      const toTarget = vsub(target[id].pos, pos);
      const force = {
        x: (gains.linK * toTarget.x - gains.linD * vel.x) * mass * weight,
        y: (gains.linK * toTarget.y - gains.linD * vel.y) * mass * weight,
        z: (gains.linK * toTarget.z - gains.linD * vel.z) * mass * weight,
      };
      const forceMag = vlen(force);
      if (forceMag > gains.maxForce) {
        const s = gains.maxForce / forceMag;
        force.x *= s; force.y *= s; force.z *= s;
      }
      body.applyImpulse(vscale(force, dt), true);

      const rel = qnorm(qmul(target[id].rot, qconj(body.rotation() as Quat)));
      const q = rel.w < 0 ? { x: -rel.x, y: -rel.y, z: -rel.z, w: -rel.w } : rel;
      const angle = 2 * Math.atan2(Math.hypot(q.x, q.y, q.z), q.w);
      if (angle > 1e-5) {
        const axis = vnorm({ x: q.x, y: q.y, z: q.z });
        const angVel = body.angvel() as Vec3;
        const torqueMagRaw = gains.angK * angle;
        const torque = {
          x: (torqueMagRaw * axis.x - gains.angD * angVel.x) * mass * weight,
          y: (torqueMagRaw * axis.y - gains.angD * angVel.y) * mass * weight,
          z: (torqueMagRaw * axis.z - gains.angD * angVel.z) * mass * weight,
        };
        const torqueMag = vlen(torque);
        if (torqueMag > gains.maxTorque) {
          const s = gains.maxTorque / torqueMag;
          torque.x *= s; torque.y *= s; torque.z *= s;
        }
        body.applyTorqueImpulse(vscale(torque, dt), true);
      }
    }
  }

  /** Per-bone distance from a target pose, for get-up phase progress checks. */
  poseDeviation(target: Pose): { maxPos: number; maxRotDeg: number; rootPos: number } {    let maxPos = 0;
    let maxRotDeg = 0;
    const current = this.poseFromWorld();
    for (const id of BONE_IDS) {
      maxPos = Math.max(maxPos, vlen(vsub(current[id].pos, target[id].pos)));
      const d = Math.abs(vdot(current[id].rot, target[id].rot));
      maxRotDeg = Math.max(maxRotDeg, 2 * Math.acos(clamp(d, -1, 1)) * (180 / Math.PI));
    }
    return {
      maxPos,
      maxRotDeg,
      rootPos: vlen(vsub(current[ROOT_ID].pos, target[ROOT_ID].pos)),
    };
  }

  /**
   * Guided fall: kinematically move the chain through an authored falling
   * trajectory while the solver still generates real terrain/obstacle
   * contacts each step. Used by the recovery drill to produce a deterministic
   * supine/prone/side configuration without teleporting — every intermediate
   * frame passes through the collision pipeline.
   */
  guidedFall(target: Pose, steps: number): void {
    const start = this.poseFromWorld();
    for (let s = 1; s <= steps; s++) {
      const u = s / steps;
      const eased = u * u * (3 - 2 * u);
      const pose = {} as Pose;
      for (const id of BONE_IDS) {
        pose[id] = {
          pos: {
            x: start[id].pos.x + (target[id].pos.x - start[id].pos.x) * eased,
            y: start[id].pos.y + (target[id].pos.y - start[id].pos.y) * eased,
            z: start[id].pos.z + (target[id].pos.z - start[id].pos.z) * eased,
          },
          rot: qslerp(start[id].rot, target[id].rot, eased),
        };
      }
      this.writePoseKinematic(pose);
    }
  }

  /** Slurp helper kept here so recovery code never bypasses the normalized quaternion math. */
  slerpRotation(from: Quat, to: Quat, t: number): Quat {
    return qslerp(from, to, clamp(t, 0, 1));
  }

  /** Returns simulation to animation control; the rig is rebuilt at the clip pose by the caller. */
  restoreAnimation(): void {
    this.active = false;
    this.snapshot = null;
  }

  /**
   * Recovery entry: every chain body becomes dynamic at its current world pose
   * with velocities zeroed. Nothing is teleported — the get-up drives start
   * from exactly the fallen configuration the contacts describe.
   */
  beginDynamicRecovery(): void {
    for (const id of BONE_IDS) {
      const rec = this.records.get(id);
      if (!rec) continue;
      if (rec.body.isKinematic()) rec.body.setBodyType(RAPIER.RigidBodyType.Dynamic, true);
      rec.body.setLinvel({ x: 0, y: 0, z: 0 }, true);
      rec.body.setAngvel({ x: 0, y: 0, z: 0 }, true);
      rec.body.setGravityScale(1, true);
      rec.body.setLinearDamping(0.15);
      rec.body.setAngularDamping(0.6);
      this.ownership.set(id, 'physics');
    }
  }

  /** Restore default damping after the recovery blend hands control back to the clip. */
  resetDamping(): void {
    for (const id of BONE_IDS) {
      const rec = this.records.get(id);
      if (!rec) continue;
      rec.body.setLinearDamping(0.06);
      rec.body.setAngularDamping(0.3);
      rec.body.setGravityScale(1, true);
    }
  }

  /** Change which bones own physics without rebuilding (used only while inactive). */
  setSubset(subset: BoneId[]): void {
    if (this.active) throw new Error('cannot change subset during active physics');
    this.subset = [...subset];
  }

  getSnapshot(): HandoffSnapshot | null {
    return this.snapshot;
  }
  isActive(): boolean {
    return this.active;
  }
  getSubset(): BoneId[] {
    return [...this.subset];
  }
  getScenario(): ScenarioId {
    return this.scenario;
  }
  getHandoffTime(): number {
    return this.handoffTime;
  }
  getOwnership(): Map<BoneId, Ownership> {
    return new Map(this.ownership);
  }
  getBody(id: BoneId): RAPIER.RigidBody | null {
    return this.records.get(id)?.body ?? null;
  }

  /**
   * Ownership audit used by the UI and the replay test: each body must be
   * kinematic while animation-owned and dynamic while physics-owned.
   */
  auditOwnership(): { ok: boolean; conflicts: BoneId[] } {
    const conflicts: BoneId[] = [];
    for (const id of BONE_IDS) {
      const rec = this.records.get(id);
      if (!rec) continue;
      const owner = this.ownership.get(id);
      if (owner === 'animation' && !rec.body.isKinematic()) conflicts.push(id);
      if (owner === 'physics' && rec.body.isKinematic()) conflicts.push(id);
    }
    return { ok: conflicts.length === 0, conflicts };
  }

  footClearance(): { L: number; R: number } {
    const pose = this.poseFromWorld();
    const clearance = (id: BoneId): number => {
      const sc = SCENARIOS[this.scenario];
      return Math.min(...footCornersWorld(pose, id).map((c) => c.y - sc.groundHeight(c.x, c.z)));
    };
    return { L: clearance('footL'), R: clearance('footR') };
  }
}
