import {
  Q,
  V,
  Quat,
  Vec3,
  qFromBasis,
  qconj,
  qmul,
  qnorm,
  qrot,
  vadd,
  vdist,
  vlen,
  vnorm,
  vsub,
} from './math';

export type BoneId =
  | 'pelvis'
  | 'spine'
  | 'chest'
  | 'neck'
  | 'head'
  | 'upperArmL'
  | 'lowerArmL'
  | 'handL'
  | 'upperArmR'
  | 'lowerArmR'
  | 'handR'
  | 'upperLegL'
  | 'lowerLegL'
  | 'footL'
  | 'upperLegR'
  | 'lowerLegR'
  | 'footR';

export type BoneShape =
  | { kind: 'capsule'; radius: number; length: number; offset: Vec3 }
  | { kind: 'ball'; radius: number; offset: Vec3 }
  | { kind: 'cuboid'; halfExtents: Vec3; offset: Vec3 };

export interface BoneDef {
  id: BoneId;
  label: string;
  parent: BoneId | null;
  /** Joint anchor in the parent bone frame. The bone's origin IS its proximal joint. */
  parentAnchor: Vec3;
  /** Bind orientation relative to the parent frame. */
  restLocalRot: Quat;
  shape: BoneShape;
  mass: number;
  joint: 'spherical' | 'revolute' | 'root';
  /** Hinge range in the bone's local joint coordinate, relative to the bind pose. */
  jointLimits?: [number, number];
}

export interface BonePose {
  pos: Vec3;
  rot: Quat;
}

export type Pose = Record<BoneId, BonePose>;

export const zero = (): Vec3 => V();

/** Bone frame: local +Y runs proximal -> distal; local +Z follows zHint as closely as possible. */
export const boneRotFromDir = (dir: Vec3, zHint: Vec3): Quat => {
  const y = vnorm(dir);
  const proj = y.x * zHint.x + y.y * zHint.y + y.z * zHint.z;
  let z = vsub(zHint, { x: y.x * proj, y: y.y * proj, z: y.z * proj });
  if (vlen(z) < 1e-6) {
    const alt = Math.abs(y.y) > 0.9 ? V(0, 0, 1) : V(0, 1, 0);
    const p2 = y.x * alt.x + y.y * alt.y + y.z * alt.z;
    z = vsub(alt, { x: y.x * p2, y: y.y * p2, z: y.z * p2 });
  }
  z = vnorm(z);
  const x = vnorm({
    x: y.y * z.z - y.z * z.y,
    y: y.z * z.x - y.x * z.z,
    z: y.x * z.y - y.y * z.x,
  });
  return qFromBasis(x, y, z);
};

// `length` is the joint-to-joint segment length: the capsule's rounded tips sit
// exactly on the proximal and distal anchors, so a bone never pokes through its
// child joint and into the ground.
const capsule = (radius: number, length: number): BoneShape => ({
  kind: 'capsule', radius, length, offset: V(0, length / 2, 0),
});

/** Cylinder half-height of a capsule whose tips reach both joint anchors. */
export const capsuleHalfHeight = (shape: { radius: number; length: number }): number =>
  Math.max(0, shape.length / 2 - shape.radius);

// Character faces +X, left side is -Z. Bone local X is therefore the sagittal hinge axis (+Z).
const FORWARD = V(1, 0, 0);
const UP = V(0, 1, 0);
const armRestL = boneRotFromDir(vnorm(V(-0.16, -1, 0)), FORWARD);
const armRestR = boneRotFromDir(vnorm(V(0.16, -1, 0)), FORWARD);
const legRest = boneRotFromDir(V(0, -1, 0), FORWARD);
const footRest = boneRotFromDir(vnorm(V(1, -0.1, 0)), UP);

export const PELVIS_HEIGHT = 0.94;
export const FOOT_CLEARANCE = 0.004;

/**
 * Single source of truth for the animated skeleton, the rigid body chain,
 * the collision shapes and the joint anchors. Nothing else may re-declare them.
 */
export const BONES: Record<BoneId, BoneDef> = {
  pelvis: {
    id: 'pelvis', label: '骨盆', parent: null,
    parentAnchor: V(), restLocalRot: Q(),
    shape: { kind: 'cuboid', halfExtents: V(0.125, 0.085, 0.095), offset: V(0, 0.005, 0) },
    mass: 12, joint: 'root',
  },
  spine: {
    id: 'spine', label: '腰椎', parent: 'pelvis',
    parentAnchor: V(0, 0.085, 0), restLocalRot: Q(),
    shape: capsule(0.09, 0.16), mass: 8, joint: 'spherical',
  },
  chest: {
    id: 'chest', label: '胸腔', parent: 'spine',
    parentAnchor: V(0, 0.16, 0), restLocalRot: Q(),
    shape: capsule(0.115, 0.20), mass: 14, joint: 'spherical',
  },
  neck: {
    id: 'neck', label: '颈', parent: 'chest',
    parentAnchor: V(0, 0.20, 0), restLocalRot: Q(),
    shape: capsule(0.045, 0.10), mass: 2, joint: 'spherical',
  },
  head: {
    id: 'head', label: '头', parent: 'neck',
    parentAnchor: V(0, 0.10, 0), restLocalRot: Q(),
    shape: { kind: 'ball', radius: 0.11, offset: V(0, 0.10, 0) }, mass: 5, joint: 'spherical',
  },
  upperArmL: {
    id: 'upperArmL', label: '左上臂', parent: 'chest',
    parentAnchor: V(0, 0.155, -0.185), restLocalRot: armRestL,
    shape: capsule(0.055, 0.28), mass: 3, joint: 'spherical',
  },
  lowerArmL: {
    id: 'lowerArmL', label: '左前臂', parent: 'upperArmL',
    parentAnchor: V(0, 0.28, 0), restLocalRot: Q(),
    shape: capsule(0.045, 0.26), mass: 2.2, joint: 'revolute', jointLimits: [0, 2.45],
  },
  handL: {
    id: 'handL', label: '左手', parent: 'lowerArmL',
    parentAnchor: V(0, 0.26, 0), restLocalRot: Q(),
    shape: capsule(0.042, 0.17), mass: 0.8, joint: 'spherical',
  },
  upperArmR: {
    id: 'upperArmR', label: '右上臂', parent: 'chest',
    parentAnchor: V(0, 0.155, 0.185), restLocalRot: armRestR,
    shape: capsule(0.055, 0.28), mass: 3, joint: 'spherical',
  },
  lowerArmR: {
    id: 'lowerArmR', label: '右前臂', parent: 'upperArmR',
    parentAnchor: V(0, 0.28, 0), restLocalRot: Q(),
    shape: capsule(0.045, 0.26), mass: 2.2, joint: 'revolute', jointLimits: [0, 2.45],
  },
  handR: {
    id: 'handR', label: '右手', parent: 'lowerArmR',
    parentAnchor: V(0, 0.26, 0), restLocalRot: Q(),
    shape: capsule(0.042, 0.17), mass: 0.8, joint: 'spherical',
  },
  upperLegL: {
    id: 'upperLegL', label: '左大腿', parent: 'pelvis',
    parentAnchor: V(0, -0.05, -0.10), restLocalRot: legRest,
    shape: capsule(0.075, 0.42), mass: 8, joint: 'spherical',
  },
  lowerLegL: {
    id: 'lowerLegL', label: '左小腿', parent: 'upperLegL',
    parentAnchor: V(0, 0.42, 0), restLocalRot: Q(),
    shape: capsule(0.06, 0.42), mass: 5, joint: 'revolute', jointLimits: [-2.4, 0.05],
  },
  footL: {
    id: 'footL', label: '左脚', parent: 'lowerLegL',
    parentAnchor: V(0, 0.42, 0), restLocalRot: footRest,
    shape: { kind: 'cuboid', halfExtents: V(0.085, 0.15, 0.045), offset: V(0, 0.06, 0) },
    mass: 1.2, joint: 'spherical',
  },
  upperLegR: {
    id: 'upperLegR', label: '右大腿', parent: 'pelvis',
    parentAnchor: V(0, -0.05, 0.10), restLocalRot: legRest,
    shape: capsule(0.075, 0.42), mass: 8, joint: 'spherical',
  },
  lowerLegR: {
    id: 'lowerLegR', label: '右小腿', parent: 'upperLegR',
    parentAnchor: V(0, 0.42, 0), restLocalRot: Q(),
    shape: capsule(0.06, 0.42), mass: 5, joint: 'revolute', jointLimits: [-2.4, 0.05],
  },
  footR: {
    id: 'footR', label: '右脚', parent: 'lowerLegR',
    parentAnchor: V(0, 0.42, 0), restLocalRot: footRest,
    shape: { kind: 'cuboid', halfExtents: V(0.085, 0.15, 0.045), offset: V(0, 0.06, 0) },
    mass: 1.2, joint: 'spherical',
  },
};

export const BONE_IDS: BoneId[] = [
  'pelvis', 'spine', 'chest', 'neck', 'head',
  'upperArmL', 'lowerArmL', 'handL', 'upperArmR', 'lowerArmR', 'handR',
  'upperLegL', 'lowerLegL', 'footL', 'upperLegR', 'lowerLegR', 'footR',
];

export const ROOT_ID: BoneId = 'pelvis';
export const TOTAL_MASS = BONE_IDS.reduce((s, id) => s + BONES[id].mass, 0);

/** Upper body subtree used by partial handoff. */
export const UPPER_BODY: BoneId[] = [
  'chest', 'neck', 'head',
  'upperArmL', 'lowerArmL', 'handL', 'upperArmR', 'lowerArmR', 'handR',
];

export const isInSubtree = (root: BoneId, id: BoneId): boolean => {
  let p: BoneId | null = id;
  while (p) {
    if (p === root) return true;
    p = BONES[p].parent;
  }
  return false;
};

export const boundaryParent = (subset: BoneId[]): Partial<Record<BoneId, BoneId>> => {
  const out: Partial<Record<BoneId, BoneId>> = {};
  const set = new Set(subset);
  for (const id of subset) {
    const p = BONES[id].parent;
    if (p && !set.has(p)) out[id] = p;
  }
  return out;
};

/** Forward kinematics. `localRot` is applied inside each bone's rest frame. */
export const computePose = (
  rootPos: Vec3,
  rootRot: Quat,
  localRot: Partial<Record<BoneId, Quat>>,
): Pose => {
  const pose = {} as Pose;
  pose[ROOT_ID] = { pos: rootPos, rot: qnorm(rootRot) };
  for (const id of BONE_IDS) {
    const def = BONES[id];
    if (!def.parent) continue;
    const parent = pose[def.parent];
    const local = localRot[id];
    pose[id] = {
      pos: vadd(parent.pos, qrot(parent.rot, def.parentAnchor)),
      rot: qnorm(qmul(qmul(parent.rot, def.restLocalRot), local ?? Q())),
    };
  }
  return pose;
};

export const bindPose = (pos: Vec3 = V(0, PELVIS_HEIGHT, 0)): Pose => computePose(pos, Q(), {});

/** Joint error: child origin vs the parent-side joint anchor. */
export const anchorErrors = (pose: Pose): Record<BoneId, number> => {
  const out = {} as Record<BoneId, number>;
  for (const id of BONE_IDS) {
    const def = BONES[id];
    if (!def.parent) {
      out[id] = 0;
      continue;
    }
    const parent = pose[def.parent];
    out[id] = vdist(pose[id].pos, vadd(parent.pos, qrot(parent.rot, def.parentAnchor)));
  }
  return out;
};

/** Hinge coordinate of the child relative to its parent about the child's local X axis. */
export const hingeCoordinate = (pose: Pose, id: BoneId): number => {
  const parentId = BONES[id].parent;
  if (!parentId) return 0;
  const rel = qmul(qconj(pose[parentId].rot), pose[id].rot);
  const q = rel.w < 0 ? { x: -rel.x, y: -rel.y, z: -rel.z, w: -rel.w } : rel;
  return 2 * Math.atan2(q.x, q.w);
};

export const copyPose = (pose: Pose): Pose => {
  const out = {} as Pose;
  for (const id of BONE_IDS) out[id] = { pos: { ...pose[id].pos }, rot: { ...pose[id].rot } };
  return out;
};

/** Transform a point from a bone's local frame to world space. */
export const localToWorld = (bone: BonePose, p: Vec3): Vec3 => vadd(bone.pos, qrot(bone.rot, p));
export const worldToLocal = (bone: BonePose, p: Vec3): Vec3 =>
  qrot(qconj(bone.rot), vsub(p, bone.pos));

/** Foot box corners in world space, for ground clamping and contact diagnostics. */
export const footCornersWorld = (pose: Pose, id: BoneId): Vec3[] => {
  const shape = BONES[id].shape;
  if (shape.kind !== 'cuboid') return [];
  const out: Vec3[] = [];
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      for (const sz of [-1, 1]) {
        out.push(
          localToWorld(
            pose[id],
            V(
              shape.offset.x + sx * shape.halfExtents.x,
              shape.offset.y + sy * shape.halfExtents.y,
              shape.offset.z + sz * shape.halfExtents.z,
            ),
          ),
        );
      }
    }
  }
  return out;
};

/** Lowest point of a bone's collision shape in world space (approximate for capsules). */
export const lowestPoint = (pose: Pose, id: BoneId): number => {
  const shape = BONES[id].shape;
  if (shape.kind === 'cuboid') {
    return Math.min(...footCornersWorld(pose, id).map((c) => c.y));
  }
  if (shape.kind === 'ball') {
    return localToWorld(pose[id], shape.offset).y - shape.radius;
  }
  const center = localToWorld(pose[id], shape.offset);
  const axis = qrot(pose[id].rot, V(0, 1, 0));
  const half = capsuleHalfHeight(shape);
  return Math.min(
    center.y - axis.y * half - shape.radius,
    center.y + axis.y * half - shape.radius,
  );
};
