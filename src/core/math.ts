export type Vec3 = { x: number; y: number; z: number };
export type Quat = { x: number; y: number; z: number; w: number };

export const V = (x = 0, y = 0, z = 0): Vec3 => ({ x, y, z });
export const Q = (x = 0, y = 0, z = 0, w = 1): Quat => ({ x, y, z, w });

export const vadd = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
export const vsub = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
export const vscale = (a: Vec3, s: number): Vec3 => ({ x: a.x * s, y: a.y * s, z: a.z * s });
export const vdot = (a: Vec3, b: Vec3): number => a.x * b.x + a.y * b.y + a.z * b.z;
export const vcross = (a: Vec3, b: Vec3): Vec3 => ({
  x: a.y * b.z - a.z * b.y,
  y: a.z * b.x - a.x * b.z,
  z: a.x * b.y - a.y * b.x,
});
export const vlen = (a: Vec3): number => Math.sqrt(vdot(a, a));
export const vnorm = (a: Vec3): Vec3 => {
  const l = vlen(a);
  return l > 1e-12 ? vscale(a, 1 / l) : V();
};
export const vlerp = (a: Vec3, b: Vec3, t: number): Vec3 => ({
  x: a.x + (b.x - a.x) * t,
  y: a.y + (b.y - a.y) * t,
  z: a.z + (b.z - a.z) * t,
});
export const vdist = (a: Vec3, b: Vec3): number => vlen(vsub(a, b));

export const qmul = (a: Quat, b: Quat): Quat => ({
  x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
  y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
  z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
  w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
});
export const qconj = (q: Quat): Quat => ({ x: -q.x, y: -q.y, z: -q.z, w: q.w });
export const qnorm = (q: Quat): Quat => {
  const l = Math.hypot(q.x, q.y, q.z, q.w);
  return l > 1e-12 ? { x: q.x / l, y: q.y / l, z: q.z / l, w: q.w / l } : Q();
};
export const qrot = (q: Quat, v: Vec3): Vec3 => {
  // q * (v,0) * q^-1, expanded
  const ix = q.w * v.x + q.y * v.z - q.z * v.y;
  const iy = q.w * v.y + q.z * v.x - q.x * v.z;
  const iz = q.w * v.z + q.x * v.y - q.y * v.x;
  const iw = -q.x * v.x - q.y * v.y - q.z * v.z;
  return {
    x: ix * q.w + iw * -q.x + iy * -q.z - iz * -q.y,
    y: iy * q.w + iw * -q.y + iz * -q.x - ix * -q.z,
    z: iz * q.w + iw * -q.z + ix * -q.y - iy * -q.x,
  };
};
export const qslerp = (a: Quat, b: Quat, t: number): Quat => {
  let cos = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  let bx = b.x, by = b.y, bz = b.z, bw = b.w;
  if (cos < 0) {
    cos = -cos;
    bx = -bx; by = -by; bz = -bz; bw = -bw;
  }
  if (cos > 0.9995) {
    return qnorm({
      x: a.x + (bx - a.x) * t,
      y: a.y + (by - a.y) * t,
      z: a.z + (bz - a.z) * t,
      w: a.w + (bw - a.w) * t,
    });
  }
  const theta = Math.acos(Math.min(1, cos));
  const sin = Math.sin(theta);
  const wa = Math.sin((1 - t) * theta) / sin;
  const wb = Math.sin(t * theta) / sin;
  return qnorm({
    x: a.x * wa + bx * wb,
    y: a.y * wa + by * wb,
    z: a.z * wa + bz * wb,
    w: a.w * wa + bw * wb,
  });
};
export const qfromAxisAngle = (axis: Vec3, angle: number): Quat => {
  const n = vnorm(axis);
  const s = Math.sin(angle / 2);
  return Q(n.x * s, n.y * s, n.z * s, Math.cos(angle / 2));
};
export const qfromEulerYXZ = (yaw: number, pitch: number, roll: number): Quat => {
  // Y * X * Z order (yaw, pitch, roll)
  return qmul(qmul(qfromAxisAngle(V(0, 1, 0), yaw), qfromAxisAngle(V(1, 0, 0), pitch)), qfromAxisAngle(V(0, 0, 1), roll));
};
export const qTwistAngle = (q: Quat, axis: Vec3): number => {
  const n = vnorm(axis);
  const proj = vdot(V(q.x, q.y, q.z), n);
  const twist = qnorm(Q(n.x * proj, n.y * proj, n.z * proj, q.w));
  const angle = 2 * Math.atan2(proj * (twist.w < 0 ? -1 : 1), Math.abs(twist.w));
  return angle;
};
export const qlookAt = (dir: Vec3, up: Vec3 = V(0, 1, 0)): Quat => {
  // Build a rotation whose +Y maps to `dir` (bone convention below is +Y along bone).
  const y = vnorm(dir);
  let x = vcross(y, up);
  if (vlen(x) < 1e-6) x = vcross(y, V(1, 0, 0));
  x = vnorm(x);
  const z = vnorm(vcross(x, y));
  // columns x,y,z -> quaternion
  const m00 = x.x, m01 = y.x, m02 = z.x;
  const m10 = x.y, m11 = y.y, m12 = z.y;
  const m20 = x.z, m21 = y.z, m22 = z.z;
  const tr = m00 + m11 + m22;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    return Q((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s);
  }
  if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    return Q(0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s);
  }
  if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    return Q((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s);
  }
  const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
  return Q((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s);
};
/** Rotation whose local +X/+Y/+Z axes are the given world-space basis vectors. */
export const qFromBasis = (x: Vec3, y: Vec3, z: Vec3): Quat => {
  const m00 = x.x, m01 = y.x, m02 = z.x;
  const m10 = x.y, m11 = y.y, m12 = z.y;
  const m20 = x.z, m21 = y.z, m22 = z.z;
  const tr = m00 + m11 + m22;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    return Q((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s);
  }
  if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    return Q(0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s);
  }
  if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    return Q((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s);
  }
  const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
  return Q((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s);
};

export const qAngle = (q: Quat): number => {
  const w = Math.min(1, Math.abs(q.w));
  return 2 * Math.acos(w);
};

/** World-space angular velocity from two orientations over dt. */
export const qAngularVelocity = (prev: Quat, now: Quat, dt: number): Vec3 => {
  const rel = qnorm(qmul(now, qconj(prev)));
  const q = rel.w < 0 ? { x: -rel.x, y: -rel.y, z: -rel.z, w: -rel.w } : rel;
  const angle = 2 * Math.acos(clamp(q.w, -1, 1));
  const s = Math.sqrt(Math.max(0, 1 - q.w * q.w));
  if (s < 1e-8 || angle < 1e-8) return V();
  return V((q.x / s) * (angle / dt), (q.y / s) * (angle / dt), (q.z / s) * (angle / dt));
};

export const qlookRotation = (dir: Vec3, upHint: Vec3 = V(0, 1, 0)): Quat => {
  const y = vnorm(dir);
  let z = vsub(upHint, vscale(y, vdot(upHint, y)));
  if (vlen(z) < 1e-6) z = vsub(V(0, 0, 1), vscale(y, vdot(V(0, 0, 1), y)));
  z = vnorm(z);
  const x = vnorm(vcross(y, z));
  return qFromBasis(x, y, z);
};

export const clamp = (x: number, lo: number, hi: number): number => (x < lo ? lo : x > hi ? hi : x);
export const smoothstep = (t: number): number => {
  const x = clamp(t, 0, 1);
  return x * x * (3 - 2 * x);
};
