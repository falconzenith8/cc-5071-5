import * as THREE from 'three';
import { BONES, BONE_IDS, BoneId, Pose, capsuleHalfHeight } from '../core/rig';
import { Vec3, qrot, vadd } from '../core/math';
export interface BodyMesh {
  id: BoneId;
  mesh: THREE.Mesh;
  material: THREE.MeshStandardMaterial;
}

const palette: Record<string, number> = {
  pelvis: 0x6b7a8f, spine: 0x7d8ea3, chest: 0x8fa2b8,
  neck: 0x9fb3c8, head: 0xc9d6e3,
  upperArmL: 0x7f9ac4, lowerArmL: 0x93abce, handL: 0xb9c7dd,
  upperArmR: 0xc49a7f, lowerArmR: 0xceab93, handR: 0xddc7b9,
  upperLegL: 0x7fa6a0, lowerLegL: 0x93bab4, footL: 0x4d5a66,
  upperLegR: 0xa67f7f, lowerLegR: 0xba9393, footR: 0x66504d,
};

export class CharacterView {
  group = new THREE.Group();
  bodies = new Map<BoneId, BodyMesh>();
  private jointMarkers = new Map<BoneId, THREE.Mesh>();
  private jointLines = new Map<BoneId, THREE.Line>();
  private trail: THREE.Line;
  private trailPositions: Float32Array;
  private trailCount = 0;

  constructor() {
    for (const id of BONE_IDS) {
      const def = BONES[id];
      const material = new THREE.MeshStandardMaterial({
        color: palette[id] ?? 0x8899aa,
        roughness: 0.52,
        metalness: 0.06,
        flatShading: false,
      });
      let geometry: THREE.BufferGeometry;
      const shape = def.shape;
      if (shape.kind === 'capsule') {
        geometry = new THREE.CapsuleGeometry(shape.radius, capsuleHalfHeight(shape) * 2, 8, 14);
        geometry.translate(0, shape.offset.y, 0);
      } else if (shape.kind === 'ball') {
        geometry = new THREE.SphereGeometry(shape.radius, 20, 16);
        geometry.translate(shape.offset.x, shape.offset.y, shape.offset.z);
      } else {
        geometry = new THREE.BoxGeometry(
          shape.halfExtents.x * 2,
          shape.halfExtents.y * 2,
          shape.halfExtents.z * 2,
        );
        geometry.translate(shape.offset.x, shape.offset.y, shape.offset.z);
      }
      const mesh = new THREE.Mesh(geometry, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      this.bodies.set(id, { id, mesh, material });
      this.group.add(mesh);

      if (def.parent) {
        const marker = new THREE.Mesh(
          new THREE.SphereGeometry(0.026, 10, 8),
          new THREE.MeshBasicMaterial({ color: 0xffcf5c, transparent: true, opacity: 0.9 }),
        );
        marker.visible = false;
        this.jointMarkers.set(id, marker);
        this.group.add(marker);

        const lineGeom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
        const line = new THREE.Line(
          lineGeom,
          new THREE.LineBasicMaterial({ color: 0xff5c5c, transparent: true, opacity: 0.9 }),
        );
        line.visible = false;
        this.jointLines.set(id, line);
        this.group.add(line);
      }
    }

    const trailGeom = new THREE.BufferGeometry();
    this.trailPositions = new Float32Array(600 * 3);
    trailGeom.setAttribute('position', new THREE.BufferAttribute(this.trailPositions, 3));
    this.trail = new THREE.Line(
      trailGeom,
      new THREE.LineBasicMaterial({ color: 0x59d3ff, transparent: true, opacity: 0.75 }),
    );
    this.trail.frustumCulled = false;
    this.trail.visible = false;
    this.group.add(this.trail);
  }

  /** Apply a world-space pose to every bone mesh. */
  setPose(pose: Pose, ownership: Map<BoneId, 'animation' | 'physics'>): void {
    for (const id of BONE_IDS) {
      const b = this.bodies.get(id)!;
      b.mesh.position.set(pose[id].pos.x, pose[id].pos.y, pose[id].pos.z);
      b.mesh.quaternion.set(pose[id].rot.x, pose[id].rot.y, pose[id].rot.z, pose[id].rot.w);
      const owner = ownership.get(id) ?? 'animation';
      b.material.emissive.setHex(owner === 'physics' ? 0x1d3550 : 0x000000);
      b.material.emissiveIntensity = owner === 'physics' ? 0.65 : 0;
    }
  }

  /** Show joint anchor coincidence: green when close, red with an error line when apart. */
  showJointErrors(pose: Pose, errors: Record<BoneId, number>, threshold = 0.01): void {
    for (const id of BONE_IDS) {
      const def = BONES[id];
      if (!def.parent) continue;
      const marker = this.jointMarkers.get(id)!;
      const line = this.jointLines.get(id)!;
      const parentPose = pose[def.parent];
      const anchor = vadd(parentPose.pos, qrot(parentPose.rot, def.parentAnchor));
      marker.position.set(anchor.x, anchor.y, anchor.z);
      marker.visible = true;
      const mat = marker.material as THREE.MeshBasicMaterial;
      mat.color.setHex(errors[id] > threshold ? 0xff4d4d : 0x7be495);

      const positions = line.geometry.getAttribute('position') as THREE.BufferAttribute;
      positions.setXYZ(0, anchor.x, anchor.y, anchor.z);
      positions.setXYZ(1, pose[id].pos.x, pose[id].pos.y, pose[id].pos.z);
      positions.needsUpdate = true;
      line.geometry.computeBoundingSphere();
      line.visible = errors[id] > threshold;
    }
  }

  hideJointMarkers(): void {
    for (const m of this.jointMarkers.values()) m.visible = false;
    for (const l of this.jointLines.values()) l.visible = false;
  }

  pushTrail(p: Vec3): void {
    if (this.trailCount >= 600) {
      this.trailPositions.copyWithin(0, 3);
      this.trailCount = 599;
    }
    this.trailPositions.set([p.x, p.y, p.z], this.trailCount * 3);
    this.trailCount++;
    const attr = this.trail.geometry.getAttribute('position') as THREE.BufferAttribute;
    attr.needsUpdate = true;
    this.trail.geometry.setDrawRange(0, this.trailCount);
    this.trail.geometry.computeBoundingSphere();
    this.trail.visible = this.trailCount > 1;
  }

  clearTrail(): void {
    this.trailCount = 0;
    this.trail.visible = false;
  }
}
