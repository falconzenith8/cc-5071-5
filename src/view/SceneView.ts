import * as THREE from 'three';
import { SCENARIOS, ScenarioId } from '../core/animator';
import { ObstacleSpec } from '../core/physics';
import { Vec3 } from '../core/math';

export class SceneView {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  camera: THREE.PerspectiveCamera;
  private ground: THREE.Mesh | null = null;
  private grid: THREE.GridHelper | null = null;
  private obstacleGroup = new THREE.Group();
  private contactGroup = new THREE.Group();
  private contactPool: THREE.Mesh[] = [];
  private target = new THREE.Vector3();
  private desired = new THREE.Vector3();

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.scene.background = new THREE.Color(0x0e141c);
    this.scene.fog = new THREE.Fog(0x0e141c, 8, 22);

    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 60);

    const hemi = new THREE.HemisphereLight(0xbcd4f0, 0x1a2028, 0.85);
    this.scene.add(hemi);
    const key = new THREE.DirectionalLight(0xfff2dd, 1.85);
    key.position.set(3.2, 6.2, 3.8);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    key.shadow.camera.left = -4;
    key.shadow.camera.right = 4;
    key.shadow.camera.top = 4;
    key.shadow.camera.bottom = -4;
    key.shadow.camera.near = 0.5;
    key.shadow.camera.far = 18;
    key.shadow.bias = -0.0006;
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x74a8ff, 0.5);
    rim.position.set(-4, 2.5, -3);
    this.scene.add(rim);

    this.scene.add(this.obstacleGroup);
    this.scene.add(this.contactGroup);
  }

  resize(): void {
    const canvas = this.renderer.domElement;
    const w = canvas.clientWidth || 1;
    const h = canvas.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  setScenario(scenario: ScenarioId): void {
    if (this.ground) {
      this.scene.remove(this.ground);
      this.ground.geometry.dispose();
      (this.ground.material as THREE.Material).dispose();
    }
    if (this.grid) {
      this.scene.remove(this.grid);
      this.grid.dispose();
    }

    const sc = SCENARIOS[scenario];
    const x0 = -3;
    const x1 = 7;
    const z0 = -2.6;
    const z1 = 2.6;
    const nx = 96;
    const nz = 32;
    const geo = new THREE.PlaneGeometry(x1 - x0, z1 - z0, nx, nz);
    geo.rotateX(-Math.PI / 2);
    geo.translate((x0 + x1) / 2, 0, (z0 + z1) / 2);
    const pos = geo.getAttribute('position') as THREE.BufferAttribute;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const z = pos.getZ(i);
      pos.setY(i, sc.groundHeight(x, z));
    }
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({
      color: scenario === 'slope' ? 0x434f5e : 0x39424e,
      roughness: 0.95,
      metalness: 0.02,
      side: THREE.DoubleSide,
    });
    this.ground = new THREE.Mesh(geo, mat);
    this.ground.receiveShadow = true;
    this.scene.add(this.ground);

    this.grid = new THREE.GridHelper(12, 48, 0x2c3a48, 0x1f2a35);
    this.grid.position.set(2, 0.002, 0);
    if (scenario === 'slope') this.grid.visible = false;
    this.scene.add(this.grid);

    this.camera.position.set(sc.camera.pos.x, sc.camera.pos.y, sc.camera.pos.z);
    this.target.set(sc.camera.target.x, sc.camera.target.y, sc.camera.target.z);
    this.camera.lookAt(this.target);
  }

  /** Render the same fixed obstacles the physics world collides with. */
  setObstacles(obstacles: ObstacleSpec[]): void {
    this.clearObstacles();
    for (const spec of obstacles) {
      const geo = new THREE.BoxGeometry(
        spec.halfExtents.x * 2,
        spec.halfExtents.y * 2,
        spec.halfExtents.z * 2,
      );
      const color = spec.kind === 'ceiling' ? 0x8a5a3c : 0x4a5d75;
      const mat = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.8,
        metalness: 0.05,
        transparent: true,
        opacity: 0.92,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(spec.center.x, spec.center.y, spec.center.z);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.name = spec.id;
      this.obstacleGroup.add(mesh);
    }
  }

  clearObstacles(): void {
    for (const child of [...this.obstacleGroup.children]) {
      const mesh = child as THREE.Mesh;
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
      this.obstacleGroup.remove(mesh);
    }
  }

  follow(target: Vec3, lead: Vec3, dt: number): void {
    // Smooth chase that never cuts: one continuous shot so handoff pops cannot be hidden.
    this.desired.set(target.x + lead.x, target.y + lead.y, target.z + lead.z);
    const k = 1 - Math.exp(-dt * 3.2);
    this.camera.position.lerp(this.desired, k);
    this.target.lerp(new THREE.Vector3(target.x, target.y + 0.05, target.z), 1 - Math.exp(-dt * 6));
    this.camera.lookAt(this.target);
  }

  setCamera(pos: Vec3, target: Vec3): void {
    this.camera.position.set(pos.x, pos.y, pos.z);
    this.target.set(target.x, target.y, target.z);
    this.camera.lookAt(this.target);
  }

  /** Contact points rendered as dots, with a short normal stick. */
  showContacts(points: { pos: Vec3; normal: Vec3 }[]): void {
    while (this.contactPool.length < points.length) {
      const dot = new THREE.Mesh(
        new THREE.SphereGeometry(0.022, 8, 6),
        new THREE.MeshBasicMaterial({ color: 0xffc857 }),
      );
      this.contactPool.push(dot);
      this.contactGroup.add(dot);
    }
    for (let i = 0; i < this.contactPool.length; i++) {
      const dot = this.contactPool[i];
      if (i < points.length) {
        dot.visible = true;
        dot.position.set(points[i].pos.x, points[i].pos.y, points[i].pos.z);
      } else {
        dot.visible = false;
      }
    }
  }

  render(): void {
    this.renderer.render(this.scene, this.camera);
  }
}
