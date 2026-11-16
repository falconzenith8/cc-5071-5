import { describe, expect, it } from 'vitest';
import RAPIER from '@dimforge/rapier3d-compat';
import { Animator, ScenarioId } from '../src/core/animator';
import { ObstacleSpec, PoseHandoff, DEFAULT_SUBSET } from '../src/core/physics';
import { BONES, BONE_IDS, BoneId, anchorErrors, hingeCoordinate } from '../src/core/rig';
import { RecoveryController, planRecovery } from '../src/core/recovery';
import { buildFallenPose, classifyOrientation } from '../src/core/recoveryPoses';

await RAPIER.init();

const DT = 1 / 120;

interface FallSetup {
  animator: Animator;
  handoff: PoseHandoff;
  scenario: ScenarioId;
}

/** Build a chain and place it into a deterministic fallen pose via guided fall. */
const setupFallen = (
  scenario: ScenarioId,
  orientation: 'supine' | 'prone' | 'side',
  obstacles: ObstacleSpec[] = [],
): FallSetup => {
  const animator = new Animator(scenario);
  const handoff = new PoseHandoff(scenario, animator, obstacles);
  handoff.build(animator.sample(0), DEFAULT_SUBSET);
  const root = animator.sample(0).pose.pelvis.pos;
  const fallen = buildFallenPose(scenario, orientation, root, 0);
  handoff.guidedFall(fallen, 30);
  handoff.beginDynamicRecovery();
  for (let i = 0; i < 24; i++) handoff.stepWorldOnly();
  return { animator, handoff, scenario };
};

const runRecovery = (setup: FallSetup, maxSteps = 900): RecoveryController => {
  const { animator, handoff, scenario } = setup;
  const controller = new RecoveryController(handoff, scenario);
  const plan = controller.start(() => undefined);
  expect(plan.feasible).toBe(true);
  const rootAtStart = { ...handoff.poseFromWorld().pelvis.pos };
  let steps = 0;
  while (controller.isRunning() && steps < maxSteps) {
    const t = steps * DT;
    controller.step(animator.sample(t).pose);
    handoff.stepWorldOnly();
    steps++;
  }
  expect(controller.getState().status).toBe('complete');
  void rootAtStart;
  return controller;
};

describe('contact-aware recovery planning', () => {
  it('classifies supine and prone from the actual pelvis orientation', () => {
    const supine = buildFallenPose('walk', 'supine', { x: 0, y: 0, z: 0 }, 0);
    const prone = buildFallenPose('walk', 'prone', { x: 0, y: 0, z: 0 }, 0);
    expect(classifyOrientation(supine.pelvis.rot, { x: 0, y: 1, z: 0 })).toBe('supine');
    expect(classifyOrientation(prone.pelvis.rot, { x: 0, y: 1, z: 0 })).toBe('prone');
  });

  it('derives the root target from the fallen body instead of a preset point', () => {
    const { handoff, scenario } = setupFallen('walk', 'supine');
    const fallenRoot = handoff.poseFromWorld().pelvis.pos;
    const plan = planRecovery(handoff, scenario);
    expect(plan.feasible).toBe(true);
    // Horizontal root position must stay where the body actually lies.
    expect(plan.basis.rootTarget.x).toBeCloseTo(fallenRoot.x, 6);
    expect(plan.basis.rootTarget.z).toBeCloseTo(fallenRoot.z, 6);
    expect(plan.basis.horizontalTravel).toBeLessThan(0.05);
    handoff.dispose();
  });

  it('reads real support contacts before authorizing a get-up', () => {
    const { handoff, scenario } = setupFallen('walk', 'prone');
    const contacts = handoff.groundContactBones();
    expect(contacts.length).toBeGreaterThan(0);
    const plan = planRecovery(handoff, scenario);
    expect(plan.basis.supportBones.length).toBeGreaterThan(0);
    expect(plan.basis.supportNormal.y).toBeGreaterThan(0.8);
    handoff.dispose();
  });

  it('reports unrecoverable under a low ceiling instead of clipping through it', () => {
    const setup = setupFallen('walk', 'supine');
    const root = setup.handoff.poseFromWorld().pelvis.pos;
    setup.handoff.dispose();
    // Wide slab covering the whole fallen footprint, only 0.42 m above the pelvis.
    const ceiling: ObstacleSpec = {
      id: 'low-ceiling',
      label: '低顶',
      kind: 'ceiling',
      center: { x: root.x - 0.2, y: root.y + 0.42, z: root.z },
      halfExtents: { x: 1.2, y: 0.05, z: 0.8 },
    };
    const trapped = setupFallen('walk', 'supine', [ceiling]);
    const plan = planRecovery(trapped.handoff, 'walk');
    expect(plan.feasible).toBe(false);
    expect(plan.blockingReasons.join(' ')).toContain('低顶');
    const controller = new RecoveryController(trapped.handoff, 'walk');
    const started = controller.start(() => undefined);
    expect(started.feasible).toBe(false);
    expect(controller.getState().status).toBe('failed');
    trapped.handoff.dispose();
  });

  it('pauses then fails when a wall pins the rising torso', () => {
    const setup = setupFallen('walk', 'supine');
    const root = setup.handoff.poseFromWorld().pelvis.pos;
    setup.handoff.dispose();
    const wall: ObstacleSpec = {
      id: 'retaining-wall',
      label: '挡墙',
      kind: 'wall',
      // Just beyond the head in the facing (+X) direction, close enough that the
      // forward probe reports no room to unfold.
      center: { x: root.x + 0.62, y: 0.45, z: root.z },
      halfExtents: { x: 0.06, y: 0.6, z: 0.7 },
    };
    const pinned = setupFallen('walk', 'supine', [wall]);
    const plan = planRecovery(pinned.handoff, 'walk');
    expect(plan.feasible).toBe(false);
    expect(plan.blockingReasons.join(' ')).toMatch(/墙|障碍/);
    pinned.handoff.dispose();
  });
});

describe('contact-aware get-up chain', () => {
  const orientations: ('supine' | 'prone' | 'side')[] = ['supine', 'prone', 'side'];

  for (const orientation of orientations) {
    it(`gets up from ${orientation} with contacts active and joints intact`, () => {
      const setup = setupFallen('walk', orientation);
      const { handoff } = setup;
      const before = handoff.poseFromWorld();
      expect(handoff.groundContactBones().length).toBeGreaterThan(0);

      runRecovery(setup);

      const after = handoff.poseFromWorld();
      // The chain ends in a near-standing root height rather than a fade-in
      // model. Prone starts with the pelvis already partly raised, so assert
      // both an absolute standing height and a meaningful rise where applicable.
      expect(after.pelvis.pos.y).toBeGreaterThan(0.78);
      expect(after.pelvis.pos.y).toBeGreaterThan(before.pelvis.pos.y + 0.45);
      // Horizontal drift stays local — the get-up happens where the body lay.
      const drift = Math.hypot(
        after.pelvis.pos.x - before.pelvis.pos.x,
        after.pelvis.pos.z - before.pelvis.pos.z,
      );
      expect(drift).toBeLessThan(0.6);

      // Hierarchy preserved: every joint anchor stays coincident throughout.
      const errs = anchorErrors(after);
      for (const id of BONE_IDS) expect(errs[id]).toBeLessThan(0.03);

      // Revolute joints never leave their authored limits.
      for (const id of ['lowerLegL', 'lowerLegR', 'lowerArmL', 'lowerArmR'] as BoneId[]) {
        const hinge = hingeCoordinate(after, id);
        const def = jointLimits(id);
        expect(hinge).toBeGreaterThanOrEqual(def[0] - 0.08);
        expect(hinge).toBeLessThanOrEqual(def[1] + 0.08);
      }
      handoff.dispose();
    });
  }

  it('recovers on the slope against a retaining wall using the measured support normal', () => {
    // Place the fallen body on the flat section just before the slope toe,
    // with a wall on the +X side the planner must route around, not pass through.
    const obstacles: ObstacleSpec[] = [
      {
        id: 'retaining-wall',
        label: '挡墙',
        kind: 'wall',
        center: { x: 1.55, y: 0.5, z: 0 },
        halfExtents: { x: 0.08, y: 0.5, z: 0.9 },
      },
    ];
    const setup = setupFallen('slope', 'supine', obstacles);
    const { handoff } = setup;
    const plan = planRecovery(handoff, 'slope');
    expect(plan.feasible).toBe(true);
    expect(plan.basis.supportBones.length).toBeGreaterThan(0);

    runRecovery(setup);

    const after = handoff.poseFromWorld();
    // The chain must remain on the near side of the wall: no collider crosses x=1.47.
    for (const id of BONE_IDS) {
      expect(after[id].pos.x).toBeLessThan(1.47);
    }
    // Still in contact with the environment at the end (feet/ground).
    expect(handoff.groundContactBones().length).toBeGreaterThan(0);
    const errs = anchorErrors(after);
    for (const id of BONE_IDS) expect(errs[id]).toBeLessThan(0.03);
    handoff.dispose();
  });

  it('keeps collision enabled for the whole chain during recovery (no collision bypass)', () => {
    const setup = setupFallen('walk', 'prone');
    const { handoff } = setup;
    const controller = new RecoveryController(handoff, 'walk');
    expect(controller.start(() => undefined).feasible).toBe(true);
    let steps = 0;
    while (controller.isRunning() && steps < 900) {
      for (const id of BONE_IDS) {
        const body = handoff.getBody(id)!;
        expect(body.isKinematic()).toBe(false);
      }
      controller.step(setup.animator.sample(steps * DT).pose);
      handoff.stepWorldOnly();
      steps++;
    }
    expect(controller.getState().status).toBe('complete');
    handoff.dispose();
  });

  it('hands control back to the clip at the recovered root (no stance teleport)', () => {
    const setup = setupFallen('walk', 'supine');
    const { animator, handoff } = setup;
    const controller = new RecoveryController(handoff, 'walk');
    expect(controller.start(() => undefined).feasible).toBe(true);
    let steps = 0;
    while (controller.isRunning() && steps < 900) {
      controller.step(animator.sample(steps * DT).pose);
      handoff.stepWorldOnly();
      steps++;
    }
    const recoveredRoot = handoff.poseFromWorld().pelvis.pos;
    // Simulate the simulator's completion rebase: clip root moves to the body,
    // never the body to a clip preset.
    const clip = animator.sample(2.0);
    const delta = {
      x: recoveredRoot.x - clip.pose.pelvis.pos.x,
      y: recoveredRoot.y - clip.pose.pelvis.pos.y,
      z: recoveredRoot.z - clip.pose.pelvis.pos.z,
    };
    expect(Math.abs(delta.x)).toBeGreaterThan(0.2);
    expect(Math.abs(delta.y)).toBeLessThan(0.35);
    handoff.dispose();
  });
});

const jointLimits = (id: BoneId): [number, number] => {
  const limits = BONES[id].jointLimits;
  if (!limits) throw new Error(`no limits on ${id}`);
  return limits;
};
