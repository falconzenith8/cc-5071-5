import { describe, expect, it } from 'vitest';
import RAPIER from '@dimforge/rapier3d-compat';
import { Animator, SCENARIOS, ScenarioId } from '../src/core/animator';
import { PoseHandoff, DEFAULT_SUBSET } from '../src/core/physics';
import { BONE_IDS, anchorErrors, bindPose, hingeCoordinate } from '../src/core/rig';

await RAPIER.init();

const run = (scenario: ScenarioId) => {
  const animator = new Animator(scenario);
  const t = SCENARIOS[scenario].handoffTime;
  const sample = animator.sample(t);
  const handoff = new PoseHandoff(scenario, animator);
  handoff.build(sample, DEFAULT_SUBSET);
  const snapshot = handoff.handoff(sample);
  return { animator, sample, handoff, snapshot };
};

describe('rig', () => {
  it('bind pose has zero joint error', () => {
    const errs = anchorErrors(bindPose());
    for (const id of BONE_IDS) expect(errs[id]).toBeLessThan(1e-9);
  });

  it('animated poses keep joints coincident', () => {
    const animator = new Animator('walk');
    for (const t of [0, 0.3, 0.86, 1.7]) {
      const s = animator.sample(t);
      const errs = anchorErrors(s.pose);
      for (const id of BONE_IDS) expect(errs[id]).toBeLessThan(1e-9);
    }
  });

  it('hinge coordinates stay inside limits during the clips', () => {
    for (const scenario of ['walk', 'turn', 'slope'] as ScenarioId[]) {
      const animator = new Animator(scenario);
      for (let t = 0; t < 2.5; t += 0.05) {
        const s = animator.sample(t);
        for (const id of ['lowerLegL', 'lowerLegR'] as const) {
          const a = hingeCoordinate(s.pose, id);
          expect(a).toBeGreaterThanOrEqual(-2.45);
          expect(a).toBeLessThanOrEqual(0.1);
        }
      }
    }
  });
});

describe('handoff', () => {
  for (const scenario of ['walk', 'turn', 'slope'] as ScenarioId[]) {
    it(`handoff in ${scenario} is continuous and keeps joint anchors`, () => {
      const { handoff, snapshot } = run(scenario);
      console.log(scenario, JSON.stringify({
        maxPosDelta: snapshot.maxPosDelta,
        maxRotDeltaDeg: snapshot.maxRotDeltaDeg,
        maxPosResidual: snapshot.maxPosResidual,
        maxJointError: snapshot.maxJointError,
        maxInheritError: snapshot.maxInheritError,
        maxInheritAngError: snapshot.maxInheritAngError,
        maxSolverSpeedChange: snapshot.maxSolverSpeedChange,
        rootBefore: snapshot.rootSpeedBefore,
        rootAfter: snapshot.rootSpeedAfter,
        contacts: snapshot.contacts,
        failed: snapshot.failedChecks,
      }, null, 1));
      expect(snapshot.maxJointError).toBeLessThan(0.01);
      expect(snapshot.maxPosResidual).toBeLessThan(0.03);
      expect(snapshot.maxInheritError).toBeLessThan(1e-6);
      expect(handoff.auditOwnership().ok).toBe(true);
      handoff.dispose();
    });
  }
});
