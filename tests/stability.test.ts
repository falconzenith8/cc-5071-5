import { describe, expect, it } from 'vitest';
import RAPIER from '@dimforge/rapier3d-compat';
import { Animator, SCENARIOS, ScenarioId } from '../src/core/animator';
import { PoseHandoff, DEFAULT_SUBSET, UPPER_BODY_SUBSET } from '../src/core/physics';
import { BONE_IDS, anchorErrors } from '../src/core/rig';

await RAPIER.init();

const build = (scenario: ScenarioId, subset = DEFAULT_SUBSET) => {
  const animator = new Animator(scenario);
  const t = SCENARIOS[scenario].handoffTime;
  const sample = animator.sample(t);
  const handoff = new PoseHandoff(scenario, animator);
  handoff.build(sample, subset);
  return { animator, sample, handoff };
};

describe('post-handoff stability', () => {
  for (const scenario of ['walk', 'turn', 'slope'] as ScenarioId[]) {
    it(`stays finite and jointed for 2s after handoff in ${scenario}`, () => {
      const { animator, sample, handoff } = build(scenario);
      handoff.handoff(sample, { impact: { bone: 'chest', impulse: { x: -90, y: 20, z: 0 } } });
      let worst = 0;
      for (let i = 0; i < 240; i++) {
        const t = sample.animTime + (i + 1) / 120;
        handoff.step(animator.sample(t));
        const errs = anchorErrors(handoff.poseFromWorld());
        worst = Math.max(worst, ...BONE_IDS.map((id) => errs[id]));
      }
      const pose = handoff.poseFromWorld();
      const allFinite = BONE_IDS.every((id) =>
        Number.isFinite(pose[id].pos.x) && Number.isFinite(pose[id].pos.y) && Number.isFinite(pose[id].pos.z),
      );
      console.log(scenario, 'worst joint error over 2s:', (worst * 1000).toFixed(2), 'mm', 'finite:', allFinite);
      expect(allFinite).toBe(true);
      expect(worst).toBeLessThan(0.03);
      handoff.dispose();
    });
  }

  it('keeps joint anchors when only the upper body is handed off', () => {
    const { animator, sample, handoff } = build('walk', UPPER_BODY_SUBSET);
    const snap = handoff.handoff(sample, { impact: { bone: 'chest', impulse: { x: 40, y: 10, z: 0 } } });
    const boundary = snap.boundary;
    console.log('boundary', boundary, 'subset', snap.subset.length);
    expect(Object.keys(boundary)).toEqual(['chest']);
    const audit = handoff.auditOwnership();
    expect(audit.ok).toBe(true);
    for (let i = 0; i < 120; i++) handoff.step(animator.sample(sample.animTime + (i + 1) / 120));
    const pose = handoff.poseFromWorld();
    const errs = anchorErrors(pose);
    console.log('boundary joint error after 1s', (errs.chest * 1000).toFixed(2), 'mm');
    expect(errs.chest).toBeLessThan(0.03);
    // Lower body must still be exactly on the animated pose.
    expect(handoff.getOwnership().get('upperLegL')).toBe('animation');
    handoff.dispose();
  });
});
