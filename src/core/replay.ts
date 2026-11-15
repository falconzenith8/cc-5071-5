import { Vec3 } from './math';
import { BoneId } from './rig';
import { ScenarioId } from './animator';

/** A self-contained, replayable recording of one drill run. */
export interface Replay {
  version: 1;
  scenario: ScenarioId;
  seed: number;
  speed: number;
  subset: BoneId[];
  handoffTime: number;
  impact: { bone: BoneId; impulse: Vec3; atTime: number } | null;
  /** Fixed timestep used by the simulation, so replay can be bit-stable across machines. */
  dt: number;
  /** Every user input with the simulation time it landed at. */
  inputs: { time: number; action: 'impact' | 'restore'; }[];
  /** Recorded per-frame root positions for visual comparison after a replay. */
  trace: { t: number; root: Vec3; handoff: boolean }[];
  createdAt: string;
}

export const emptyReplay = (scenario: ScenarioId, speed: number, subset: BoneId[]): Replay => ({
  version: 1,
  scenario,
  seed: 1,
  speed,
  subset,
  handoffTime: 0,
  impact: null,
  dt: 1 / 120,
  inputs: [],
  trace: [],
  createdAt: new Date().toISOString(),
});

export const serializeReplay = (r: Replay): string => JSON.stringify(r, null, 2);

export const parseReplay = (text: string): Replay => {
  const raw = JSON.parse(text) as Replay;
  if (raw.version !== 1) throw new Error(`unsupported replay version ${String(raw.version)}`);
  if (!raw.scenario || !Array.isArray(raw.subset) || !Array.isArray(raw.trace)) {
    throw new Error('replay file is missing required fields');
  }
  return raw;
};

/** Deterministic PRNG so a seed fully determines the drill. */
export const mulberry32 = (seed: number): (() => number) => {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};
