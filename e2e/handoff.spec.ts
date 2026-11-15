import { expect, test } from '@playwright/test';

const readLab = async (page: import('@playwright/test').Page) =>
  page.evaluate(() => {
    const lab = (window as unknown as {
      __lab: {
        sim: {
          mode: string;
          time: number;
          lastSnapshot: { converged: boolean; maxPosResidual: number } | null;
          ownership: Map<string, string>;
          replay: { inputs: unknown[] } | null;
          recording: unknown;
          handoff: { auditOwnership(): { ok: boolean; conflicts: string[] } };
        };
        failureBone: () => string | null;
      };
    }).__lab;
    const snap = lab.sim.lastSnapshot;
    return {
      mode: lab.sim.mode,
      time: lab.sim.time,
      converged: snap?.converged ?? null,
      residualMm: snap ? snap.maxPosResidual * 1000 : null,
      auditOk: lab.sim.handoff.auditOwnership().ok,
      physicsBones: [...lab.sim.ownership.values()].filter((v) => v === 'physics').length,
      totalBones: lab.sim.ownership.size,
      hasReplay: lab.sim.replay !== null,
      failureBone: lab.failureBone(),
    };
  });

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('#stage')).toBeVisible();
  await page.waitForFunction(
    () => (window as unknown as { __lab?: { sim?: { time: number } } }).__lab?.sim?.time !== undefined,
  );
});

test('drives walk handoff from the UI and reports the per-bone audit', async ({ page }) => {
  await page.locator('#auto-handoff').uncheck();
  await page.waitForTimeout(150);
  await page.locator('#strike').click();

  await expect(page.locator('#result .badge.good')).toHaveText('交接通过');
  const lab = await readLab(page);
  expect(lab.mode).toBe('physics');
  expect(lab.converged).toBe(true);
  expect(lab.residualMm).toBeLessThan(30);
  expect(lab.auditOk).toBe(true);
  expect(lab.physicsBones).toBe(lab.totalBones);
  expect(lab.failureBone).toBeNull();

  const rows = page.locator('#result tbody tr');
  await expect(rows).toHaveCount(lab.totalBones);
  await expect(page.locator('#result .metric').first()).toBeVisible();
  await expect(page.locator('#owner-value')).toHaveText(`${lab.totalBones} / ${lab.totalBones}`);

  const jointCells = await page.locator('#result tbody tr td:nth-child(3)').allTextContents();
  for (const cell of jointCells) expect(Number(cell)).toBeLessThan(10);
});

test('restore hands control back to the animation clip without a stance pop', async ({ page }) => {
  await page.locator('#auto-handoff').uncheck();
  await page.locator('#strike').click();
  await expect(page.locator('#result .badge.good')).toBeVisible();

  await page.locator('#restore').click();
  await expect(page.locator('#mode')).toHaveText('动画控制');
  const lab = await readLab(page);
  expect(lab.mode).toBe('animating');
  expect(lab.auditOk).toBe(true);
  await expect(page.locator('#owner-value')).toHaveText(`0 / ${lab.totalBones}`);

  await page.locator('#strike').click();
  await expect(page.locator('#result .badge.good')).toBeVisible();
  expect((await readLab(page)).converged).toBe(true);
});

test('upper-body-only handoff marks the boundary joint and keeps legs animated', async ({ page }) => {
  await page.locator('#auto-handoff').uncheck();
  await page.locator('[data-subset="upper"]').click();
  await page.waitForTimeout(150);
  await page.locator('#strike').click();

  await expect(page.locator('#result .badge.good')).toBeVisible();
  await expect(page.locator('#result tbody tr.boundary')).toHaveCount(1);
  await expect(page.locator('#result tbody tr.boundary .tag')).toHaveText('边界');

  const lab = await readLab(page);
  expect(lab.auditOk).toBe(true);
  expect(lab.physicsBones).toBeGreaterThan(0);
  expect(lab.physicsBones).toBeLessThan(lab.totalBones);
  await expect(page.locator('#subset-desc')).not.toBeEmpty();
});

test('turn and slope scenarios hand off through the same core', async ({ page }) => {
  await page.locator('#auto-handoff').uncheck();
  for (const scenario of ['turn', 'slope'] as const) {
    await page.locator(`[data-scenario="${scenario}"]`).click();
    await page.waitForTimeout(150);
    await page.locator('#strike').click();
    await expect(page.locator('#result .badge.good')).toHaveText('交接通过');
    const lab = await readLab(page);
    expect(lab.converged).toBe(true);
    expect(lab.auditOk).toBe(true);
    expect(lab.residualMm).toBeLessThan(30);
    await page.locator('#reset').click();
  }
});

test('replay round-trips through export and import without a UI camera cut', async ({ page }) => {
  await page.locator('#auto-handoff').uncheck();
  await page.locator('#record').click();
  await page.locator('#strike').click();
  await expect(page.locator('#result .badge.good')).toBeVisible();

  const downloadPromise = page.waitForEvent('download');
  await page.locator('#export').click();
  const download = await downloadPromise;
  const path = await download.path();
  expect(path).toBeTruthy();

  await page.locator('#reset').click();
  await page.locator('#import').setInputFiles(path!);
  await expect(page.locator('#replay-state')).toContainText('已载入回放');
  await page.locator('#replay').click();
  await page.waitForFunction(
    () => (window as unknown as { __lab: { sim: { mode: string } } }).__lab.sim.mode === 'physics',
  );

  await expect(page.locator('#result .badge.good')).toBeVisible();
  const lab = await readLab(page);
  expect(lab.converged).toBe(true);
  expect(lab.auditOk).toBe(true);
});
