import { expect, test } from '@playwright/test';

const readRecovery = async (page: import('@playwright/test').Page) =>
  page.evaluate(() => {
    const lab = (window as unknown as {
      __lab: {
        sim: {
          mode: string;
          time: number;
          getRecoveryState(): unknown;
          handoff: { auditOwnership(): { ok: boolean } };
        };
      };
    }).__lab;
    const state = lab.sim.getRecoveryState() as {
      status: string;
      phase: string;
      progress: number;
      plan: {
        feasible: boolean;
        blockingReasons: string[];
        basis: { orientation: string; strategy: string; horizontalTravel: number; headClearance: number };
      } | null;
      message: string;
    } | null;
    return {
      mode: lab.sim.mode,
      auditOk: lab.sim.handoff.auditOwnership().ok,
      state,
    };
  });

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(
    () => (window as unknown as { __lab?: { sim?: { time: number } } }).__lab?.sim?.time !== undefined,
  );
});

test('guided fall then contact-aware recovery returns control to the animation in place', async ({ page }) => {
  await page.locator('#auto-handoff').uncheck();
  await page.locator('[data-fall="supine"]').click();
  await page.locator('#fall').click();

  let info = await readRecovery(page);
  expect(info.mode).toBe('physics');

  await page.locator('#recover').click();
  await page.waitForFunction(
    () =>
      (window as unknown as { __lab: { sim: { getRecoveryState(): { status: string } | null } } }).__lab.sim
        .getRecoveryState()?.status === 'complete',
    { timeout: 30000 },
  );

  info = await readRecovery(page);
  expect(info.state?.status).toBe('complete');
  expect(info.mode).toBe('animating');
  expect(info.auditOk).toBe(true);
  await expect(page.locator('#mode')).toHaveText('动画控制');
  // The recovery basis panel explains the orientation/strategy/clearance decision.
  await expect(page.locator('#recovery-panel .basis-list')).toContainText('supine');
  await expect(page.locator('#recovery-panel')).toContainText('根目标由当前身体位置推导');
});

for (const orientation of ['prone', 'side'] as const) {
  test(`recovery works from a ${orientation} fall with joints intact`, async ({ page }) => {
    await page.locator('#auto-handoff').uncheck();
    await page.locator(`[data-fall="${orientation}"]`).click();
    await page.locator('#fall').click();
    await page.locator('#recover').click();
    await page.waitForFunction(
      () =>
        (window as unknown as { __lab: { sim: { getRecoveryState(): { status: string } | null } } }).__lab.sim
          .getRecoveryState()?.status === 'complete',
      { timeout: 30000 },
    );
    const info = await readRecovery(page);
    expect(info.state?.status).toBe('complete');
    expect(info.mode).toBe('animating');
    expect(info.state?.plan?.basis.orientation).toBe(orientation);
  });
}

test('low ceiling is reported unrecoverable without disabling collision', async ({ page }) => {
  await page.locator('#auto-handoff').uncheck();
  await page.locator('[data-env="ceiling"]').click();
  await page.locator('#fall').click();
  await page.locator('#recover').click();

  await expect(page.locator('#failure')).toBeVisible();
  await expect(page.locator('#failure')).toContainText('恢复不可行');
  const info = await readRecovery(page);
  expect(info.state?.status).toBe('failed');
  expect(info.state?.plan?.feasible).toBe(false);
  expect(info.state?.plan?.blockingReasons.join(' ')).toContain('低顶');
  // The character stays in physics mode; nothing teleports or passes through.
  expect(info.mode).toBe('physics');
});

test('wall environment exposes the blocking reason and recovers after switching to clear space', async ({ page }) => {
  await page.locator('#auto-handoff').uncheck();
  await page.locator('[data-env="wall"]').click();
  await page.locator('#fall').click();
  await page.locator('#recover').click();
  const blocked = await readRecovery(page);
  expect(blocked.state?.plan?.feasible).toBe(false);
  expect(blocked.state?.plan?.blockingReasons.join(' ')).toMatch(/墙|障碍/);

  await page.locator('[data-env="clear"]').click();
  await page.locator('#fall').click();
  await page.locator('#recover').click();
  await page.waitForFunction(
    () =>
      (window as unknown as { __lab: { sim: { getRecoveryState(): { status: string } | null } } }).__lab.sim
        .getRecoveryState()?.status === 'complete',
    { timeout: 30000 },
  );
  const done = await readRecovery(page);
  expect(done.state?.status).toBe('complete');
  expect(done.mode).toBe('animating');
});

test('slope scenario supports the full fall-and-recovery handoff', async ({ page }) => {
  await page.locator('#auto-handoff').uncheck();
  await page.locator('[data-scenario="slope"]').click();
  await page.locator('#fall').click();
  await page.locator('#recover').click();
  await page.waitForFunction(
    () =>
      (window as unknown as { __lab: { sim: { getRecoveryState(): { status: string } | null } } }).__lab.sim
        .getRecoveryState()?.status === 'complete',
    { timeout: 30000 },
  );
  const info = await readRecovery(page);
  expect(info.state?.status).toBe('complete');
  expect(info.mode).toBe('animating');
  expect(info.auditOk).toBe(true);
});
