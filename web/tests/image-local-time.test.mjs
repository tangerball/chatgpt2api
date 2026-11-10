import assert from "node:assert/strict";
import { test } from "node:test";

import { formatImageLocalTime, imageLocalDate } from "../src/lib/image-local-time.ts";

test("UTC image timestamps use the browser's local date and time", () => {
  process.env.TZ = "Asia/Shanghai";
  assert.equal(formatImageLocalTime("2026-11-03 02:49:16"), "2026-11-03 10:49:16");
  assert.equal(formatImageLocalTime("2026-11-03T02:49:16Z"), "2026-11-03 10:49:16");
  assert.equal(imageLocalDate("2026-11-03 23:30:00", "2026-11-03"), "2026-11-04");
});

test("invalid timestamps preserve their original display and fallback date", () => {
  assert.equal(formatImageLocalTime("unknown"), "unknown");
  assert.equal(imageLocalDate("unknown", "2026-11-03"), "2026-11-03");
});
