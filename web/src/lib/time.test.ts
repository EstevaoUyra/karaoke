import { expect, it } from "vitest";
import { fmtTime, parseTime } from "./time";

it("formats and parses times", () => {
  expect(fmtTime(83.456)).toBe("1:23.46");
  expect(fmtTime(5.5)).toBe("0:05.50");
  expect(fmtTime(59.999)).toBe("1:00.00");
  expect(fmtTime(65, 0)).toBe("1:05");
  expect(parseTime("1:23.46")).toBeCloseTo(83.46, 6);
  expect(parseTime("83.46")).toBeCloseTo(83.46, 6);
  expect(parseTime("abc")).toBeNull();
});
