import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { demoDoc } from "../lib/fixtures";
import type { LyricsDoc } from "../types";
import { Inspector } from "./Inspector";
import { LineRow } from "./LineRow";

function renderInspector(doc: LyricsDoc, li: number, wi: number) {
  const apply = vi.fn();
  render(<Inspector doc={doc} li={li} wi={wi} apply={apply} playFrom={() => {}} editLine={() => {}}
    startTap={() => {}} duration={24} />);
  return apply;
}

describe("Inspector", () => {
  it("editing the word text saves it as a manual word", async () => {
    const apply = renderInspector(demoDoc(), 0, 5);
    const input = screen.getByLabelText("Word text");
    await userEvent.clear(input);
    await userEvent.type(input, "window,{Enter}");
    const next = apply.mock.calls.at(-1)![0] as LyricsDoc;
    expect(next.lines[0].words[5]).toMatchObject({ text: "window,", origin: "manual", conf: 1 });
    expect(next.lines[0].words[4].origin).toBe("matched");
  });

  it("the ±50 ms buttons nudge the word", () => {
    const apply = renderInspector(demoDoc(), 1, 0);
    fireEvent.click(screen.getByLabelText("Move word 50 ms later"));
    const next = apply.mock.calls[0][0] as LyricsDoc;
    expect(next.lines[1].words[0]).toMatchObject({ start: 4.85, end: 5.35, origin: "manual" });
    fireEvent.click(screen.getByLabelText("Start 50 ms earlier"));
    expect((apply.mock.calls[1][0] as LyricsDoc).lines[1].words[0].start).toBe(4.75);
  });

  it("typing exact times", async () => {
    const apply = renderInspector(demoDoc(), 2, 1);
    const end = screen.getByLabelText("Word end (seconds)");
    await userEvent.clear(end);
    await userEvent.type(end, "18.12{Enter}");
    expect((apply.mock.calls.at(-1)![0] as LyricsDoc).lines[2].words[1]).toMatchObject({ end: 18.12, origin: "manual" });
  });

  it("shows where the word came from", () => {
    renderInspector(demoDoc(), 2, 0);
    expect(screen.getByText(/not heard, time guessed/)).toBeInTheDocument();
  });
});

describe("LineRow", () => {
  const noop = () => {};
  it("highlights doubtful words and calls back with the clicked word", () => {
    const onWord = vi.fn();
    const doc = demoDoc();
    render(<LineRow line={doc.lines[0]} li={0} selectedWi={null} activeWi={null} tapWi={null} errorWords="" editing={false}
      onWord={onWord} onSeekLine={noop} onCommitLine={noop} rowRef={noop} />);
    const widow = screen.getByRole("button", { name: "widow," });
    expect(widow).toHaveClass("low");
    expect(screen.getByRole("button", { name: "Morning" })).not.toHaveClass("low");
    expect(screen.getByText("check")).toBeInTheDocument();
    fireEvent.click(widow);
    expect(onWord).toHaveBeenCalledWith(0, 5);
  });
});
