import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { ExtensionAPI } from "../../src/core/extensions/index.ts";
import { MAX_CONTEXT_HANDOFF_CHARS, type SessionEntry, SessionManager } from "../../src/core/session-manager.ts";
import { createHarness, getMessageText, type Harness } from "./harness.ts";

const OVERFLOW = "prompt is too long: 300000 tokens > 128000 maximum";

type CompactionInternals = {
	_getSummarizationRequestAuth: (...args: unknown[]) => Promise<unknown>;
};

function createFileSession(tempDirs: string[]): SessionManager {
	const tempDir = mkdtempSync(join(tmpdir(), "pi-auto-boundary-"));
	tempDirs.push(tempDir);
	return SessionManager.create(tempDir, join(tempDir, "sessions"));
}

function claimBoundary(pi: ExtensionAPI): void {
	pi.on("session_before_auto_compact", (event) => ({
		newContext: { handoff: `handoff after ${event.reason}` },
	}));
}

function overflowResponse() {
	return fauxAssistantMessage("", { stopReason: "error", errorMessage: OVERFLOW });
}

function forbidSummarizationAuth(harness: Harness) {
	return vi
		.spyOn(harness.session as unknown as CompactionInternals, "_getSummarizationRequestAuth")
		.mockRejectedValue(new Error("summarization auth must not be resolved"));
}

describe("automatic compaction context-window boundary", () => {
	const harnesses: Harness[] = [];
	const tempDirs: string[] = [];

	afterEach(() => {
		vi.restoreAllMocks();
		while (harnesses.length > 0) harnesses.pop()?.cleanup();
		while (tempDirs.length > 0) rmSync(tempDirs.pop()!, { recursive: true, force: true });
	});

	it("claims an oversized first-turn overflow before summarizer auth and retries from the handoff", async () => {
		const sessionManager = createFileSession(tempDirs);
		const harness = await createHarness({ sessionManager, extensionFactories: [claimBoundary] });
		harnesses.push(harness);
		const summarizationAuth = forbidSummarizationAuth(harness);
		let retryTexts: string[] = [];
		harness.setResponses([
			overflowResponse(),
			(context) => {
				retryTexts = context.messages.map(getMessageText);
				return fauxAssistantMessage("continued in a fresh window");
			},
		]);

		await harness.session.prompt("x".repeat(600_000));

		expect(summarizationAuth).not.toHaveBeenCalled();
		expect(harness.faux.state.callCount).toBe(2);
		expect(sessionManager.getEntries().filter((entry) => entry.type === "context_window")).toHaveLength(1);
		expect(sessionManager.getEntries().filter((entry) => entry.type === "compaction")).toHaveLength(0);
		expect(retryTexts).toEqual([expect.stringContaining("handoff after overflow")]);
		expect(harness.eventsOfType("compaction_end").at(-1)).toMatchObject({
			reason: "overflow",
			aborted: false,
			willRetry: true,
			contextWindowStarted: true,
		});
		const sessionFile = sessionManager.getSessionFile();
		if (!sessionFile) throw new Error("Expected a file-backed session");
		expect(
			SessionManager.open(sessionFile)
				.getEntries()
				.filter((entry) => entry.type === "context_window"),
		).toHaveLength(1);
	});

	it("rejects an oversized claimed handoff without retrying or reporting success", async () => {
		const sessionManager = createFileSession(tempDirs);
		const harness = await createHarness({
			sessionManager,
			extensionFactories: [
				(pi) => {
					pi.on("session_before_auto_compact", () => ({
						newContext: { handoff: "h".repeat(MAX_CONTEXT_HANDOFF_CHARS + 1) },
					}));
				},
			],
		});
		harnesses.push(harness);
		const summarizationAuth = forbidSummarizationAuth(harness);
		harness.setResponses([overflowResponse(), fauxAssistantMessage("must remain unused")]);

		await harness.session.prompt("x".repeat(600_000));

		expect(summarizationAuth).not.toHaveBeenCalled();
		expect(harness.faux.state.callCount).toBe(1);
		expect(harness.getPendingResponseCount()).toBe(1);
		expect(sessionManager.getEntries().filter((entry) => entry.type === "context_window")).toHaveLength(0);
		expect(harness.session.messages).toHaveLength(1);
		expect(getMessageText(harness.session.messages[0])).toHaveLength(600_000);
		expect(harness.session.messages.some((message) => message.role === "custom")).toBe(false);
		expect(harness.eventsOfType("compaction_end").some((event) => event.contextWindowStarted)).toBe(false);
		expect(harness.eventsOfType("compaction_end").at(-1)).toMatchObject({
			reason: "overflow",
			willRetry: false,
			errorMessage: `Context overflow recovery failed: Context window handoff exceeds ${MAX_CONTEXT_HANDOFF_CHARS} character limit`,
		});
	});

	it("does not report or project a boundary when persistence fails", async () => {
		const failures: string[] = [];
		const sessionManager = createFileSession(tempDirs);
		const harness = await createHarness({
			sessionManager,
			extensionFactories: [
				claimBoundary,
				(pi) => {
					pi.on("session_compact_failed", (event) => {
						if (event.errorMessage) failures.push(event.errorMessage);
					});
				},
			],
		});
		harnesses.push(harness);
		const summarizationAuth = forbidSummarizationAuth(harness);
		const persist = sessionManager._persist.bind(sessionManager);
		vi.spyOn(sessionManager, "_persist").mockImplementation((entry: SessionEntry) => {
			if (entry.type === "context_window") throw new Error("disk full");
			persist(entry);
		});
		harness.setResponses([overflowResponse()]);

		await harness.session.prompt("x".repeat(600_000));

		expect(summarizationAuth).not.toHaveBeenCalled();
		expect(sessionManager.getEntries().filter((entry) => entry.type === "context_window")).toHaveLength(0);
		expect(sessionManager.buildContextEntries().some((entry) => entry.type === "context_window")).toBe(false);
		expect(harness.eventsOfType("compaction_end").some((event) => event.contextWindowStarted)).toBe(false);
		expect(harness.eventsOfType("compaction_end").at(-1)).toMatchObject({
			reason: "overflow",
			aborted: false,
			willRetry: false,
			errorMessage: "Context overflow recovery failed: disk full",
		});
		expect(failures).toEqual(["Context overflow recovery failed: disk full"]);
		const sessionFile = sessionManager.getSessionFile();
		if (!sessionFile) throw new Error("Expected a file-backed session");
		expect(
			SessionManager.open(sessionFile)
				.getEntries()
				.some((entry) => entry.type === "context_window"),
		).toBe(false);
	});

	it("reports success when exact boundary bytes persist before the write reports failure", async () => {
		const failures: string[] = [];
		const sessionManager = createFileSession(tempDirs);
		const harness = await createHarness({
			sessionManager,
			extensionFactories: [
				claimBoundary,
				(pi) => {
					pi.on("session_compact_failed", (event) => {
						if (event.errorMessage) failures.push(event.errorMessage);
					});
				},
			],
		});
		harnesses.push(harness);
		const summarizationAuth = forbidSummarizationAuth(harness);
		const persist = sessionManager._persist.bind(sessionManager);
		vi.spyOn(sessionManager, "_persist").mockImplementation((entry: SessionEntry) => {
			persist(entry);
			if (entry.type === "context_window") throw new Error("close failed after write");
		});
		let retryTexts: string[] = [];
		harness.setResponses([
			overflowResponse(),
			(context) => {
				retryTexts = context.messages.map(getMessageText);
				return fauxAssistantMessage("continued after reconciled write");
			},
		]);

		await harness.session.prompt("x".repeat(600_000));

		expect(summarizationAuth).not.toHaveBeenCalled();
		expect(failures).toEqual([]);
		expect(harness.faux.state.callCount).toBe(2);
		expect(retryTexts).toEqual([expect.stringContaining("handoff after overflow")]);
		expect(sessionManager.getEntries().filter((entry) => entry.type === "context_window")).toHaveLength(1);
		expect(harness.eventsOfType("compaction_end").at(-1)).toMatchObject({
			reason: "overflow",
			aborted: false,
			willRetry: true,
			contextWindowStarted: true,
		});
		const sessionFile = sessionManager.getSessionFile();
		if (!sessionFile) throw new Error("Expected a file-backed session");
		const reopened = SessionManager.open(sessionFile);
		const reopenedBoundaries = reopened.getEntries().filter((entry) => entry.type === "context_window");
		expect(reopenedBoundaries).toHaveLength(1);
		expect(reopenedBoundaries[0]).toMatchObject({ handoff: "handoff after overflow" });
		expect(reopened.buildSessionContext().messages.map(getMessageText)).toEqual([
			expect.stringContaining("handoff after overflow"),
			"continued after reconciled write",
		]);
	});
});
