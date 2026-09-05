import { appendFileSync, existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { prepareCompaction } from "../../src/core/compaction/compaction.ts";
import { MAX_CONTEXT_HANDOFF_CHARS, SessionManager } from "../../src/core/session-manager.ts";

function messageText(message: unknown): string {
	if (!message || typeof message !== "object" || !("content" in message)) return "";
	const content = (message as { content?: string | Array<{ type: string; text?: string }> }).content;
	if (typeof content === "string") return content;
	return (content ?? [])
		.filter((part) => part.type === "text")
		.map((part) => part.text ?? "")
		.join("\n");
}

describe("SessionManager context windows", () => {
	const tempDirs: string[] = [];

	afterEach(() => {
		vi.restoreAllMocks();
		while (tempDirs.length > 0) {
			rmSync(tempDirs.pop()!, { recursive: true, force: true });
		}
	});

	function createPoisonedSession() {
		const tempDir = mkdtempSync(join(tmpdir(), "pi-context-window-poisoned-"));
		tempDirs.push(tempDir);
		const session = SessionManager.create(tempDir, join(tempDir, "sessions"));
		const labelledId = session.appendMessage({ role: "user", content: "old task", timestamp: 1 });
		session.appendMessage(fauxAssistantMessage("old response", { timestamp: 2 }));
		session.appendLabelChange(labelledId, "checkpoint");
		const sessionFile = session.getSessionFile();
		if (!sessionFile) throw new Error("Expected a file-backed session");
		const persist = session._persist.bind(session);
		vi.spyOn(session, "_persist").mockImplementation((entry) => {
			if (entry.type === "context_window") {
				appendFileSync(sessionFile, `${JSON.stringify(entry)}extra`);
				throw new Error("uncertain write");
			}
			persist(entry);
		});
		expect(() => session.appendContextWindow("must fail", 10)).toThrow("uncertain write");
		vi.restoreAllMocks();
		return { session, sessionFile, labelledId, suspectBytes: readFileSync(sessionFile) };
	}

	it("preserves JSONL history while reopening only the latest context window", () => {
		const tempDir = mkdtempSync(join(tmpdir(), "pi-context-window-"));
		tempDirs.push(tempDir);
		const sessionDir = join(tempDir, "sessions");
		const session = SessionManager.create(tempDir, sessionDir);

		session.appendMessage({ role: "user", content: "old request", timestamp: 1 });
		session.appendMessage(fauxAssistantMessage("old response", { timestamp: 2 }));
		const sessionFile = session.getSessionFile();
		if (!sessionFile) throw new Error("Expected a file-backed session");
		const oldBytes = readFileSync(sessionFile);

		session.appendContextWindow("first handoff", 42);
		session.appendMessage({ role: "user", content: "later request", timestamp: 3 });
		session.appendMessage(fauxAssistantMessage("later response", { timestamp: 4 }));

		const appendedBytes = readFileSync(sessionFile);
		expect(appendedBytes.subarray(0, oldBytes.length)).toEqual(oldBytes);

		const reopened = SessionManager.open(sessionFile, sessionDir);
		expect(reopened.getHeader()?.version).toBe(3);
		expect(reopened.getEntries().map((entry) => entry.type)).toEqual([
			"message",
			"message",
			"context_window",
			"message",
			"message",
		]);
		expect(reopened.buildContextEntries().map((entry) => entry.type)).toEqual([
			"context_window",
			"message",
			"message",
		]);
		const firstWindowTexts = reopened.buildSessionContext().messages.map(messageText);
		expect(firstWindowTexts).toEqual([expect.stringContaining("first handoff"), "later request", "later response"]);
		expect(firstWindowTexts).not.toContain("old request");
		expect(firstWindowTexts).not.toContain("old response");

		reopened.appendContextWindow("second handoff", 12);
		reopened.appendMessage({ role: "user", content: "newest request", timestamp: 5 });

		expect(reopened.buildContextEntries().map((entry) => entry.type)).toEqual(["context_window", "message"]);
		const secondWindowTexts = reopened.buildSessionContext().messages.map(messageText);
		expect(secondWindowTexts).toEqual([expect.stringContaining("second handoff"), "newest request"]);
		expect(secondWindowTexts.join("\n")).not.toContain("first handoff");
	});

	it("repairs an exact boundary missing only its final newline before a following append", () => {
		const tempDir = mkdtempSync(join(tmpdir(), "pi-context-window-framing-"));
		tempDirs.push(tempDir);
		const session = SessionManager.create(tempDir, join(tempDir, "sessions"));

		session.appendMessage({ role: "user", content: "old task", timestamp: 1 });
		session.appendMessage(fauxAssistantMessage("old response", { timestamp: 2 }));
		const sessionFile = session.getSessionFile();
		if (!sessionFile) throw new Error("Expected a file-backed session");
		const oldBytes = readFileSync(sessionFile);
		const persist = session._persist.bind(session);
		vi.spyOn(session, "_persist").mockImplementation((entry) => {
			if (entry.type === "context_window") {
				appendFileSync(sessionFile, JSON.stringify(entry));
				throw new Error("partial write: final newline not written");
			}
			persist(entry);
		});

		const boundaryId = session.appendContextWindow("must survive the next append", 10);
		expect(readFileSync(sessionFile).subarray(0, oldBytes.length)).toEqual(oldBytes);
		expect(readFileSync(sessionFile, "utf8").endsWith("\n")).toBe(true);
		session.appendMessage({ role: "user", content: "new input", timestamp: 3 });

		const reopened = SessionManager.open(sessionFile);
		expect(reopened.getEntries().some((entry) => entry.id === boundaryId)).toBe(true);
		const reopenedText = JSON.stringify(reopened.buildSessionContext());
		expect(reopenedText).toContain("new input");
		expect(reopenedText).not.toContain("old task");
		expect(readFileSync(sessionFile, "utf8")).not.toContain("}{");
	});

	it("persists the first boundary with prior user input before any assistant response", () => {
		const tempDir = mkdtempSync(join(tmpdir(), "pi-context-window-first-turn-"));
		tempDirs.push(tempDir);
		const session = SessionManager.create(tempDir, join(tempDir, "sessions"));
		const userId = session.appendMessage({ role: "user", content: "pending task", timestamp: 1 });
		const sessionFile = session.getSessionFile();
		if (!sessionFile) throw new Error("Expected a file-backed session");
		expect(existsSync(sessionFile)).toBe(false);

		const boundaryId = session.appendContextWindow("explicit recovery record", null);

		expect(existsSync(sessionFile)).toBe(true);
		const contents = readFileSync(sessionFile, "utf8");
		expect(contents.endsWith("\n")).toBe(true);
		expect(
			contents
				.trimEnd()
				.split("\n")
				.map((line) => JSON.parse(line).type),
		).toEqual(["session", "message", "context_window"]);
		const reopened = SessionManager.open(sessionFile);
		expect(reopened.getEntry(userId)).toMatchObject({ message: { role: "user", content: "pending task" } });
		expect(reopened.getEntry(boundaryId)).toMatchObject({
			type: "context_window",
			handoff: "explicit recovery record",
			tokensBefore: null,
		});
	});

	it("poisons the original and reopened managers when boundary readback has an invalid final tail", () => {
		const tempDir = mkdtempSync(join(tmpdir(), "pi-context-window-mismatch-"));
		tempDirs.push(tempDir);
		const session = SessionManager.create(tempDir, join(tempDir, "sessions"));
		session.appendMessage({ role: "user", content: "old task", timestamp: 1 });
		session.appendMessage(fauxAssistantMessage("old response", { timestamp: 2 }));
		const sessionFile = session.getSessionFile();
		if (!sessionFile) throw new Error("Expected a file-backed session");
		const oldBytes = readFileSync(sessionFile);
		const persist = session._persist.bind(session);
		vi.spyOn(session, "_persist").mockImplementation((entry) => {
			if (entry.type === "context_window") {
				appendFileSync(sessionFile, `${JSON.stringify(entry)}extra`);
				throw new Error("uncertain write");
			}
			persist(entry);
		});

		expect(() => session.appendContextWindow("must not be acknowledged", 10)).toThrow("uncertain write");
		expect(session.getEntries().some((entry) => entry.type === "context_window")).toBe(false);
		expect(session.buildSessionContext().messages.map(messageText)).toEqual(["old task", "old response"]);
		expect(readFileSync(sessionFile).subarray(0, oldBytes.length)).toEqual(oldBytes);
		const suspectBytes = readFileSync(sessionFile);
		const originalEntries = session.getEntries();
		const originalLeafId = session.getLeafId();

		expect(() => session.appendMessage({ role: "user", content: "must not be accepted", timestamp: 3 })).toThrow(
			/Session persistence is uncertain/,
		);
		expect(session.getEntries()).toEqual(originalEntries);
		expect(session.getLeafId()).toBe(originalLeafId);
		expect(readFileSync(sessionFile)).toEqual(suspectBytes);

		const reopened = SessionManager.open(sessionFile);
		expect(readFileSync(sessionFile)).toEqual(suspectBytes);
		const reopenedEntries = reopened.getEntries();
		const reopenedLeafId = reopened.getLeafId();
		expect(() =>
			reopened.appendMessage({ role: "user", content: "must still not be accepted", timestamp: 4 }),
		).toThrow(/Session persistence is uncertain/);
		expect(reopened.getEntries()).toEqual(reopenedEntries);
		expect(reopened.getLeafId()).toBe(reopenedLeafId);
		expect(readFileSync(sessionFile)).toEqual(suspectBytes);
	});

	it("blocks poisoned files loaded by an in-memory manager without changing its aggregate", () => {
		const { sessionFile, labelledId, suspectBytes } = createPoisonedSession();
		const session = SessionManager.inMemory();
		session.setSessionFile(sessionFile);
		const entries = session.getEntries();
		const leafId = session.getLeafId();
		const context = session.buildSessionContext();
		const label = session.getLabel(labelledId);

		expect(() => session.appendMessage({ role: "user", content: "must be blocked", timestamp: 3 })).toThrow(
			/Session persistence is uncertain/,
		);
		expect(session.getEntries()).toEqual(entries);
		expect(session.getLeafId()).toBe(leafId);
		expect(session.buildSessionContext()).toEqual(context);
		expect(session.getLabel(labelledId)).toBe(label);
		expect(readFileSync(sessionFile)).toEqual(suspectBytes);
	});

	it("rejects a poisoned branch summary before changing the active branch", () => {
		const { session, sessionFile, labelledId, suspectBytes } = createPoisonedSession();
		const entries = session.getEntries();
		const leafId = session.getLeafId();
		const context = session.buildSessionContext();
		const label = session.getLabel(labelledId);

		expect(() => session.branchWithSummary(null, "must be blocked")).toThrow(/Session persistence is uncertain/);
		expect(session.getEntries()).toEqual(entries);
		expect(session.getLeafId()).toBe(leafId);
		expect(session.buildSessionContext()).toEqual(context);
		expect(session.getLabel(labelledId)).toBe(label);
		expect(readFileSync(sessionFile)).toEqual(suspectBytes);
	});

	it("remains writable when failed boundary persistence leaves exact pre-boundary bytes", () => {
		const tempDir = mkdtempSync(join(tmpdir(), "pi-context-window-clean-failure-"));
		tempDirs.push(tempDir);
		const session = SessionManager.create(tempDir, join(tempDir, "sessions"));
		session.appendMessage({ role: "user", content: "old task", timestamp: 1 });
		session.appendMessage(fauxAssistantMessage("old response", { timestamp: 2 }));
		const sessionFile = session.getSessionFile();
		if (!sessionFile) throw new Error("Expected a file-backed session");
		const oldBytes = readFileSync(sessionFile);
		const persist = session._persist.bind(session);
		vi.spyOn(session, "_persist").mockImplementation((entry) => {
			if (entry.type === "context_window") throw new Error("write rejected before append");
			persist(entry);
		});

		expect(() => session.appendContextWindow("must not be acknowledged", 10)).toThrow("write rejected before append");
		expect(readFileSync(sessionFile)).toEqual(oldBytes);
		const followingId = session.appendMessage({
			role: "user",
			content: "accepted after clean rollback",
			timestamp: 3,
		});

		const reopened = SessionManager.open(sessionFile);
		expect(reopened.getEntry(followingId)).toMatchObject({
			type: "message",
			message: { role: "user", content: "accepted after clean rollback" },
		});
	});

	it("keeps lazy persistence writable when failed first boundary leaves the file absent", () => {
		const tempDir = mkdtempSync(join(tmpdir(), "pi-context-window-clean-lazy-failure-"));
		tempDirs.push(tempDir);
		const session = SessionManager.create(tempDir, join(tempDir, "sessions"));
		session.appendMessage({ role: "user", content: "initial input", timestamp: 1 });
		const sessionFile = session.getSessionFile();
		if (!sessionFile) throw new Error("Expected a file-backed session");
		expect(existsSync(sessionFile)).toBe(false);
		const persist = session._persist.bind(session);
		vi.spyOn(session, "_persist").mockImplementation((entry) => {
			if (entry.type === "context_window") throw new Error("write rejected before create");
			persist(entry);
		});

		expect(() => session.appendContextWindow("must not be acknowledged", null)).toThrow(
			"write rejected before create",
		);
		expect(existsSync(sessionFile)).toBe(false);
		const followingId = session.appendMessage({ role: "user", content: "retry after clean failure", timestamp: 2 });
		session.appendMessage(fauxAssistantMessage("retry persisted", { timestamp: 3 }));

		const reopened = SessionManager.open(sessionFile);
		expect(reopened.getEntry(followingId)).toMatchObject({
			type: "message",
			message: { role: "user", content: "retry after clean failure" },
		});
	});

	it("rejects an oversized handoff without changing the active projection", () => {
		const session = SessionManager.inMemory();
		session.appendMessage({ role: "user", content: "still active", timestamp: 1 });
		const activeBefore = session.buildSessionContext();

		expect(() => session.appendContextWindow("h".repeat(MAX_CONTEXT_HANDOFF_CHARS + 1), null)).toThrow(
			`Context window handoff exceeds ${MAX_CONTEXT_HANDOFF_CHARS} character limit`,
		);
		expect(session.getEntries().filter((entry) => entry.type === "context_window")).toHaveLength(0);
		expect(session.buildSessionContext()).toEqual(activeBefore);
	});

	it("keeps later compaction preparation and projection inside the latest window", () => {
		const session = SessionManager.inMemory();
		const oldRequestId = session.appendMessage({ role: "user", content: "pre-boundary secret", timestamp: 1 });
		session.appendMessage(fauxAssistantMessage("pre-boundary answer", { timestamp: 2 }));
		session.appendCompaction("pre-boundary summary", oldRequestId, 100);
		session.appendContextWindow("current handoff", 100);
		session.appendMessage({ role: "user", content: "current request one", timestamp: 3 });
		session.appendMessage(fauxAssistantMessage("current answer one", { timestamp: 4 }));
		session.appendMessage({ role: "user", content: "current request two", timestamp: 5 });
		session.appendMessage(fauxAssistantMessage("current answer two", { timestamp: 6 }));

		const preparation = prepareCompaction(session.getBranch(), {
			enabled: true,
			reserveTokens: 100,
			keepRecentTokens: 1,
		});
		if (!preparation) throw new Error("Expected compaction preparation");
		const preparedText = JSON.stringify({
			messagesToSummarize: preparation.messagesToSummarize,
			turnPrefixMessages: preparation.turnPrefixMessages,
			previousSummary: preparation.previousSummary,
		});
		expect(preparedText).toContain("current handoff");
		expect(preparedText).not.toContain("pre-boundary secret");
		expect(preparedText).not.toContain("pre-boundary answer");
		expect(preparedText).not.toContain("pre-boundary summary");
		expect(preparation.previousSummary).toBeUndefined();

		session.appendCompaction("current-window summary", preparation.firstKeptEntryId, preparation.tokensBefore);
		const activeText = JSON.stringify(session.buildSessionContext().messages);
		expect(activeText).toContain("current-window summary");
		expect(activeText).not.toContain("pre-boundary secret");
		expect(activeText).not.toContain("pre-boundary answer");
		expect(activeText).not.toContain("pre-boundary summary");
	});
});
