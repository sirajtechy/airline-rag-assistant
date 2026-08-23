import { useEffect, useRef, useState } from "react";
import { AskResponse, Config, Examples, ask, getConfig, getExamples } from "./api";

type Turn = { question: string; response?: AskResponse; error?: string };

const ROUTE_LABEL: Record<string, string> = {
  passenger: "Passenger — Contract of Carriage",
  cargo: "Cargo — Shipping Rules Tariff",
  financial: "Financial — SEC 10-K filings",
};

/** Renders [1]-style citation markers as superscript chips linked to the sources. */
function AnswerText({ text }: { text: string }) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <p className="answer">
      {parts.map((part, i) => {
        const match = part.match(/^\[(\d+)\]$/);
        if (!match) return <span key={i}>{part}</span>;
        return (
          <sup key={i} className="cite" title={`Source ${match[1]}`}>
            {match[1]}
          </sup>
        );
      })}
    </p>
  );
}

function Metrics({ r }: { r: AskResponse }) {
  return (
    <div className="metrics">
      <span>retrieval {r.retrieval_ms.toFixed(0)} ms</span>
      {r.generation_ms > 0 && <span>generation {(r.generation_ms / 1000).toFixed(1)} s</span>}
      <span>{r.model}</span>
      {!r.blocked && (
        <span className={r.citations_valid ? "ok" : "warn"}>
          {r.citations_valid ? "citations verified" : "citations unverified"}
        </span>
      )}
    </div>
  );
}

export default function App() {
  const [config, setConfig] = useState<Config | null>(null);
  const [examples, setExamples] = useState<Examples>({});
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getConfig().then(setConfig).catch(() => setConfig(null));
    getExamples().then(setExamples).catch(() => setExamples({}));
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, busy]);

  async function submit(q: string) {
    const text = q.trim();
    if (!text || busy) return;
    setQuestion("");
    setBusy(true);
    setTurns((t) => [...t, { question: text }]);
    try {
      const response = await ask(text);
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, response } : turn)));
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setTurns((t) => t.map((turn, i) => (i === t.length - 1 ? { ...turn, error: message } : turn)));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="app">
      <header>
        <div>
          <h1>Delta Air Lines Support Assistant</h1>
          <p className="sub">
            Answers strictly from Delta's Contract of Carriage, Cargo tariff and SEC filings —
            with the exact rule cited.
          </p>
        </div>
        {config && (
          <div className="config" title="Configuration selected by the ablation study">
            <code>{config.retrieval.chunking}</code>
            <code>{config.retrieval.embedding}</code>
            <code>{config.retrieval.mode} / {config.retrieval.fusion}</code>
            <code>{config.chunks} chunks</code>
          </div>
        )}
      </header>

      <main>
        {turns.length === 0 && (
          <div className="empty">
            {Object.entries(examples).map(([group, items]) => (
              <section key={group}>
                <h3>{group === "guardrails" ? "out of scope (try these too)" : group}</h3>
                {items.map((item) => (
                  <button key={item} onClick={() => submit(item)} disabled={busy}>
                    {item}
                  </button>
                ))}
              </section>
            ))}
          </div>
        )}

        {turns.map((turn, i) => (
          <div key={i} className="turn">
            <div className="q">{turn.question}</div>

            {turn.error && <div className="a error">{turn.error}</div>}

            {turn.response && (
              <div className={`a ${turn.response.blocked ? "blocked" : ""}`}>
                {turn.response.route && !turn.response.blocked && (
                  <div className={`route route-${turn.response.route}`}>
                    {ROUTE_LABEL[turn.response.route] ?? turn.response.route}
                  </div>
                )}
                {turn.response.blocked && <div className="route route-blocked">out of scope</div>}

                <AnswerText text={turn.response.answer} />

                {turn.response.sources.length > 0 && (
                  <details open>
                    <summary>{turn.response.sources.length} sources</summary>
                    <ol className="sources">
                      {turn.response.sources.map((s) => (
                        <li key={s.n}>
                          <span className="cite-n">{s.n}</span> {s.citation}
                        </li>
                      ))}
                    </ol>
                  </details>
                )}
                <Metrics r={turn.response} />
              </div>
            )}

            {!turn.response && !turn.error && <div className="a thinking">searching Delta's documents…</div>}
          </div>
        ))}
        <div ref={endRef} />
      </main>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit(question);
        }}
      >
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask about tickets, refunds, cargo shipping or Delta's financials…"
          disabled={busy}
        />
        <button type="submit" disabled={busy || !question.trim()}>
          {busy ? "…" : "Ask"}
        </button>
      </form>
    </div>
  );
}
