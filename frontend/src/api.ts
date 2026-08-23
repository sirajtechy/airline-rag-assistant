const BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export type Source = {
  n: number;
  citation: string;
  doc_id: string;
  business_line: string;
  locators: string[];
  score: number;
};

export type AskResponse = {
  question: string;
  answer: string;
  route: string | null;
  blocked: boolean;
  reason: string | null;
  sources: Source[];
  retrieval_ms: number;
  generation_ms: number;
  model: string;
  citations_valid: boolean;
};

export type Config = {
  retrieval: Record<string, string>;
  generator: string;
  business_lines: string[];
  documents: { doc_id: string; title: string; business_line: string }[];
  chunks: number;
};

export type Examples = Record<string, string[]>;

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`${path} failed: ${res.status}`);
  return res.json() as Promise<T>;
}

export const getConfig = () => get<Config>("/config");
export const getExamples = () => get<Examples>("/examples");

export async function ask(question: string): Promise<AskResponse> {
  const res = await fetch(`${BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Request failed (${res.status}): ${detail}`);
  }
  return res.json();
}
