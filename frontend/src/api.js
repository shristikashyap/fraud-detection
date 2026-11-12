const configuredBase = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/+$/, "");
const API_ROOT = configuredBase
  ? configuredBase.endsWith("/api")
    ? configuredBase
    : `${configuredBase}/api`
  : "/api";

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(`${API_ROOT}${path}`, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...options.headers,
      },
    });
  } catch {
    throw new Error("Cannot reach the fraud detection API. Check the backend URL and service status.");
  }

  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    throw new Error(
      body?.error?.message || `The API request failed with status ${response.status}.`,
    );
  }
  return body;
}

export const api = {
  health: () => request("/health"),
  analyze: (payload, idempotencyKey = globalThis.crypto?.randomUUID?.()) =>
    request("/analyze", {
      method: "POST",
      body: JSON.stringify(payload),
      headers: idempotencyKey ? { "Idempotency-Key": idempotencyKey } : {},
    }),
  users: () => request("/users"),
  user: (userId) => request(`/users/${encodeURIComponent(userId)}`),
  alerts: (page = 1, perPage = 50) =>
    request(`/alerts?page=${page}&per_page=${perPage}`),
  transactions: (page = 1, perPage = 20) =>
    request(`/transactions?page=${page}&per_page=${perPage}`),
  reports: () => request("/reports"),
};
