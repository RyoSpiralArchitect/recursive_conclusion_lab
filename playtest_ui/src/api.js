export async function fetchJson(url, options) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options?.headers || {}),
    },
    ...options,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    let payloadDetail = null;
    try {
      const payload = await response.json();
      if (payload?.detail) {
        payloadDetail = payload.detail;
        detail = typeof payloadDetail === "string"
          ? payloadDetail
          : payloadDetail.message || detail;
      }
    } catch {}
    const error = new Error(detail);
    error.status = response.status;
    error.detail = payloadDetail;
    error.code = payloadDetail && typeof payloadDetail === "object"
      ? payloadDetail.code
      : undefined;
    throw error;
  }
  return response.json();
}
