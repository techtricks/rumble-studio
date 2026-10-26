// deno-lint-ignore-file no-explicit-any
// Dashboard gate: every path is locked behind a single shared password.
// The password lives in the DASH_PASSWORD environment variable; without it
// the site stays open (fail-open) so nobody can ever be locked out.
// The cookie is a SHA-256 of the password — as soon as the password
// changes, every saved session dies at once.

export const config = { path: "/*" };

async function tokenFor(secret: string): Promise<string> {
  const h = await crypto.subtle.digest(
    "SHA-256",
    new TextEncoder().encode("ytauto-dash:" + secret),
  );
  return [...new Uint8Array(h)]
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

function loginPage(error: boolean, status = 200): Response {
  const msg = error
    ? `<p class="err">गलत password — दुबारा try करें</p>`
    : "";
  const html = `<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>🔒 Rumble Studio — private</title>
<style>
  body { background:#071312; color:#d7efe9; font:15px/1.6 -apple-system,'Segoe UI',sans-serif;
         display:grid; place-items:center; min-height:100vh; margin:0 }
  body::before { content:''; position:fixed; inset:0; pointer-events:none;
         background:radial-gradient(700px 420px at 70% -10%, rgba(45,212,191,.09), transparent 60%) }
  .card { background:#0d211f; border:1px solid #16413c; border-radius:16px; padding:34px 30px;
          width:min(360px, 88vw); box-shadow:0 20px 60px rgba(0,0,0,.5); position:relative }
  .logo { width:52px; height:52px; border-radius:13px; display:grid; place-items:center; font-size:24px;
          background:linear-gradient(135deg,#134e4a,#0f766e); box-shadow:0 0 18px rgba(45,212,191,.35);
          margin-bottom:16px }
  h1 { font-size:19px; margin:0 0 4px } small { color:#6d948c }
  input { width:100%; padding:12px 14px; margin:18px 0 12px; border-radius:10px; border:1px solid #1d544d;
          background:#0a1a19; color:#d7efe9; font-size:15px; outline:none }
  input:focus { border-color:#2dd4bf; box-shadow:0 0 14px rgba(45,212,191,.25) }
  button { width:100%; padding:12px; border-radius:10px; border:0; cursor:pointer; font-weight:700;
           background:linear-gradient(90deg,#0f766e,#14b8a6); color:#04120f; font-size:15px }
  .err { color:#f87171; font-weight:600; margin:10px 0 0 }
</style></head><body>
<form class="card" method="POST" action="/__dash_login">
  <div class="logo">🔒</div>
  <h1>Rumble Studio</h1>
  <small>Indian History pipeline — private dashboard</small>
  <input type="password" name="pw" placeholder="Password" autocomplete="current-password" autofocus>
  <button type="submit">Open dashboard →</button>
  ${msg}
</form></body></html>`;
  return new Response(html, {
    status,
    headers: { "content-type": "text/html; charset=utf-8" },
  });
}

export default async (request: Request, context: any) => {
  const secret = Deno.env.get("DASH_PASSWORD");
  if (!secret) return context.next();          // no password configured: open

  const url = new URL(request.url);

  if (request.method === "POST" && url.pathname === "/__dash_login") {
    const form = await request.formData().catch(() => null);
    const pw = String(form?.get("pw") ?? "");
    if (pw && pw === secret) {
      return new Response(null, {
        status: 303,
        headers: {
          Location: "/",
          "Set-Cookie": `dash=${await tokenFor(secret)}; Max-Age=2592000; ` +
            "Path=/; HttpOnly; Secure; SameSite=Lax",
        },
      });
    }
    return loginPage(true, 401);
  }

  const m = /(?:^|;\s*)dash=([a-f0-9]{64})/.exec(request.headers.get("cookie") ?? "");
  if (m && m[1] === await tokenFor(secret)) return context.next();

  return loginPage(false);
};
