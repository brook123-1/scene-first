import { Container, getContainer } from "@cloudflare/containers";
import { env } from "cloudflare:workers";

declare global {
  namespace Cloudflare {
    interface Env {
      SCENE_FIRST_CONTAINER: DurableObjectNamespace<SceneFirstContainer>;
      ARK_API_KEY?: string;
      FAL_KEY?: string;
      SCENE_FIRST_LOCAL_MASTER?: string;
      SCENE_FIRST_TEST_PASSWORD?: string;
      SCENE_FIRST_ADMIN_RESTART_TOKEN?: string;
    }
  }
}

type SceneFirstEnv = Cloudflare.Env;

export class SceneFirstContainer extends Container<SceneFirstEnv> {
  defaultPort = 8765;
  sleepAfter = "10m";
  pingEndpoint = "localhost/api/health";
  enableInternet = true;

  envVars = {
    SCENE_FIRST_PUBLIC_MODE: "1",
    MAX_PARALLEL_PERSON_EDITS: "2",
    ARK_IMAGE_BASE_URL:
      "https://ark.cn-beijing.volces.com/api/plan/v3/images/generations",
    ARK_IMAGE_MODEL: "doubao-seedream-5.0-lite",
    FAL_IMAGE_MODEL: "fal-ai/nano-banana-pro/edit",
    ARK_API_KEY: env.ARK_API_KEY ?? "",
    FAL_KEY: env.FAL_KEY ?? "",
    SCENE_FIRST_LOCAL_MASTER: env.SCENE_FIRST_LOCAL_MASTER ?? "0",
  };
}

const ACCESS_COOKIE = "scene_first_staging_access";

async function sha256Hex(value: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function isAuthorized(request: Request, password?: string): Promise<boolean> {
  if (!password) return true;
  const cookie = request.headers.get("cookie") ?? "";
  const cookieToken = await sha256Hex(`cookie:${password}`);
  if (cookie.split(";").some((part) => part.trim() === `${ACCESS_COOKIE}=${cookieToken}`)) return true;
  const authorization = request.headers.get("authorization") ?? "";
  if (!authorization.startsWith("Basic ")) return false;
  try {
    const decoded = atob(authorization.slice(6));
    const separator = decoded.indexOf(":");
    if (separator < 0 || decoded.slice(0, separator) !== "scene") return false;
    return (await sha256Hex(decoded.slice(separator + 1))) === await sha256Hex(password);
  } catch {
    return false;
  }
}

function isAdminRestartAuthorized(request: Request, token?: string): boolean {
  if (!token) return false;
  return request.headers.get("authorization") === `Bearer ${token}`;
}

export default {
  async fetch(request: Request, workerEnv: SceneFirstEnv): Promise<Response> {
    if (!await isAuthorized(request, workerEnv.SCENE_FIRST_TEST_PASSWORD)) {
      return new Response("This private staging build requires the test access code.", {
        status: 401,
        headers: { "WWW-Authenticate": 'Basic realm="Scene First staging"' },
      });
    }
    const container = getContainer(
      workerEnv.SCENE_FIRST_CONTAINER,
      "scene-first-primary",
    );
    const url = new URL(request.url);
    if (url.pathname === "/__ops/restart-container") {
      if (request.method !== "POST" || !isAdminRestartAuthorized(request, workerEnv.SCENE_FIRST_ADMIN_RESTART_TOKEN)) {
        return new Response("Not found", { status: 404 });
      }
      await container.destroy();
      return Response.json({ ok: true, status: "container_destroyed" });
    }
    const response = await container.fetch(request);
    if (workerEnv.SCENE_FIRST_TEST_PASSWORD && request.headers.has("authorization")) {
      const wrapped = new Response(response.body, response);
      const cookieToken = await sha256Hex(`cookie:${workerEnv.SCENE_FIRST_TEST_PASSWORD}`);
      wrapped.headers.append(
        "Set-Cookie",
        `${ACCESS_COOKIE}=${cookieToken}; Max-Age=28800; Path=/; Secure; HttpOnly; SameSite=Strict`,
      );
      return wrapped;
    }
    return response;
  },
};
