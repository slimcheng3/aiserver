import { useEffect, useState } from "react";
import { api } from "./api";
import type { MetaInfo } from "./types";
import { BuildPage } from "./pages/BuildPage";
import { EnginesPage } from "./pages/EnginesPage";
import { NewPackagePage } from "./pages/NewPackagePage";
import { PackageDetailPage } from "./pages/PackageDetailPage";
import { PackagesPage } from "./pages/PackagesPage";
import { RuntimePage } from "./pages/RuntimePage";
import { TestPage } from "./pages/TestPage";

type Route =
  | { name: "engines" }
  | { name: "build"; model: string | null; image: string | null }
  | { name: "new"; model: string | null }
  | { name: "packages" }
  | { name: "package"; id: string }
  | { name: "runtime"; packageId: string | null }
  | { name: "test"; packageId: string | null };

function parseHash(hash: string): Route {
  const raw = hash.replace(/^#/, "") || "/";
  const [path, query = ""] = raw.split("?");
  const params = new URLSearchParams(query);
  if (path === "/engines") return { name: "engines" };
  if (path === "/build") return { name: "build", model: params.get("model"), image: params.get("image") };
  if (path === "/" || path === "/models" || path === "/new") return { name: "new", model: params.get("model") };
  if (path === "/packages") return { name: "packages" };
  if (path === "/runtime") return { name: "runtime", packageId: params.get("package") };
  if (path === "/test") return { name: "test", packageId: params.get("package") };
  const match = path.match(/^\/packages\/([^/]+)$/);
  if (match) return { name: "package", id: decodeURIComponent(match[1]) };
  return { name: "new", model: null };
}

export function App() {
  const [route, setRoute] = useState<Route>(() => parseHash(location.hash));
  const [meta, setMeta] = useState<MetaInfo | null>(null);

  useEffect(() => {
    const onHash = () => setRoute(parseHash(location.hash));
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  useEffect(() => {
    window.scrollTo(0, 0);
    const titles: Record<Route["name"], string> = {
      engines: "运行时",
      build: "构建运行时",
      new: "装配",
      packages: "模型包",
      package: "模型包",
      runtime: "部署",
      test: "测试",
    };
    document.title = `${titles[route.name]} · 具身模型交付台`;
  }, [route]);

  useEffect(() => {
    api.meta().then(setMeta).catch(() => setMeta(null));
  }, []);

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <p className="brand-kicker">EMBODY</p>
            <p className="brand-title">具身模型交付台</p>
          </div>
        </div>
        <nav className="side-nav">
          <a className={route.name === "engines" || route.name === "build" ? "side-link active" : "side-link"} href="#/engines">
            运行时
          </a>
          <a className={route.name === "new" ? "side-link active" : "side-link"} href="#/new">
            装配
          </a>
          <a className={route.name === "packages" || route.name === "package" ? "side-link active" : "side-link"} href="#/packages">
            模型包
          </a>
          <a className={route.name === "runtime" ? "side-link active" : "side-link"} href="#/runtime">
            部署
          </a>
          <a className={route.name === "test" ? "side-link active" : "side-link"} href="#/test">
            测试
          </a>
        </nav>
        <div className="side-foot">
          <strong>机体存储</strong>
          {meta ? meta.dataDir : "读取中"}
        </div>
      </aside>
      <main>
        {route.name === "engines" && <EnginesPage />}
        {route.name === "build" && <BuildPage modelId={route.model} image={route.image} />}
        {route.name === "new" && <NewPackagePage initialModel={route.model} />}
        {route.name === "packages" && <PackagesPage />}
        {route.name === "package" && <PackageDetailPage id={route.id} />}
        {route.name === "runtime" && <RuntimePage focus={route.packageId} />}
        {route.name === "test" && <TestPage packageId={route.packageId} />}
      </main>
    </div>
  );
}
