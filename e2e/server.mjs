import http from "node:http";
import { createReadStream, statSync } from "node:fs";
import { extname, join, resolve } from "node:path";

const rootDir = resolve(process.cwd(), "e2e");
const port = Number(process.env.PORT || 5173);

const contentTypeFor = (ext) => {
  switch (ext) {
    case ".html":
      return "text/html; charset=utf-8";
    case ".js":
      return "text/javascript; charset=utf-8";
    case ".css":
      return "text/css; charset=utf-8";
    case ".json":
      return "application/json; charset=utf-8";
    default:
      return "application/octet-stream";
  }
};

const server = http.createServer((req, res) => {
  const url = new URL(req.url ?? "/", `http://localhost:${port}`);
  const pathname = url.pathname === "/" ? "/index.html" : url.pathname;
  const filePath = resolve(join(rootDir, pathname));

  if (!filePath.startsWith(rootDir)) {
    res.statusCode = 403;
    res.end("Forbidden");
    return;
  }

  try {
    const stat = statSync(filePath);
    if (!stat.isFile()) {
      res.statusCode = 404;
      res.end("Not Found");
      return;
    }
    res.statusCode = 200;
    res.setHeader("Content-Type", contentTypeFor(extname(filePath)));
    createReadStream(filePath).pipe(res);
  } catch (err) {
    res.statusCode = 404;
    res.end("Not Found");
  }
});

server.listen(port, () => {
  // eslint-disable-next-line no-console
  console.log(`e2e server: http://localhost:${port}`);
});
