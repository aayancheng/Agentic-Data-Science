import http from 'node:http'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { readFile, stat } from 'node:fs/promises'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const projectRoot = path.resolve(__dirname, '..')
const port = process.env.PORT || 5173

const mimeTypes = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
}

const server = http.createServer(async (req, res) => {
  const urlPath = req.url && req.url !== '/' ? req.url : '/index.html'
  const filePath = path.join(projectRoot, urlPath)
  try {
    const fileStat = await stat(filePath)
    if (fileStat.isDirectory()) {
      res.writeHead(403)
      res.end('Directory access is not allowed.')
      return
    }
    const ext = path.extname(filePath)
    res.setHeader('Content-Type', mimeTypes[ext] || 'application/octet-stream')
    const file = await readFile(filePath)
    res.writeHead(200)
    res.end(file)
  } catch (err) {
    res.writeHead(404)
    res.end('Not found')
  }
})

server.listen(port, () => {
  console.log(`Development server running at http://localhost:${port}`)
})
