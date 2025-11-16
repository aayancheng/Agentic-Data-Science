import { rm, mkdir, cp } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const projectRoot = path.resolve(__dirname, '..')
const distDir = path.join(projectRoot, 'dist')

async function build() {
  await rm(distDir, { recursive: true, force: true })
  await mkdir(distDir, { recursive: true })
  await cp(path.join(projectRoot, 'index.html'), path.join(distDir, 'index.html'))
  await cp(path.join(projectRoot, 'src'), path.join(distDir, 'src'), { recursive: true })
  console.log('Static assets copied to dist/.')
}

build().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
