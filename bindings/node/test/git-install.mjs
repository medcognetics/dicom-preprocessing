import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const PACKAGE_NAME = '@medcognetics/dicom-preprocessing'
const REPOSITORY_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..')
const FULL_COMMIT_SHA = /^[0-9a-f]{40}$/u
const EXPECTED_BINARY_BY_HOST = {
  'darwin:arm64': 'dicom_preprocessing_node.darwin-arm64.node',
  'darwin:x64': 'dicom_preprocessing_node.darwin-x64.node',
  'linux:x64': 'dicom_preprocessing_node.linux-x64-gnu.node',
  'win32:x64': 'dicom_preprocessing_node.win32-x64-msvc.node',
}

function run(command, args, cwd, options = {}) {
  return execFileSync(command, args, {
    cwd,
    env: { ...process.env, npm_config_foreground_scripts: 'true' },
    stdio: 'inherit',
    ...options,
  })
}

function git(args, cwd, options = {}) {
  return run('git', args, cwd, options)
}

function trackedAndUntrackedFiles() {
  return execFileSync('git', ['ls-files', '--cached', '--others', '--exclude-standard', '-z'], {
    cwd: REPOSITORY_ROOT,
    encoding: 'utf8',
  })
    .split('\0')
    .filter(Boolean)
}

function snapshotRepository() {
  const snapshot = mkdtempSync(join(tmpdir(), 'dicom-preprocessing-git-source-'))

  for (const repositoryPath of trackedAndUntrackedFiles()) {
    const source = join(REPOSITORY_ROOT, repositoryPath)
    if (!existsSync(source)) {
      continue
    }

    const destination = join(snapshot, repositoryPath)
    mkdirSync(dirname(destination), { recursive: true })
    copyFileSync(source, destination)
  }

  git(['init', '-b', 'main'], snapshot)
  git(['add', '--all'], snapshot)
  git(
    [
      '-c',
      'user.name=dicom-preprocessing tests',
      '-c',
      'user.email=tests@invalid.example',
      'commit',
      '-m',
      'Create Git installation fixture',
    ],
    snapshot,
  )

  const sha = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: snapshot, encoding: 'utf8' }).trim()
  assert.match(sha, FULL_COMMIT_SHA)
  const gitUrl = pathToFileURL(snapshot).href.replace(/^file:/u, 'git+file:')
  return { dependency: `${gitUrl}#${sha}`, sha, snapshot }
}

function dependencySource() {
  const repositoryUrl = process.env.DICOM_PREPROCESSING_GIT_URL
  if (!repositoryUrl) {
    return snapshotRepository()
  }

  assert.match(repositoryUrl, /^git\+https:\/\//u)
  const sha = process.env.DICOM_PREPROCESSING_GIT_SHA
  assert.match(sha ?? '', FULL_COMMIT_SHA)
  return { dependency: `${repositoryUrl.replace(/#.*$/u, '')}#${sha}`, sha, snapshot: undefined }
}

function writeConsumerVerification(consumer, expectedBinary) {
  const script = `
const assert = require('node:assert/strict')
const { readdirSync } = require('node:fs')
const { dirname, join } = require('node:path')

const packageName = ${JSON.stringify(PACKAGE_NAME)}
const expectedBinary = ${JSON.stringify(expectedBinary)}
const packageRoot = dirname(require.resolve(packageName + '/package.json'))
const api = require(packageName)
const binaries = readdirSync(join(packageRoot, 'bindings', 'node'))
  .filter((name) => name.endsWith('.node'))

assert.equal(typeof api.prepareDicom, 'function')
assert.equal(typeof api.renderDisplayFrame, 'function')
assert.deepEqual(binaries, [expectedBinary])
console.log(JSON.stringify({ packageRoot, binary: binaries[0], platform: process.platform, arch: process.arch }))
`
  const verificationPath = join(consumer, 'verify.cjs')
  writeFileSync(verificationPath, script)
  return verificationPath
}

function assertLockedCommit(consumer, dependency, sha) {
  const lockfile = JSON.parse(readFileSync(join(consumer, 'package-lock.json'), 'utf8'))
  assert.equal(lockfile.packages[''].dependencies[PACKAGE_NAME], dependency)
  const installedPackage = lockfile.packages[`node_modules/${PACKAGE_NAME}`]
  assert.equal(installedPackage.resolved, dependency)
  assert.ok(installedPackage.resolved.endsWith(`#${sha}`))
}

const expectedBinary = EXPECTED_BINARY_BY_HOST[`${process.platform}:${process.arch}`]
assert.ok(expectedBinary, `Unsupported integration-test host: ${process.platform}:${process.arch}`)

const { dependency, sha, snapshot } = dependencySource()
const consumer = mkdtempSync(join(tmpdir(), 'dicom-preprocessing-git-consumer-'))
const verificationPath = writeConsumerVerification(consumer, expectedBinary)
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm'

try {
  writeFileSync(
    join(consumer, 'package.json'),
    `${JSON.stringify(
      {
        name: 'dicom-preprocessing-git-consumer',
        version: '1.0.0',
        private: true,
        dependencies: { [PACKAGE_NAME]: dependency },
      },
      null,
      2,
    )}\n`,
  )

  run(npm, ['install', '--omit=optional'], consumer)
  assertLockedCommit(consumer, dependency, sha)
  run(process.execPath, [verificationPath], consumer)

  rmSync(join(consumer, 'node_modules'), { recursive: true, force: true })
  run(npm, ['ci', '--omit=optional'], consumer)
  assertLockedCommit(consumer, dependency, sha)
  run(process.execPath, [verificationPath], consumer)
} finally {
  rmSync(consumer, { recursive: true, force: true })
  if (snapshot) {
    rmSync(snapshot, { recursive: true, force: true })
  }
}
