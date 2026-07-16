import assert from 'node:assert/strict'

const PACKAGE_NAME = '@medcognetics/dicom-preprocessing'

function gitRepository(value) {
  const url = new URL(value.replace(/^git\+/u, '').replace(/#.*$/u, ''))
  const path = url.pathname.replace(/\/+$/u, '').replace(/\.git$/u, '')
  return `${url.host.toLowerCase()}${path}`
}

export function assertLockedCommit(lockfile, dependency, sha) {
  assert.equal(lockfile.packages[''].dependencies[PACKAGE_NAME], dependency)
  const installedPackage = lockfile.packages[`node_modules/${PACKAGE_NAME}`]
  assert.equal(gitRepository(installedPackage.resolved), gitRepository(dependency))
  assert.ok(installedPackage.resolved.endsWith(`#${sha}`))
}
