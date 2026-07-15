import assert from 'node:assert/strict'
import { test } from 'node:test'

import { assertLockedCommit } from './git-install-lock.mjs'

const PACKAGE_NAME = '@medcognetics/dicom-preprocessing'
const COMMIT_SHA = '83b01149f80b93ac37259da2b35c7a90c28ad4cf'
const HTTPS_DEPENDENCY = `git+https://github.com/medcognetics/dicom-preprocessing.git#${COMMIT_SHA}`
const NPM_RESOLVED_DEPENDENCY = `git+ssh://git@github.com/medcognetics/dicom-preprocessing.git#${COMMIT_SHA}`

test('accepts npm GitHub URL normalization when the full commit remains pinned', () => {
  const lockfile = {
    packages: {
      '': { dependencies: { [PACKAGE_NAME]: HTTPS_DEPENDENCY } },
      [`node_modules/${PACKAGE_NAME}`]: { resolved: NPM_RESOLVED_DEPENDENCY },
    },
  }

  assert.doesNotThrow(() => assertLockedCommit(lockfile, HTTPS_DEPENDENCY, COMMIT_SHA))
})

test('rejects a resolved dependency for another repository', () => {
  const lockfile = {
    packages: {
      '': { dependencies: { [PACKAGE_NAME]: HTTPS_DEPENDENCY } },
      [`node_modules/${PACKAGE_NAME}`]: {
        resolved: `git+ssh://git@github.com/medcognetics/another-repository.git#${COMMIT_SHA}`,
      },
    },
  }

  assert.throws(() => assertLockedCommit(lockfile, HTTPS_DEPENDENCY, COMMIT_SHA))
})

test('rejects a resolved dependency at another commit', () => {
  const otherCommit = '0715ee9a32de7c6e67e0c62d093f83fff6aa6c72'
  const lockfile = {
    packages: {
      '': { dependencies: { [PACKAGE_NAME]: HTTPS_DEPENDENCY } },
      [`node_modules/${PACKAGE_NAME}`]: {
        resolved: `git+ssh://git@github.com/medcognetics/dicom-preprocessing.git#${otherCommit}`,
      },
    },
  }

  assert.throws(() => assertLockedCommit(lockfile, HTTPS_DEPENDENCY, COMMIT_SHA))
})
