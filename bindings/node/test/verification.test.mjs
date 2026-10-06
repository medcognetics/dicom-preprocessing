import assert from 'node:assert/strict'
import { test } from 'node:test'
import { verifyRuntime } from '../index.js'

test('runtime report passes, is deterministic, and covers every enabled decoder', () => {
  const report = verifyRuntime()
  assert.deepEqual(report, verifyRuntime())
  assert.equal(report.schemaVersion, 1)
  assert.ok(report.cases.length >= 25)
  assert.deepEqual(report.cases.filter(result => !result.passed).map(result => result.id), [])
  assert.ok(report.codecs.length > 0)
  assert.ok(report.codecs.every(codec => codec.passed && codec.requiredCases.length > 0))
  assert.equal(report.passed, true)
})

test('report keys are camelCase and every check is named', () => {
  const report = verifyRuntime()
  assert.ok('transferSyntaxUid' in report.cases[0])
  assert.ok(report.cases.every(result => result.checks.length > 0 && result.checks.every(check => check.name)))
  const registrations = report.cases.filter(result => result.id.startsWith('builtin/decoder-registration/'))
  assert.equal(registrations.length, 5)
  assert.ok(registrations.every(result => result.source === 'builtin' && result.id.endsWith(`/${result.transferSyntaxUid}`)))
})
