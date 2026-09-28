import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import { verifyRuntime } from '../index.js'

function fixtureCase() {
  return {
    id: 'example/fixture',
    dicomBytes: readFileSync(new URL('../../../fixtures/verification/explicit.dcm', import.meta.url)),
    transferSyntaxUid: '1.2.840.10008.1.2.1',
    tags: [{ path: [{ group: 0x28, element: 0x10 }], vr: 'US', values: ['8'] }],
    frames: [{ width: 8, height: 8, samplesPerPixel: 1, planarConfiguration: 0, sampleType: 'u8', values: Array.from({ length: 64 }, (_, i) => i * 3) }],
  }
}

test('runtime report is deterministic and covers every enabled decoder', () => {
  const report = verifyRuntime()
  assert.deepEqual(report, verifyRuntime())
  assert.equal(report.schemaVersion, 1)
  assert.ok(report.cases.length >= 25)
  assert.ok(report.codecs.every(codec => codec.requiredCases.length > 0))
  assert.equal(report.passed, report.cases.every(result => result.passed))
})

test('extension fixtures and custom integrations have distinct coverage', () => {
  const fixture = fixtureCase()
  let calls = 0
  const report = verifyRuntime({
    cases: [fixture],
    tests: [{ id: 'example/custom', run: () => { calls++; return [{ name: 'decode', passed: true }] } }],
    codecs: [{ id: 'example/codecs', requiredCases: { [fixture.transferSyntaxUid]: [fixture.id], '1.2.3': ['example/custom'] } }],
  })
  assert.equal(calls, 1)
  assert.ok(report.cases.slice(-2).every(result => result.passed))
  const coverage = report.codecs.filter(codec => codec.id === 'example/codecs')
  assert.ok(coverage.every(codec => codec.passed))
  assert.equal(coverage.find(codec => codec.transferSyntaxUid === '1.2.3').sharedLibraryVerified, false)
  assert.equal(coverage.find(codec => codec.transferSyntaxUid === fixture.transferSyntaxUid).sharedLibraryVerified, true)
})

test('invalid declarations fail before callbacks', () => {
  let called = false
  const tests = [{ id: 'example/callback', run: () => { called = true; return [] } }]
  for (const id of ['builtin/override', 'invalid', 'example/callback']) {
    assert.throws(() => verifyRuntime({ cases: [{ ...fixtureCase(), id }], tests }))
  }
  const fixture = fixtureCase()
  fixture.frames[0].values.pop()
  assert.throws(() => verifyRuntime({ cases: [fixture], tests }))
  assert.throws(() => verifyRuntime({ codecs: [{ id: 'example/codec', requiredCases: { '1.2.3': ['missing/case'] } }], tests }))
  assert.throws(() => verifyRuntime({ tests: [{ id: 'example/not-callable', run: 42 }, ...tests] }))
  assert.equal(called, false)
})

test('wrong pixels and malformed bytes are report failures', () => {
  for (const corrupt of [false, true]) {
    const fixture = fixtureCase()
    if (corrupt) fixture.dicomBytes = Buffer.from('invalid DICOM')
    else fixture.frames[0].values[0]++
    const report = verifyRuntime({ cases: [fixture] })
    assert.equal(report.passed, false)
    assert.equal(report.cases.at(-1).passed, false)
  }
})

test('callback errors, empty results, promises and malformed checks fail without stopping later tests', () => {
  const calls = []
  const report = verifyRuntime({ tests: [
    { id: 'example/error', run: () => { calls.push(1); throw Error('sensitive detail') } },
    { id: 'example/empty', run: () => [] },
    { id: 'example/promise', run: () => Promise.resolve([]) },
    { id: 'example/malformed', run: () => [{ name: 'check', passed: 'yes' }] },
    { id: 'example/pass', run: () => { calls.push(2); return [{ name: 'check', passed: true }] } },
  ] })
  assert.deepEqual(calls, [1, 2])
  assert.deepEqual(report.cases.slice(-5).map(result => result.passed), [false, false, false, false, true])
  assert.ok(!JSON.stringify(report).includes('sensitive detail'))
})

test('a rejected promise is reported without an unhandled rejection', async () => {
  const report = verifyRuntime({ tests: [{ id: 'example/rejected', run: () => Promise.reject(Error('private rejection')) }] })
  assert.equal(report.cases.at(-1).passed, false)
  await new Promise(resolve => setImmediate(resolve))
})

test('throwing result getters fail only their own callback', () => {
  for (const property of ['name', 'diagnostic']) {
    const calls = []
    const report = verifyRuntime({ tests: [
      { id: 'example/getter', run: () => {
        calls.push('getter')
        const check = { name: 'check', passed: true }
        Object.defineProperty(check, property, {
          enumerable: true,
          get() { throw Error('private getter failure') },
        })
        return [check]
      } },
      ...['second', 'third'].map(id => ({
        id: `example/${id}`,
        run: () => { calls.push(id); return [{ name: 'check', passed: true }] },
      })),
    ] })
    assert.deepEqual(calls, ['getter', 'second', 'third'])
    assert.deepEqual(report.cases.slice(-3).map(result => result.passed), [false, true, true])
    assert.ok(!JSON.stringify(report).includes('private getter failure'))
  }
})
