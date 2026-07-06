import assert from 'node:assert/strict'
import { existsSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'

import { prepareDicom, renderFrame } from '../index.js'

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../../..')
const DEFAULT_FIXTURES = {
  DICOM_PREPROCESSING_CT_FIXTURE: 'target/dicom_test_files/pydicom/CT_small.dcm',
  DICOM_PREPROCESSING_MULTIFRAME_FIXTURE: 'target/dicom_test_files/pydicom/emri_small.dcm',
  DICOM_PREPROCESSING_RGB_FIXTURE: 'target/dicom_test_files/pydicom/SC_rgb.dcm',
}

function requireFixture(name) {
  const candidates = [process.env[name], resolve(REPO_ROOT, DEFAULT_FIXTURES[name])].filter(Boolean)
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return candidate
    }
  }
  throw new Error(`${name} environment variable is required, or run make test-node to download fixtures`)
}

test('prepareDicom accepts a file path and renders raw monochrome pixels', () => {
  const prepared = prepareDicom({ path: requireFixture('DICOM_PREPROCESSING_CT_FIXTURE') })
  const rendered = prepared.renderFrame(0)

  assert.deepEqual(prepared.framePlan.displayFrames, [{ kind: 'stored', storedFrameIndex: 0 }])
  assert.equal(rendered.displayFrameIndex, 0)
  assert.deepEqual(rendered.source, { kind: 'stored', storedFrameIndex: 0 })
  assert.equal(rendered.samplesPerPixel, 1)
  assert.equal(rendered.photometricInterpretation.startsWith('MONOCHROME'), true)
  assert.match(rendered.dtype, /^u?int(8|16)$/)
  assert.equal(Buffer.isBuffer(rendered.data), true)
  assert.equal(
    rendered.data.length,
    rendered.width * rendered.height * rendered.samplesPerPixel * (rendered.dtype === 'uint8' ? 1 : 2),
  )
})

test('prepareDicom accepts bytes with parity against path input', () => {
  const path = requireFixture('DICOM_PREPROCESSING_CT_FIXTURE')
  const fromPath = renderFrame(prepareDicom({ path }), 0)
  const fromBytes = renderFrame(prepareDicom({ bytes: readFileSync(path), filename: 'CT_small.dcm' }), 0)

  assert.equal(fromBytes.width, fromPath.width)
  assert.equal(fromBytes.height, fromPath.height)
  assert.equal(fromBytes.dtype, fromPath.dtype)
  assert.deepEqual(fromBytes.source, fromPath.source)
  assert.deepEqual(fromBytes.data, fromPath.data)
})

test('prepareDicom accepts Uint8Array byte views', () => {
  const path = requireFixture('DICOM_PREPROCESSING_CT_FIXTURE')
  const file = readFileSync(path)
  const padded = new Uint8Array(file.byteLength + 2)
  padded.set(file, 1)
  const bytes = padded.subarray(1, 1 + file.byteLength)

  assert.equal(Buffer.isBuffer(bytes), false)
  const fromPath = renderFrame(prepareDicom({ path }), 0)
  const fromBytes = renderFrame(prepareDicom({ bytes, filename: 'CT_small.dcm' }), 0)

  assert.equal(fromBytes.width, fromPath.width)
  assert.equal(fromBytes.height, fromPath.height)
  assert.equal(fromBytes.dtype, fromPath.dtype)
  assert.deepEqual(fromBytes.data, fromPath.data)
})

test('renderFrame returns RGB metadata for RGB DICOM input', () => {
  const rendered = renderFrame(prepareDicom({ path: requireFixture('DICOM_PREPROCESSING_RGB_FIXTURE') }), 0)

  assert.equal(rendered.samplesPerPixel, 3)
  assert.equal(rendered.photometricInterpretation, 'RGB')
  assert.equal(rendered.dtype, 'uint8')
  assert.equal(rendered.data.length, rendered.width * rendered.height * rendered.samplesPerPixel)
})

test('derived frame sources are exposed and raw rendering rejects them', () => {
  const prepared = prepareDicom(
    { path: requireFixture('DICOM_PREPROCESSING_MULTIFRAME_FIXTURE') },
    { volumeHandler: { kind: 'interpolate', targetFrames: 2 } },
  )

  assert.deepEqual(prepared.framePlan.displayFrames, [{ kind: 'derived' }, { kind: 'derived' }])
  assert.throws(() => prepared.renderFrame(0), { code: 'DERIVED_FRAME_NO_RAW_SOURCE' })
})

test('invalid frame index is structured', () => {
  const prepared = prepareDicom({ path: requireFixture('DICOM_PREPROCESSING_CT_FIXTURE') })

  assert.throws(() => renderFrame(prepared, 999), { code: 'FRAME_INDEX_OUT_OF_RANGE' })
})

test('frame indexes are validated before rendering', () => {
  const prepared = prepareDicom({ path: requireFixture('DICOM_PREPROCESSING_CT_FIXTURE') })

  assert.throws(() => prepared.renderFrame(0.5), { code: 'FRAME_INDEX_OUT_OF_RANGE' })
  assert.throws(() => renderFrame(prepared, 2 ** 40), { code: 'FRAME_INDEX_OUT_OF_RANGE' })
})

test('unreadable path and non-DICOM bytes are structured', () => {
  assert.throws(() => prepareDicom({ path: '/definitely/not/a/file.dcm' }), { code: 'READ_FILE' })
  assert.throws(
    () => prepareDicom({ bytes: Buffer.from('not dicom') }),
    (error) => {
      assert.equal(error.code, 'READ_BYTES')
      assert.doesNotMatch(error.message, /Backtrace|dicom_preprocessing_node::/)
      return true
    },
  )
})
