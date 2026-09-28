import { verifyRuntime, type VerificationCase, type VerificationOptions, type VerificationReport } from '../index.js'

const fixture: VerificationCase = {
  id: 'app/fixture', dicomBytes: new Uint8Array(), transferSyntaxUid: '1.2.3',
  tags: [{ path: [{ group: 0x28, element: 0x10 }], vr: 'US', values: ['1'] }],
  frames: [{ width: 1, height: 1, samplesPerPixel: 1, planarConfiguration: 0, sampleType: 'u8', values: [42] }],
}
const options: VerificationOptions = {
  cases: [fixture], codecs: [{ id: 'app/codec', requiredCases: { '1.2.3': ['app/test'] } }],
  tests: [{ id: 'app/test', run: () => [{ name: 'check', passed: true }] }],
}
const report: VerificationReport = verifyRuntime(options)
const passed: boolean = report.passed
void passed
// @ts-expect-error callbacks must be synchronous
const invalid: VerificationOptions = { tests: [{ id: 'app/async', run: async () => [] }] }
void invalid
