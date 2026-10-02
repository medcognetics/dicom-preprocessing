import { verifyRuntime, type VerificationCaseResult, type VerificationReport } from '../index.js'

const report: VerificationReport = verifyRuntime()
const passed: boolean = report.passed
const sources: Array<VerificationCaseResult['source']> = report.cases.map(result => result.source)
void passed
void sources
// @ts-expect-error verification takes no options
verifyRuntime({ cases: [] })
