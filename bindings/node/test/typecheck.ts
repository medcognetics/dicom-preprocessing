import { readFileSync } from 'node:fs'

import {
  PreparedDicom,
  prepareDicom,
  renderFrame,
  type DicomInput,
  type FrameSource,
  type RenderedFrame,
  type VolumeHandler,
} from '../index'

const input: DicomInput = { bytes: readFileSync('/tmp/example.dcm'), filename: 'example.dcm' }
const byteViewInput: DicomInput = { bytes: new Uint8Array([0]), filename: 'example.dcm' }
const handler: VolumeHandler = { kind: 'max-intensity', skipStart: 1, skipEnd: 1 }
const prepared: PreparedDicom = prepareDicom(input, { volumeHandler: handler })
const source: FrameSource = prepared.framePlan.displayFrames[0]
const rendered: RenderedFrame = renderFrame(prepared, 0)
const dtype: RenderedFrame['dtype'] = 'int8'

if (source.kind === 'stored') {
  source.storedFrameIndex.toFixed()
}

rendered.data.byteLength.toFixed()
prepareDicom(byteViewInput).renderFrame(0)
dtype.toUpperCase()
