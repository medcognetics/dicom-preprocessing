import { readFileSync } from 'node:fs'

import {
  PreparedDicom,
  prepareDicom,
  renderDisplayFrame,
  renderFrame,
  type CoordinateTransform,
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
const displayRendered: RenderedFrame = renderDisplayFrame(prepared, 0)
const dtype: RenderedFrame['dtype'] = 'int8'
const coordinateTransform: CoordinateTransform = displayRendered.coordinateTransform

if (source.kind === 'stored') {
  source.storedFrameIndex.toFixed()
}

rendered.data.byteLength.toFixed()
displayRendered.data.byteLength.toFixed()
coordinateTransform.sourceToDisplay[0].toFixed()
coordinateTransform.validDisplayRect.width.toFixed()
prepareDicom(byteViewInput).renderFrame(0)
prepareDicom(byteViewInput).renderDisplayFrame(0)
dtype.toUpperCase()
