import assert from 'node:assert/strict'
import { test } from 'node:test'

import { npmInvocation } from './git-install-command.mjs'

const NPM_ARGUMENTS = ['install', '--omit=optional']
const WINDOWS_COMMAND_INTERPRETER = 'C:\\Windows\\System32\\cmd.exe'

test('invokes npm directly outside Windows', () => {
  assert.deepEqual(npmInvocation(NPM_ARGUMENTS, { platform: 'linux' }), {
    command: 'npm',
    args: NPM_ARGUMENTS,
  })
})

test('invokes npm.cmd through the Windows command interpreter', () => {
  assert.deepEqual(
    npmInvocation(NPM_ARGUMENTS, {
      platform: 'win32',
      commandInterpreter: WINDOWS_COMMAND_INTERPRETER,
    }),
    {
      command: WINDOWS_COMMAND_INTERPRETER,
      args: ['/d', '/s', '/c', 'npm.cmd', ...NPM_ARGUMENTS],
    },
  )
})
