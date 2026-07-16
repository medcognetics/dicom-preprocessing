const DEFAULT_WINDOWS_COMMAND_INTERPRETER = 'cmd.exe'
const WINDOWS_NPM_COMMAND = 'npm.cmd'

export function npmInvocation(
  args,
  {
    platform = process.platform,
    commandInterpreter = process.env.ComSpec ?? DEFAULT_WINDOWS_COMMAND_INTERPRETER,
  } = {},
) {
  if (platform === 'win32') {
    // Command scripts require cmd.exe; /d also prevents user AutoRun commands from affecting the test.
    return {
      command: commandInterpreter,
      args: ['/d', '/s', '/c', WINDOWS_NPM_COMMAND, ...args],
    }
  }

  return {
    command: 'npm',
    args,
  }
}
