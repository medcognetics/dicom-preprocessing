# Runtime typing exports for the native verification report.

import typing as _typing


class VerificationCheck(_typing.TypedDict):
    name: str
    passed: bool
    diagnostic: _typing.Optional[str]


class VerificationCaseResult(_typing.TypedDict):
    id: str
    source: _typing.Literal["embedded_fixture", "builtin"]
    transfer_syntax_uid: _typing.Optional[str]
    passed: bool
    checks: _typing.List[VerificationCheck]


class VerificationCodecResult(_typing.TypedDict):
    transfer_syntax_uid: str
    required_cases: _typing.List[str]
    passed: bool


class VerificationReport(_typing.TypedDict):
    schema_version: int
    suite_version: str
    library_version: str
    passed: bool
    cases: _typing.List[VerificationCaseResult]
    codecs: _typing.List[VerificationCodecResult]
