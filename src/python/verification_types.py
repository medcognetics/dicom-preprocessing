# Runtime typing exports for the native verification API.

import typing as _typing


class _VerificationTagPathRequired(_typing.TypedDict):
    group: int
    element: int


class VerificationTagPath(_VerificationTagPathRequired, total=False):
    item: _typing.Optional[int]


class VerificationTagExpectation(_typing.TypedDict):
    path: _typing.List[VerificationTagPath]
    vr: str
    values: _typing.List[str]


class _VerificationFrameRequired(_typing.TypedDict):
    width: int
    height: int
    samples_per_pixel: int
    planar_configuration: _typing.Literal[0, 1]
    sample_type: _typing.Literal["u8", "i8", "u16", "i16"]
    values: _typing.List[int]


class VerificationFrameExpectation(_VerificationFrameRequired, total=False):
    absolute_tolerance: int


class VerificationCase(_typing.TypedDict):
    id: str
    dicom_bytes: bytes
    transfer_syntax_uid: str
    tags: _typing.List[VerificationTagExpectation]
    frames: _typing.List[VerificationFrameExpectation]


class VerificationCodec(_typing.TypedDict):
    id: str
    required_cases: _typing.Dict[str, _typing.List[str]]


class _VerificationCheckRequired(_typing.TypedDict):
    name: str
    passed: bool


class VerificationCheck(_VerificationCheckRequired, total=False):
    diagnostic: _typing.Optional[str]


class VerificationCustomTest(_typing.TypedDict):
    id: str
    run: _typing.Callable[[], _typing.Sequence[VerificationCheck]]


class VerificationCaseResult(_typing.TypedDict):
    id: str
    source: _typing.Literal["embedded_fixture", "generated_fixture", "extension_fixture", "custom"]
    path: _typing.Literal["shared_library", "caller_integration"]
    transfer_syntax_uid: _typing.Optional[str]
    passed: bool
    checks: _typing.List[VerificationCheck]


class VerificationCodecResult(_typing.TypedDict):
    id: str
    transfer_syntax_uid: str
    required_cases: _typing.List[str]
    shared_library_verified: bool
    passed: bool


class VerificationReport(_typing.TypedDict):
    schema_version: int
    suite_version: str
    library_version: str
    passed: bool
    cases: _typing.List[VerificationCaseResult]
    codecs: _typing.List[VerificationCodecResult]
