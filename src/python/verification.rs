use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde::de::DeserializeOwned;

use crate::verification::{
    CodecDeclaration, PreparedVerification, VerificationCase, VerificationCheck,
};

fn decode<T: DeserializeOwned>(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<T> {
    let json: String = py
        .import("json")?
        .call_method1("dumps", (value,))?
        .extract()?;
    serde_json::from_str(&json)
        .map_err(|_| PyValueError::new_err("Invalid verification declaration"))
}

#[pyfunction]
#[pyo3(name = "verify_runtime", signature = (*, cases=None, codecs=None, tests=None))]
fn verify_runtime<'py>(
    py: Python<'py>,
    cases: Option<&Bound<'py, PyAny>>,
    codecs: Option<&Bound<'py, PyAny>>,
    tests: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut fixture_cases = Vec::<VerificationCase>::new();
    if let Some(cases) = cases {
        for case in cases.try_iter()? {
            let case = case?.cast_into::<PyDict>()?.copy()?;
            let bytes: Vec<u8> = case
                .get_item("dicom_bytes")?
                .ok_or_else(|| PyValueError::new_err("Missing dicom_bytes"))?
                .extract()?;
            case.set_item("dicom_bytes", PyList::new(py, bytes)?)?;
            fixture_cases.push(decode(py, case.as_any())?);
        }
    }
    let declarations = match codecs {
        Some(codecs) => codecs
            .try_iter()?
            .map(|item| decode::<CodecDeclaration>(py, &item?))
            .collect::<PyResult<Vec<_>>>()?,
        None => Vec::new(),
    };
    let mut callbacks = Vec::new();
    let mut ids = Vec::new();
    if let Some(tests) = tests {
        for test in tests.try_iter()? {
            let test = test?.cast_into::<PyDict>()?;
            if test.len() != 2 {
                return Err(PyValueError::new_err(
                    "Custom tests require only id and run",
                ));
            }
            let id: String = test
                .get_item("id")?
                .ok_or_else(|| PyValueError::new_err("Missing test id"))?
                .extract()?;
            let callback = test
                .get_item("run")?
                .ok_or_else(|| PyValueError::new_err("Missing test callback"))?;
            if !callback.is_callable() {
                return Err(PyValueError::new_err("Test run must be callable"));
            }
            ids.push(id);
            callbacks.push(callback);
        }
    }
    let suite = PreparedVerification::new(fixture_cases, declarations, ids)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let mut run = py.detach(move || suite.run_shared());
    for callback in callbacks {
        let outcome = callback
            .call0()
            .and_then(|value| decode::<Vec<VerificationCheck>>(py, &value));
        match outcome {
            Ok(checks) => run.record_custom(Ok(checks)),
            Err(error) if error.is_instance_of::<PyException>(py) => {
                run.record_custom(Err("callback failed".into()))
            }
            Err(error) => return Err(error), // KeyboardInterrupt, SystemExit, and other BaseExceptions.
        }
    }
    let report = serde_json::to_string(&run.finish())
        .map_err(|_| PyValueError::new_err("Cannot serialize verification report"))?;
    py.import("json")?.call_method1("loads", (report,))
}

pub(crate) fn register_submodule<'py>(
    py: Python<'py>,
    module: &Bound<'py, PyModule>,
) -> PyResult<()> {
    let definitions = std::ffi::CStr::from_bytes_with_nul(
        concat!(include_str!("verification_types.py"), "\0").as_bytes(),
    )
    .expect("static Python definitions contain no NUL bytes");
    py.run(definitions, Some(&module.dict()), None)?;
    for name in [
        "VerificationTagPath",
        "VerificationTagExpectation",
        "VerificationFrameExpectation",
        "VerificationCase",
        "VerificationCodec",
        "VerificationCheck",
        "VerificationCustomTest",
        "VerificationCaseResult",
        "VerificationCodecResult",
        "VerificationReport",
    ] {
        // add() also includes the class in PyO3's __all__, used by wheel imports.
        module.add(name, module.getattr(name)?)?;
    }
    module.add_function(wrap_pyfunction!(verify_runtime, module)?)
}
