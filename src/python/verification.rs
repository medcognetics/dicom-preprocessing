use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Runs the built-in runtime self-test with the GIL released and returns the report as a dict.
#[pyfunction]
#[pyo3(name = "verify_runtime")]
fn verify_runtime(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let report = py.detach(crate::verification::verify_runtime);
    let report = serde_json::to_string(&report)
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
        "VerificationCheck",
        "VerificationCaseResult",
        "VerificationCodecResult",
        "VerificationReport",
    ] {
        // add() also includes the class in PyO3's __all__, used by wheel imports.
        module.add(name, module.getattr(name)?)?;
    }
    module.add_function(wrap_pyfunction!(verify_runtime, module)?)
}
