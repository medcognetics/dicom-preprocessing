use napi::Result;
use napi_derive::napi;
use serde_json::Value;

/// Converts snake_case report keys to the camelCase used by the Node API.
fn camel_keys(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(camel_keys).collect()),
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .map(|(key, value)| {
                    let mut upper = false;
                    let key = key
                        .chars()
                        .filter_map(|c| {
                            if c == '_' {
                                upper = true;
                                None
                            } else if upper {
                                upper = false;
                                Some(c.to_ascii_uppercase())
                            } else {
                                Some(c)
                            }
                        })
                        .collect();
                    (key, camel_keys(value))
                })
                .collect(),
        ),
        other => other,
    }
}

/// Runs the built-in runtime self-test synchronously. Failures are report data.
#[napi(ts_return_type = "VerificationReport")]
pub fn verify_runtime() -> Result<Value> {
    let report = serde_json::to_value(dicom_preprocessing::verify_runtime())
        .map_err(|error| napi::Error::from_reason(error.to_string()))?;
    Ok(camel_keys(report))
}
