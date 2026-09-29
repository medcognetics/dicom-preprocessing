use dicom_preprocessing::verification::{
    CodecDeclaration, PreparedVerification, VerificationCase, VerificationCheck,
};
use napi::bindgen_prelude::{
    FromNapiValue, Function, JsObjectValue, Object, Uint8Array, Unknown, ValidateNapiValue,
};
use napi::{Env, Error, JsValue, Result, Status};
use napi_derive::napi;
use serde_json::Value;

fn invalid() -> Error {
    Error::new(Status::InvalidArg, "Invalid verification configuration")
}

fn required<T: FromNapiValue + ValidateNapiValue>(object: &Object<'_>, name: &str) -> Result<T> {
    object.get_named_property(name).map_err(|_| invalid())
}

fn snake_keys(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(snake_keys).collect()),
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .map(|(key, value)| {
                    let key = key
                        .chars()
                        .flat_map(|c| {
                            if c.is_ascii_uppercase() {
                                vec!['_', c.to_ascii_lowercase()]
                            } else {
                                vec![c]
                            }
                        })
                        .collect();
                    (key, snake_keys(value))
                })
                .collect(),
        ),
        other => other,
    }
}

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

#[napi(
    ts_args_type = "options?: VerificationOptions",
    ts_return_type = "VerificationReport"
)]
pub fn verify_runtime<'env>(env: &'env Env, options: Option<Object<'env>>) -> Result<Value> {
    let mut cases = Vec::<VerificationCase>::new();
    let mut codecs = Vec::<CodecDeclaration>::new();
    let mut ids = Vec::new();
    let mut callbacks = Vec::new();
    if let Some(options) = options {
        let keys = options.get_property_names()?;
        for index in 0..keys.get_array_length()? {
            let key: String = keys.get_element(index)?;
            if !matches!(key.as_str(), "cases" | "codecs" | "tests") {
                return Err(invalid());
            }
        }
        for case in options.get::<Vec<Object>>("cases")?.unwrap_or_default() {
            let bytes: Uint8Array = required(&case, "dicomBytes")?;
            let value = serde_json::json!({
                "id": required::<String>(&case, "id")?,
                "dicom_bytes": bytes.to_vec(),
                "transfer_syntax_uid": required::<String>(&case, "transferSyntaxUid")?,
                "tags": snake_keys(env.from_js_value(required::<Unknown>(&case, "tags")?)?),
                "frames": snake_keys(env.from_js_value(required::<Unknown>(&case, "frames")?)?),
            });
            cases.push(serde_json::from_value(value).map_err(|_| invalid())?);
        }
        for codec in options.get::<Vec<Value>>("codecs")?.unwrap_or_default() {
            codecs.push(serde_json::from_value(snake_keys(codec)).map_err(|_| invalid())?);
        }
        for test in options.get::<Vec<Object>>("tests")?.unwrap_or_default() {
            ids.push(required::<String>(&test, "id")?);
            let callback: Unknown = required(&test, "run")?;
            if callback.get_type()? != napi::ValueType::Function {
                return Err(invalid());
            }
            callbacks.push(required::<Function<(), Unknown>>(&test, "run")?);
        }
    }
    let suite = PreparedVerification::new(cases, codecs, ids).map_err(|_| invalid())?;
    let mut run = suite.run_shared();
    for callback in callbacks {
        let outcome = (|| -> Result<Vec<VerificationCheck>> {
            let value = callback.call(())?;
            if value.is_promise()? {
                // Consume a rejection before rejecting this synchronous test result.
                // Otherwise an unsupported callback could crash the host later.
                let promise = value.coerce_to_object()?;
                let catch: Function<Function<Unknown, ()>, Unknown> = required(&promise, "catch")?;
                let ignore = env.create_function_from_closure::<Unknown, (), _>(
                    "ignoreVerificationRejection",
                    |_| Ok(()),
                )?;
                catch.apply(promise, ignore)?;
                return Err(invalid());
            }
            let value: Value = env.from_js_value(value).or_else(|error| {
                // Unlike Function::call, serde conversion can leave a throwing
                // property getter's exception pending. Clear it before the next test.
                napi::check_pending_exception!(env.raw(), error.status.into())?;
                Err(error)
            })?;
            serde_json::from_value(snake_keys(value)).map_err(|_| invalid())
        })();
        run.record_custom(outcome.map_err(|_| "callback failed".into()));
    }
    serde_json::to_value(run.finish())
        .map(camel_keys)
        .map_err(|_| {
            Error::new(
                Status::GenericFailure,
                "Cannot serialize verification report",
            )
        })
}
