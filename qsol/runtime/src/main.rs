use qsol_harness_runtime::{CompositionRoot, InferenceRequest, RuntimeError, load_config};
use std::io::Read;

#[derive(Debug, PartialEq, Eq)]
struct CliArgs {
    config_path: String,
    prompt: String,
    experiment_id: String,
    seed: Option<u64>,
}

fn set_once(slot: &mut Option<String>, flag: &str, value: String) -> Result<(), RuntimeError> {
    if slot.replace(value).is_some() {
        return Err(RuntimeError::Config(format!(
            "duplicate command-line option: {flag}"
        )));
    }
    Ok(())
}

fn parse_cli_args(args: &[String]) -> Result<CliArgs, RuntimeError> {
    let mut config_path = None;
    let mut prompt = None;
    let mut experiment_id = None;
    let mut seed = None;
    let mut index = 1;

    while index < args.len() {
        let flag = &args[index];
        let known = matches!(
            flag.as_str(),
            "--config" | "--prompt" | "--experiment-id" | "--seed"
        );
        if !known {
            return Err(RuntimeError::Config(format!(
                "unknown command-line option: {flag}"
            )));
        }
        let value = args.get(index + 1).ok_or_else(|| {
            RuntimeError::Config(format!("missing value for command-line option: {flag}"))
        })?;
        if value.starts_with("--") {
            return Err(RuntimeError::Config(format!(
                "missing value for command-line option: {flag}"
            )));
        }

        match flag.as_str() {
            "--config" => set_once(&mut config_path, flag, value.clone())?,
            "--prompt" => set_once(&mut prompt, flag, value.clone())?,
            "--experiment-id" => set_once(&mut experiment_id, flag, value.clone())?,
            "--seed" => {
                if seed.is_some() {
                    return Err(RuntimeError::Config(
                        "duplicate command-line option: --seed".into(),
                    ));
                }
                seed = Some(value.parse::<u64>().map_err(|_| {
                    RuntimeError::Config("--seed must be an unsigned integer".into())
                })?);
            }
            _ => unreachable!("known flags are exhaustively matched"),
        }
        index += 2;
    }

    let config_path = config_path.ok_or_else(|| {
        RuntimeError::Config(
            "usage: qsol-harness-headless --config <json> --prompt <text|-> [--experiment-id <id>] [--seed <u64>]"
                .into(),
        )
    })?;
    let prompt = prompt.ok_or_else(|| RuntimeError::Config("--prompt is required".into()))?;

    Ok(CliArgs {
        config_path,
        prompt,
        experiment_id: experiment_id.unwrap_or_else(|| "qsol-headless".into()),
        seed,
    })
}

fn read_prompt(value: String) -> Result<String, RuntimeError> {
    if value != "-" {
        return Ok(value);
    }
    let mut prompt = String::new();
    std::io::stdin().read_to_string(&mut prompt)?;
    if prompt.ends_with('\n') {
        prompt.pop();
        if prompt.ends_with('\r') {
            prompt.pop();
        }
    }
    Ok(prompt)
}

fn run() -> Result<(), RuntimeError> {
    let args: Vec<String> = std::env::args().collect();
    let parsed = parse_cli_args(&args)?;
    let config = load_config(&parsed.config_path)?;
    let root = CompositionRoot::from_config(config)?;
    let result = root.run(InferenceRequest {
        experiment_id: parsed.experiment_id,
        prompt: read_prompt(parsed.prompt)?,
        seed: parsed.seed,
    })?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!(
            "{{\"schema\":\"qsol-harness/headless-error/1\",\"error\":{}}}",
            serde_json::to_string(&error.to_string())
                .unwrap_or_else(|_| "\"serialization failure\"".into())
        );
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn rejects_unknown_flags_instead_of_defaulting_identity() {
        let error = parse_cli_args(&args(&[
            "qsol-harness-headless",
            "--config",
            "config.json",
            "--prompt",
            "test",
            "--experiment-idd",
            "run-42",
        ]))
        .unwrap_err();
        assert!(error.to_string().contains("unknown command-line option"));
    }

    #[test]
    fn rejects_duplicate_options() {
        let error = parse_cli_args(&args(&[
            "qsol-harness-headless",
            "--config",
            "a.json",
            "--config",
            "b.json",
            "--prompt",
            "test",
        ]))
        .unwrap_err();
        assert!(error.to_string().contains("duplicate command-line option"));
    }

    #[test]
    fn defaults_experiment_id_only_after_strict_parse() {
        let parsed = parse_cli_args(&args(&[
            "qsol-harness-headless",
            "--config",
            "config.json",
            "--prompt",
            "test",
            "--seed",
            "7",
        ]))
        .unwrap();
        assert_eq!(parsed.experiment_id, "qsol-headless");
        assert_eq!(parsed.seed, Some(7));
    }
}
