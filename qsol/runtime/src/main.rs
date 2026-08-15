use qsol_harness_runtime::{CompositionRoot, InferenceRequest, RuntimeError, load_config};
use std::io::Read;

fn value_after(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|pair| pair[0] == flag)
        .map(|pair| pair[1].clone())
}

fn parse_seed(args: &[String]) -> Result<Option<u64>, RuntimeError> {
    match value_after(args, "--seed") {
        Some(value) => value
            .parse::<u64>()
            .map(Some)
            .map_err(|_| RuntimeError::Config("--seed must be an unsigned integer".into())),
        None => Ok(None),
    }
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
    let config_path = value_after(&args, "--config").ok_or_else(|| {
        RuntimeError::Config(
            "usage: qsol-harness-headless --config <json> --prompt <text|-> [--experiment-id <id>] [--seed <u64>]"
                .into(),
        )
    })?;
    let prompt = value_after(&args, "--prompt")
        .ok_or_else(|| RuntimeError::Config("--prompt is required".into()))?;
    let experiment_id = value_after(&args, "--experiment-id")
        .unwrap_or_else(|| "qsol-headless".into());
    let seed = parse_seed(&args)?;

    let config = load_config(config_path)?;
    let root = CompositionRoot::from_config(config)?;
    let result = root.run(InferenceRequest {
        experiment_id,
        prompt: read_prompt(prompt)?,
        seed,
    })?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{{\"schema\":\"qsol-harness/headless-error/1\",\"error\":{}}}", serde_json::to_string(&error.to_string()).unwrap_or_else(|_| "\"serialization failure\"".into()));
        std::process::exit(2);
    }
}
