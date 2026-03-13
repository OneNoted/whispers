use std::path::Path;

use dialoguer::{Confirm, Select};

use crate::config::{self, PostprocessMode, resolve_config_path};
use crate::error::Result;
use crate::model::{self, MODELS};
use crate::rewrite_model::{self, REWRITE_MODELS};

pub async fn run_setup(config_path_override: Option<&Path>) -> Result<()> {
    println!("whispers setup");
    println!();

    // Build selection items
    let items: Vec<String> = MODELS
        .iter()
        .map(|m| format!("{:<22} {:>8}  {}", m.name, m.size, m.description))
        .collect();

    let selection = Select::new()
        .with_prompt("Choose a whisper model to download")
        .items(&items)
        .default(0) // large-v3-turbo
        .interact()
        .map_err(|e| crate::error::WhsprError::Config(format!("selection cancelled: {e}")))?;

    let chosen = &MODELS[selection];
    println!();
    tracing::info!("setup selected model: {}", chosen.name);

    // Download the model
    model::download_model(chosen.name).await?;
    println!();

    let enable_advanced = Confirm::new()
        .with_prompt("Enable advanced local dictation rewriting?")
        .default(false)
        .interact()
        .map_err(|e| crate::error::WhsprError::Config(format!("confirmation cancelled: {e}")))?;

    let selected_rewrite_model = if enable_advanced {
        let items: Vec<String> = REWRITE_MODELS
            .iter()
            .map(|m| format!("{:<24} {:>9}  {}", m.name, m.size, m.description))
            .collect();
        let selection = Select::new()
            .with_prompt("Choose a local rewrite model to download")
            .items(&items)
            .default(1)
            .interact()
            .map_err(|e| crate::error::WhsprError::Config(format!("selection cancelled: {e}")))?;

        let chosen = &REWRITE_MODELS[selection];
        println!();
        tracing::info!("setup selected rewrite model: {}", chosen.name);
        rewrite_model::download_model(chosen.name).await?;
        println!();
        Some(chosen.name)
    } else {
        None
    };

    // Generate or update config
    let config_path = resolve_config_path(config_path_override);
    let model_path_str = model::model_path_for_config(chosen.filename);

    if config_path.exists() {
        println!("Config already exists at {}", config_path.display());
        tracing::info!("updating existing config at {}", config_path.display());
        config::update_config_model_path(&config_path, &model_path_str)?;
    } else {
        tracing::info!("writing new config at {}", config_path.display());
        config::write_default_config(&config_path, &model_path_str)?;
    }

    if let Some(rewrite_model) = selected_rewrite_model {
        config::update_config_rewrite_selection(&config_path, rewrite_model)?;
        println!(
            "Enabled advanced_local mode with rewrite model: {}",
            rewrite_model
        );
    } else {
        config::update_config_postprocess_mode(&config_path, PostprocessMode::Raw)?;
        println!("Configured raw postprocessing mode.");
    }

    println!("Config updated: {}", config_path.display());

    println!();
    println!("Setup complete! You can now use whispers.");
    println!("Bind it to a key in your compositor, e.g. for Hyprland:");
    println!("  bind = SUPER ALT, D, exec, whispers");

    Ok(())
}
