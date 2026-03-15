use std::path::Path;

use clap::Parser;
use whispers::cli::{
    AppRuleAction, AsrModelAction, Cli, CloudAction, Command, DictionaryAction, GlossaryAction,
    ModelAction, RewriteModelAction, SnippetAction,
};
use whispers::config::Config;
use whispers::error::Result;
use whispers::rewrite_protocol::RewriteSurfaceKind;
use whispers::{
    agentic_rewrite, app, asr, asr_model, audio, cloud, completions, file_audio, model,
    personalization, postprocess, rewrite_model, runtime_support, setup,
};

fn build_context_matcher(
    surface_kind: Option<RewriteSurfaceKind>,
    app_id: Option<&String>,
    window_title_contains: Option<&String>,
    browser_domain_contains: Option<&String>,
) -> agentic_rewrite::ContextMatcher {
    agentic_rewrite::ContextMatcher {
        surface_kind,
        app_id: app_id.cloned(),
        window_title_contains: window_title_contains.cloned(),
        browser_domain_contains: browser_domain_contains.cloned(),
    }
}

async fn transcribe_file(cli: &Cli, file: &Path, output: Option<&Path>, raw: bool) -> Result<()> {
    let config = Config::load(cli.config.as_deref())?;
    asr::validation::validate_transcription_config(&config)?;
    tracing::info!("decoding audio file: {}", file.display());
    let mut samples = file_audio::decode_audio_file(file)?;
    audio::preprocess_audio(&mut samples, file_audio::TARGET_SAMPLE_RATE);
    let rewrite_service = if raw {
        None
    } else {
        postprocess::execution::prepare_rewrite_service(&config)
    };
    if let Some(service) = rewrite_service.as_ref() {
        postprocess::execution::prewarm_rewrite_service(service, "file transcription");
    }
    let prepared = asr::prepare::prepare_transcriber(&config)?;
    asr::prepare::prewarm_transcriber(&prepared, "file transcription");
    let transcript =
        asr::execute::transcribe_audio(&config, prepared, samples, file_audio::TARGET_SAMPLE_RATE)
            .await?;

    let text = if raw {
        postprocess::planning::raw_text(&transcript)
    } else {
        postprocess::finalize::finalize_transcript(
            &config,
            transcript,
            rewrite_service.as_ref(),
            None,
            None,
        )
        .await
        .text
    };

    if let Some(out_path) = output {
        tokio::fs::write(out_path, &text).await?;
        tracing::info!("transcription written to {}", out_path.display());
    } else {
        println!("{text}");
    }

    Ok(())
}

async fn run_default(cli: &Cli) -> Result<()> {
    let Some(_pid_lock) = runtime_support::acquire_or_signal_lock()? else {
        return Ok(());
    };

    tracing::info!("whispers v{}", env!("CARGO_PKG_VERSION"));

    // Load config
    let config = Config::load(cli.config.as_deref())?;
    asr::validation::validate_transcription_config(&config)?;
    tracing::debug!("config loaded: {config:?}");

    app::run(config).await
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    runtime_support::init_tracing(cli.verbose);

    match &cli.command {
        None => run_default(&cli).await,
        Some(Command::Completions { shell }) => completions::run_completions(*shell),
        Some(Command::Setup) => setup::run_setup(cli.config.as_deref()).await,
        Some(Command::Transcribe { file, output, raw }) => {
            transcribe_file(&cli, file, output.as_deref(), *raw).await
        }
        Some(Command::Model { action }) => match action {
            ModelAction::List => {
                model::list_models(cli.config.as_deref());
                Ok(())
            }
            ModelAction::Download { name } => {
                model::download_model(name).await?;
                Ok(())
            }
            ModelAction::Select { name } => model::select_model(name, cli.config.as_deref()),
        },
        Some(Command::AsrModel { action }) => match action {
            AsrModelAction::List => {
                asr_model::list_models(cli.config.as_deref());
                Ok(())
            }
            AsrModelAction::Download { name } => {
                asr_model::download_model(name).await?;
                Ok(())
            }
            AsrModelAction::Select { name } => asr_model::select_model(name, cli.config.as_deref()),
        },
        Some(Command::RewriteModel { action }) => match action {
            RewriteModelAction::List => {
                rewrite_model::list_models(cli.config.as_deref());
                Ok(())
            }
            RewriteModelAction::Download { name } => {
                rewrite_model::download_model(name).await?;
                Ok(())
            }
            RewriteModelAction::Select { name } => {
                rewrite_model::select_model(name, cli.config.as_deref())
            }
        },
        Some(Command::Dictionary { action }) => match action {
            DictionaryAction::List => personalization::list_dictionary(cli.config.as_deref()),
            DictionaryAction::Add { phrase, replace } => {
                personalization::add_dictionary(cli.config.as_deref(), phrase, replace)
            }
            DictionaryAction::Remove { phrase } => {
                personalization::remove_dictionary(cli.config.as_deref(), phrase)
            }
        },
        Some(Command::AppRule { action }) => match action {
            AppRuleAction::Path => agentic_rewrite::print_app_rule_path(cli.config.as_deref()),
            AppRuleAction::List => agentic_rewrite::list_app_rules(cli.config.as_deref()),
            AppRuleAction::Add {
                name,
                instructions,
                surface_kind,
                app_id,
                window_title_contains,
                browser_domain_contains,
                correction_policy,
            } => agentic_rewrite::add_app_rule(
                cli.config.as_deref(),
                name,
                instructions,
                build_context_matcher(
                    *surface_kind,
                    app_id.as_ref(),
                    window_title_contains.as_ref(),
                    browser_domain_contains.as_ref(),
                ),
                *correction_policy,
            ),
            AppRuleAction::Remove { name } => {
                agentic_rewrite::remove_app_rule(cli.config.as_deref(), name)
            }
        },
        Some(Command::Glossary { action }) => match action {
            GlossaryAction::Path => agentic_rewrite::print_glossary_path(cli.config.as_deref()),
            GlossaryAction::List => agentic_rewrite::list_glossary(cli.config.as_deref()),
            GlossaryAction::Add {
                term,
                aliases,
                surface_kind,
                app_id,
                window_title_contains,
                browser_domain_contains,
            } => agentic_rewrite::add_glossary_entry(
                cli.config.as_deref(),
                term,
                aliases,
                build_context_matcher(
                    *surface_kind,
                    app_id.as_ref(),
                    window_title_contains.as_ref(),
                    browser_domain_contains.as_ref(),
                ),
            ),
            GlossaryAction::Remove { term } => {
                agentic_rewrite::remove_glossary_entry(cli.config.as_deref(), term)
            }
        },
        Some(Command::Cloud { action }) => match action {
            CloudAction::Check => {
                let config = Config::load(cli.config.as_deref())?;
                cloud::validate_config(&config)?;
                let service = cloud::CloudService::new(&config)?;
                service.check().await?;
                println!(
                    "Cloud provider '{}' is configured and reachable.",
                    config.cloud.provider.as_str()
                );
                Ok(())
            }
        },
        Some(Command::Snippets { action }) => match action {
            SnippetAction::List => personalization::list_snippets(cli.config.as_deref()),
            SnippetAction::Add { name, text } => {
                personalization::add_snippet(cli.config.as_deref(), name, text)
            }
            SnippetAction::Remove { name } => {
                personalization::remove_snippet(cli.config.as_deref(), name)
            }
        },
        Some(Command::RewriteInstructionsPath) => {
            personalization::print_rewrite_instructions_path(cli.config.as_deref())
        }
    }
}
