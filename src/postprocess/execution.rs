use std::path::Path;

use crate::cloud;
use crate::config::{Config, RewriteBackend, RewriteFallback};
use crate::rewrite_worker::{self, RewriteService};

use super::planning::{self, RewritePlan};

pub fn prepare_rewrite_service(config: &Config) -> Option<RewriteService> {
    if !config.postprocess.mode.uses_rewrite() {
        return None;
    }

    if config.rewrite.backend != RewriteBackend::Local
        && config.rewrite.fallback != RewriteFallback::Local
    {
        return None;
    }

    if !crate::rewrite::local_rewrite_available() {
        return None;
    }

    let model_path = planning::resolve_rewrite_model_path(config)?;
    Some(rewrite_worker::RewriteService::new(
        &config.rewrite,
        &model_path,
    ))
}

pub fn prewarm_rewrite_service(service: &RewriteService, phase: &str) {
    match service.prewarm() {
        Ok(()) => tracing::info!("prewarming rewrite worker via {}", phase,),
        Err(err) => tracing::warn!("failed to prewarm rewrite worker: {err}"),
    }
}

pub(crate) async fn execute_rewrite(
    config: &Config,
    rewrite_service: Option<&RewriteService>,
    plan: &RewritePlan,
) -> crate::error::Result<String> {
    match config.rewrite.backend {
        RewriteBackend::Local => {
            local_rewrite_result(
                config,
                rewrite_service,
                plan.local_model_path
                    .as_ref()
                    .expect("local rewrite requires resolved model path"),
                &plan.rewrite_transcript,
                plan.custom_instructions.as_deref(),
            )
            .await
        }
        RewriteBackend::Cloud => {
            let cloud_service = cloud::CloudService::new(config);
            let cloud_result = match cloud_service {
                Ok(service) => {
                    service
                        .rewrite_transcript(
                            config,
                            &plan.rewrite_transcript,
                            plan.custom_instructions.as_deref(),
                        )
                        .await
                }
                Err(err) => Err(err),
            };
            match cloud_result {
                Ok(text) => Ok(text),
                Err(err)
                    if config.rewrite.fallback == RewriteFallback::Local
                        && crate::rewrite::local_rewrite_available() =>
                {
                    tracing::warn!("cloud rewrite failed: {err}; falling back to local rewrite");
                    local_rewrite_result(
                        config,
                        rewrite_service,
                        plan.local_model_path
                            .as_ref()
                            .expect("local rewrite fallback requires resolved model path"),
                        &plan.rewrite_transcript,
                        plan.custom_instructions.as_deref(),
                    )
                    .await
                }
                Err(err) => Err(err),
            }
        }
    }
}

async fn local_rewrite_result(
    config: &Config,
    rewrite_service: Option<&RewriteService>,
    model_path: &Path,
    rewrite_transcript: &crate::rewrite_protocol::RewriteTranscript,
    custom_instructions: Option<&str>,
) -> crate::error::Result<String> {
    if !crate::rewrite::local_rewrite_available() {
        return Err(crate::error::WhsprError::Rewrite(
            "local rewrite is unavailable in this build; rebuild with --features local-rewrite"
                .into(),
        ));
    }

    if let Some(service) = rewrite_service {
        rewrite_worker::rewrite_with_service(
            service,
            &config.rewrite,
            rewrite_transcript,
            custom_instructions,
        )
        .await
    } else {
        rewrite_worker::rewrite_transcript(
            &config.rewrite,
            model_path,
            rewrite_transcript,
            custom_instructions,
        )
        .await
    }
}
