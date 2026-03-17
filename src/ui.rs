use std::fmt;
use std::io::IsTerminal;
use std::process::Stdio;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

use console::{Style, style};
use dialoguer::theme::{ColorfulTheme, Theme};
use dialoguer::{Confirm, Input, Select};
use indicatif::{ProgressBar, ProgressStyle};

use crate::error::WhsprError;

static VERBOSITY: AtomicU8 = AtomicU8::new(0);

pub fn configure_terminal_colors() {
    let no_color = std::env::var_os("NO_COLOR").is_some();
    let stdout_is_tty = std::io::stdout().is_terminal();
    let stderr_is_tty = std::io::stderr().is_terminal();
    let colors_enabled = !no_color && (stdout_is_tty || stderr_is_tty);
    console::set_colors_enabled(colors_enabled);
    console::set_colors_enabled_stderr(colors_enabled);
}

pub fn set_verbosity(level: u8) {
    VERBOSITY.store(level, Ordering::Relaxed);
}

pub fn verbosity() -> u8 {
    VERBOSITY.load(Ordering::Relaxed)
}

pub fn is_verbose() -> bool {
    verbosity() > 0
}

pub fn child_stdio() -> Stdio {
    if is_verbose() {
        Stdio::inherit()
    } else {
        Stdio::null()
    }
}

pub fn spinner(message: impl Into<String>) -> ProgressBar {
    if is_verbose() {
        return ProgressBar::hidden();
    }

    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg}")
            .expect("spinner template should be valid"),
    );
    pb.enable_steady_tick(Duration::from_millis(120));
    pb.set_message(message.into());
    pb
}

pub fn progress_bar(total_size: u64) -> ProgressBar {
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .expect("progress bar template should be valid")
            .progress_chars("#>-"),
    );
    pb
}

pub fn header(text: impl AsRef<str>) -> String {
    style(text.as_ref()).bold().cyan().to_string()
}

pub fn section(text: impl AsRef<str>) -> String {
    style(text.as_ref()).bold().blue().to_string()
}

pub fn subtle(text: impl AsRef<str>) -> String {
    style(text.as_ref()).dim().to_string()
}

pub fn value(text: impl AsRef<str>) -> String {
    style(text.as_ref()).bold().to_string()
}

pub fn info_label() -> String {
    style("INFO").bold().cyan().to_string()
}

pub fn ok_label() -> String {
    style("OK").bold().green().to_string()
}

pub fn warn_label() -> String {
    style("WARN").bold().yellow().to_string()
}

pub fn experimental_label() -> String {
    style("EXPERIMENTAL").bold().yellow().to_string()
}

pub fn danger_text(text: impl AsRef<str>) -> String {
    style(text.as_ref()).bold().red().to_string()
}

pub fn summary_key(text: impl AsRef<str>) -> String {
    style(text.as_ref()).bold().blue().to_string()
}

pub fn bullet(text: impl AsRef<str>) -> String {
    style(text.as_ref()).bold().yellow().to_string()
}

pub fn fact_label(text: impl AsRef<str>) -> String {
    style(text.as_ref()).bold().yellow().to_string()
}

pub fn category_token(text: impl AsRef<str>) -> String {
    let text = text.as_ref();
    match text.trim() {
        "ASR" => style(text).bold().cyan().to_string(),
        "Rewrite" => style(text).bold().magenta().to_string(),
        "Cloud" => style(text).bold().blue().to_string(),
        _ => style(text).bold().to_string(),
    }
}

pub fn ready_message(kind: &str, name: impl AsRef<str>) -> String {
    format!(
        "{} {} ready: {}",
        ok_label(),
        category_token(kind),
        value(name)
    )
}

pub struct ConfirmTheme {
    base: ColorfulTheme,
}

pub struct SetupUi {
    theme: ColorfulTheme,
    confirm_theme: ConfirmTheme,
    danger_confirm_theme: ConfirmTheme,
}

pub fn dialog_theme() -> ColorfulTheme {
    ColorfulTheme {
        prompt_style: Style::new().for_stderr().bold().cyan(),
        prompt_prefix: style("›".to_string()).for_stderr().cyan(),
        prompt_suffix: style("".to_string()).for_stderr(),
        success_prefix: style("✓".to_string()).for_stderr().green(),
        success_suffix: style("".to_string()).for_stderr(),
        active_item_style: Style::new().for_stderr().bold().cyan(),
        active_item_prefix: style("›".to_string()).for_stderr().green(),
        checked_item_prefix: style("✓".to_string()).for_stderr().green(),
        unchecked_item_prefix: style("·".to_string()).for_stderr().black().bright(),
        picked_item_prefix: style("›".to_string()).for_stderr().green(),
        unpicked_item_prefix: style(" ".to_string()).for_stderr(),
        defaults_style: Style::new().for_stderr().green(),
        values_style: Style::new().for_stderr().green(),
        hint_style: Style::new().for_stderr().black().bright(),
        ..ColorfulTheme::default()
    }
}

pub fn confirm_dialog_theme() -> ConfirmTheme {
    ConfirmTheme {
        base: dialog_theme(),
    }
}

pub fn danger_dialog_theme() -> ConfirmTheme {
    ConfirmTheme {
        base: ColorfulTheme {
            prompt_style: Style::new().for_stderr().bold().red(),
            prompt_prefix: style("!".to_string()).for_stderr().red(),
            prompt_suffix: style("".to_string()).for_stderr(),
            success_prefix: style("✓".to_string()).for_stderr().green(),
            success_suffix: style("".to_string()).for_stderr(),
            defaults_style: Style::new().for_stderr().red(),
            values_style: Style::new().for_stderr().green(),
            hint_style: Style::new().for_stderr().yellow(),
            active_item_style: Style::new().for_stderr().bold().yellow(),
            active_item_prefix: style("›".to_string()).for_stderr().yellow(),
            ..ColorfulTheme::default()
        },
    }
}

impl Theme for ConfirmTheme {
    fn format_confirm_prompt(
        &self,
        f: &mut dyn fmt::Write,
        prompt: &str,
        default: Option<bool>,
    ) -> fmt::Result {
        if !prompt.is_empty() {
            write!(
                f,
                "{} {} ",
                &self.base.prompt_prefix,
                self.base.prompt_style.apply_to(prompt)
            )?;
        }

        let yes = style("y").for_stderr().bold().green();
        let no = style("n").for_stderr().bold().red();
        let yes_default = style("Y").for_stderr().bold().green();
        let no_default = style("N").for_stderr().bold().red();
        let brackets = Style::new().for_stderr().yellow();

        match default {
            None => write!(
                f,
                "{}{}{}{}{} {}",
                brackets.apply_to("["),
                yes,
                brackets.apply_to("/"),
                no,
                brackets.apply_to("]"),
                &self.base.prompt_suffix
            ),
            Some(true) => write!(
                f,
                "{}{}{}{}{} {} {}",
                brackets.apply_to("["),
                yes_default,
                brackets.apply_to("/"),
                no,
                brackets.apply_to("]"),
                &self.base.prompt_suffix,
                self.base.defaults_style.apply_to("yes")
            ),
            Some(false) => write!(
                f,
                "{}{}{}{}{} {} {}",
                brackets.apply_to("["),
                yes,
                brackets.apply_to("/"),
                no_default,
                brackets.apply_to("]"),
                &self.base.prompt_suffix,
                self.base.defaults_style.apply_to("no")
            ),
        }
    }

    fn format_confirm_prompt_selection(
        &self,
        f: &mut dyn fmt::Write,
        prompt: &str,
        selection: Option<bool>,
    ) -> fmt::Result {
        self.base
            .format_confirm_prompt_selection(f, prompt, selection)
    }
}

impl SetupUi {
    pub fn new() -> Self {
        Self {
            theme: dialog_theme(),
            confirm_theme: confirm_dialog_theme(),
            danger_confirm_theme: danger_dialog_theme(),
        }
    }

    pub fn print_header(&self, text: impl AsRef<str>) {
        println!("{}", header(text));
    }

    pub fn print_section(&self, text: impl AsRef<str>) {
        println!("{}", section(text));
    }

    pub fn print_subtle(&self, text: impl AsRef<str>) {
        println!("{}", subtle(text));
    }

    pub fn print_info(&self, text: impl AsRef<str>) {
        println!("{} {}", info_label(), text.as_ref());
    }

    pub fn print_ok(&self, text: impl AsRef<str>) {
        println!("{} {}", ok_label(), text.as_ref());
    }

    pub fn print_warn(&self, text: impl AsRef<str>) {
        println!("{} {}", warn_label(), text.as_ref());
    }

    pub fn print_experimental_notice(&self, title: impl AsRef<str>, facts: &[(&str, &str)]) {
        println!("{} {}", experimental_label(), value(title));
        for (label, text) in facts {
            println!(
                "  {} {} {}",
                bullet("•"),
                fact_label(format!("{label}:")),
                subtle(text)
            );
        }
    }

    pub fn blank(&self) {
        println!();
    }

    pub fn confirm(&self, prompt: &str, default: bool) -> Result<bool, WhsprError> {
        Confirm::with_theme(&self.confirm_theme)
            .with_prompt(prompt)
            .default(default)
            .show_default(false)
            .report(false)
            .interact()
            .map_err(prompt_error)
    }

    pub fn danger_confirm(
        &self,
        prompt: impl Into<String>,
        default: bool,
    ) -> Result<bool, WhsprError> {
        Confirm::with_theme(&self.danger_confirm_theme)
            .with_prompt(prompt.into())
            .default(default)
            .show_default(false)
            .report(false)
            .interact()
            .map_err(prompt_error)
    }

    pub fn select<T>(&self, prompt: &str, items: &[T], default: usize) -> Result<usize, WhsprError>
    where
        T: std::fmt::Display,
    {
        Select::with_theme(&self.theme)
            .with_prompt(prompt)
            .items(items)
            .default(default)
            .interact()
            .map_err(prompt_error)
    }

    pub fn input_string(&self, prompt: &str, default: Option<&str>) -> Result<String, WhsprError> {
        let input = match default {
            Some(default) => Input::<String>::with_theme(&self.theme)
                .with_prompt(prompt)
                .default(default.to_string()),
            None => Input::<String>::with_theme(&self.theme).with_prompt(prompt),
        };
        input.interact_text().map_err(input_error)
    }
}

impl Default for SetupUi {
    fn default() -> Self {
        Self::new()
    }
}

fn prompt_error(err: dialoguer::Error) -> WhsprError {
    WhsprError::Config(format!("prompt cancelled: {err}"))
}

fn input_error(err: dialoguer::Error) -> WhsprError {
    WhsprError::Config(format!("input cancelled: {err}"))
}

pub fn backend_token(text: impl AsRef<str>) -> String {
    let text = text.as_ref();
    match text.trim() {
        "whisper_cpp" => style(text).bold().cyan().to_string(),
        "faster_whisper" => style(text).bold().magenta().to_string(),
        "nemo" => style(text).bold().yellow().to_string(),
        "cloud" => style(text).bold().blue().to_string(),
        _ => style(text).bold().to_string(),
    }
}

pub fn scope_token(text: impl AsRef<str>) -> String {
    let text = text.as_ref();
    match text.trim() {
        "multi" => style(text).bold().blue().to_string(),
        "en-only" => style(text).bold().green().to_string(),
        _ => style(text).bold().to_string(),
    }
}

pub fn tier_token(text: impl AsRef<str>) -> String {
    let text = text.as_ref();
    match text.trim() {
        "recommended" | "Recommended" => style(text).bold().green().to_string(),
        "optional" | "Optional" => style(text).bold().cyan().to_string(),
        "experimental" | "Experimental" => style(text).bold().yellow().to_string(),
        _ => style(text).bold().to_string(),
    }
}

pub fn status_token(text: impl AsRef<str>) -> String {
    let text = text.as_ref();
    match text.trim() {
        "active" => style(text).bold().green().to_string(),
        "local" => style(text).bold().cyan().to_string(),
        "remote" => style(text).dim().to_string(),
        "blocked" => style(text).bold().red().to_string(),
        "active (missing)" => style(text).bold().yellow().to_string(),
        _ => style(text).to_string(),
    }
}

pub fn provider_token(text: impl AsRef<str>) -> String {
    let text = text.as_ref();
    match text.trim() {
        "openai" | "OpenAI" => style(text).bold().green().to_string(),
        "openai_compatible" | "OpenAI-compatible endpoint" => style(text).bold().blue().to_string(),
        _ => style(text).bold().to_string(),
    }
}

pub fn size_token(text: impl AsRef<str>) -> String {
    style(text.as_ref()).bold().yellow().to_string()
}

pub fn description_token(text: impl AsRef<str>) -> String {
    style(text.as_ref()).to_string()
}
