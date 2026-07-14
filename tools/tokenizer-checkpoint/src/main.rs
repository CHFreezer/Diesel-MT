use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail, ensure};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use tokenizers::models::bpe::BpeTrainer;
use tokenizers::models::{ModelWrapper, TrainerWrapper};
use tokenizers::{AddedToken, Result as TokenizerResult, Tokenizer, Trainer};

const CHECKPOINT_SCHEMA_VERSION: u32 = 2;
const TOKENIZERS_ENGINE_VERSION: &str = "0.22.2";
const SNAPSHOT_MAGIC: &[u8; 16] = b"DMTTOKSNAPV1\0\0\0\0";
const PROGRESS_RECORD_INTERVAL: u64 = 100_000;
const PROGRESS_TIME_INTERVAL: Duration = Duration::from_secs(10);
const MAX_RECORD_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Parser)]
#[command(about = "Persist and resume the Hugging Face BPE trainer feed state")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run the official Tokenizer preprocessing/feed path and save BpeTrainer.words.
    Feed {
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        trainer_config: PathBuf,
        #[arg(long)]
        snapshot: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Load a feed checkpoint and run tokenization, pair counting, and BPE merges.
    Train {
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        vocab_size: usize,
        #[arg(long)]
        min_frequency: u64,
        #[arg(long)]
        limit_alphabet: usize,
    },
    /// Validate checkpoint metadata without allocating merge-stage structures.
    Inspect {
        #[arg(long)]
        checkpoint: PathBuf,
    },
}

#[derive(Serialize, Deserialize)]
struct TrainerCheckpoint {
    schema_version: u32,
    tokenizers_engine_version: String,
    tokenizer_sha256: [u8; 32],
    trainer_config_sha256: [u8; 32],
    snapshot_sha256: [u8; 32],
    input_order_sha256: [u8; 32],
    input_records: u64,
    input_utf8_bytes: u64,
    trainer: TrainerWrapper,
}

#[derive(Debug, Serialize)]
struct CheckpointSummary {
    schema_version: u32,
    tokenizers_engine_version: String,
    tokenizer_sha256: String,
    trainer_config_sha256: String,
    snapshot_sha256: String,
    input_order_sha256: String,
    input_records: u64,
    input_utf8_bytes: u64,
    checkpoint_bytes: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct BpeTrainerConfig {
    #[serde(rename = "type")]
    trainer_type: String,
    vocab_size: usize,
    min_frequency: u64,
    show_progress: bool,
    special_tokens: Vec<AddedToken>,
    limit_alphabet: usize,
    initial_alphabet: Vec<char>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    max_token_length: Option<usize>,
}

impl BpeTrainerConfig {
    fn build(self) -> Result<TrainerWrapper> {
        ensure!(self.trainer_type == "BPE", "trainer config is not BPE");
        let mut builder = BpeTrainer::builder()
            .vocab_size(self.vocab_size)
            .min_frequency(self.min_frequency)
            .show_progress(self.show_progress)
            .special_tokens(self.special_tokens)
            .limit_alphabet(self.limit_alphabet)
            .initial_alphabet(self.initial_alphabet.into_iter().collect::<HashSet<_>>())
            .max_token_length(self.max_token_length);
        if let Some(prefix) = self.continuing_subword_prefix {
            builder = builder.continuing_subword_prefix(prefix);
        }
        if let Some(suffix) = self.end_of_word_suffix {
            builder = builder.end_of_word_suffix(suffix);
        }
        Ok(builder.build().into())
    }
}

/// Delegate the official feed implementation while deliberately making the
/// final train call a no-op. Passing this adapter to Tokenizer::train keeps the
/// exact Tokenizer normalization and pre-tokenization path without duplicating
/// private TokenizerImpl helpers.
struct FeedOnlyTrainer {
    inner: TrainerWrapper,
}

impl FeedOnlyTrainer {
    fn into_inner(self) -> TrainerWrapper {
        self.inner
    }
}

impl Trainer for FeedOnlyTrainer {
    type Model = ModelWrapper;

    fn should_show_progress(&self) -> bool {
        self.inner.should_show_progress()
    }

    fn train(&self, _model: &mut Self::Model) -> TokenizerResult<Vec<AddedToken>> {
        Ok(Vec::new())
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> TokenizerResult<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> TokenizerResult<Vec<String>> + Sync,
    {
        self.inner.feed(iterator, process)
    }
}

/// Reader for the Python-produced canonical snapshot:
/// 16-byte magic, u64 record count, then repeated u64 byte length + UTF-8 bytes.
struct SnapshotInput {
    path: PathBuf,
    reader: BufReader<File>,
    expected_records: u64,
    records: u64,
    utf8_bytes: u64,
    digest: Sha256,
    error: Option<anyhow::Error>,
    saw_eof: bool,
    started: Instant,
    last_report: Instant,
}

impl SnapshotInput {
    fn open(path: &Path) -> Result<Self> {
        ensure!(path.is_file(), "snapshot not found: {}", path.display());
        let file = File::open(path)
            .with_context(|| format!("failed to open snapshot {}", path.display()))?;
        let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
        let mut magic = [0_u8; 16];
        reader
            .read_exact(&mut magic)
            .with_context(|| format!("snapshot header is truncated: {}", path.display()))?;
        ensure!(
            &magic == SNAPSHOT_MAGIC,
            "unsupported snapshot format in {}",
            path.display()
        );
        let mut count_bytes = [0_u8; 8];
        reader
            .read_exact(&mut count_bytes)
            .with_context(|| format!("snapshot count is truncated: {}", path.display()))?;
        let expected_records = u64::from_be_bytes(count_bytes);
        ensure!(expected_records > 0, "snapshot declares zero records");
        let now = Instant::now();
        Ok(Self {
            path: path.to_owned(),
            reader,
            expected_records,
            records: 0,
            utf8_bytes: 0,
            digest: Sha256::new(),
            error: None,
            saw_eof: false,
            started: now,
            last_report: now,
        })
    }

    fn read_record(&mut self) -> Result<Option<String>> {
        let mut length_bytes = [0_u8; 8];
        let first = self
            .reader
            .read(&mut length_bytes[..1])
            .with_context(|| format!("failed reading {}", self.path.display()))?;
        if first == 0 {
            self.saw_eof = true;
            ensure!(
                self.records == self.expected_records,
                "snapshot ended after {} records, expected {}",
                self.records,
                self.expected_records
            );
            return Ok(None);
        }
        self.reader
            .read_exact(&mut length_bytes[1..])
            .context("truncated snapshot record length")?;
        let length = u64::from_be_bytes(length_bytes);
        ensure!(
            length <= MAX_RECORD_BYTES,
            "snapshot record {} is too large: {} bytes",
            self.records + 1,
            length
        );
        let length_usize = usize::try_from(length).context("record length exceeds usize")?;
        let mut encoded = vec![0_u8; length_usize];
        self.reader
            .read_exact(&mut encoded)
            .with_context(|| format!("snapshot record {} is truncated", self.records + 1))?;
        self.digest.update(length_bytes);
        self.digest.update(&encoded);
        let text = String::from_utf8(encoded)
            .with_context(|| format!("snapshot record {} is not UTF-8", self.records + 1))?;
        self.records += 1;
        self.utf8_bytes += length;
        ensure!(
            self.records <= self.expected_records,
            "snapshot contains more than the declared {} records",
            self.expected_records
        );

        let now = Instant::now();
        if self.records % PROGRESS_RECORD_INTERVAL == 0
            && now.duration_since(self.last_report) >= PROGRESS_TIME_INTERVAL
        {
            let elapsed = now.duration_since(self.started).as_secs_f64();
            eprintln!(
                "SNAPSHOT records={}/{} utf8_gib={:.3} rate={:.0} records/s",
                self.records,
                self.expected_records,
                self.utf8_bytes as f64 / 1024_f64.powi(3),
                self.records as f64 / elapsed.max(1e-9)
            );
            self.last_report = now;
        }
        Ok(Some(text))
    }

    fn input_order_sha256(&self) -> [u8; 32] {
        self.digest.clone().finalize().into()
    }

    fn finish(self) -> Result<Self> {
        if let Some(error) = self.error.as_ref() {
            bail!("snapshot iteration failed: {error:#}");
        }
        ensure!(self.saw_eof, "snapshot iterator was not consumed to EOF");
        ensure!(
            self.records == self.expected_records,
            "consumed {} records, expected {}",
            self.records,
            self.expected_records
        );
        Ok(self)
    }
}

impl Iterator for SnapshotInput {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if self.error.is_some() || self.saw_eof {
            return None;
        }
        match self.read_record() {
            Ok(value) => value,
            Err(error) => {
                self.error = Some(error);
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.expected_records.saturating_sub(self.records);
        match usize::try_from(remaining) {
            Ok(value) => (value, Some(value)),
            Err(_) => (usize::MAX, None),
        }
    }
}

fn sha256_file(path: &Path) -> Result<[u8; 32]> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 8 * 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("failed reading {}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(digest.finalize().into())
}

fn hex_digest(value: &[u8; 32]) -> String {
    value.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn summary(checkpoint: &TrainerCheckpoint, checkpoint_bytes: Option<u64>) -> CheckpointSummary {
    CheckpointSummary {
        schema_version: checkpoint.schema_version,
        tokenizers_engine_version: checkpoint.tokenizers_engine_version.clone(),
        tokenizer_sha256: hex_digest(&checkpoint.tokenizer_sha256),
        trainer_config_sha256: hex_digest(&checkpoint.trainer_config_sha256),
        snapshot_sha256: hex_digest(&checkpoint.snapshot_sha256),
        input_order_sha256: hex_digest(&checkpoint.input_order_sha256),
        input_records: checkpoint.input_records,
        input_utf8_bytes: checkpoint.input_utf8_bytes,
        checkpoint_bytes,
    }
}

fn validate_checkpoint(checkpoint: &TrainerCheckpoint) -> Result<()> {
    ensure!(
        checkpoint.schema_version == CHECKPOINT_SCHEMA_VERSION,
        "unsupported checkpoint schema {}, expected {}",
        checkpoint.schema_version,
        CHECKPOINT_SCHEMA_VERSION
    );
    ensure!(
        checkpoint.tokenizers_engine_version == TOKENIZERS_ENGINE_VERSION,
        "checkpoint tokenizers version {} does not match helper {}",
        checkpoint.tokenizers_engine_version,
        TOKENIZERS_ENGINE_VERSION
    );
    match &checkpoint.trainer {
        TrainerWrapper::BpeTrainer(_) => Ok(()),
        _ => bail!("checkpoint does not contain a BPE trainer"),
    }
}

fn temporary_path(output: &Path) -> Result<PathBuf> {
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).with_context(|| format!("failed to create {}", parent.display()))?;
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .context("output path has no UTF-8 filename")?;
    Ok(parent.join(format!(".{name}.{}.tmp", std::process::id())))
}

fn write_checkpoint(output: &Path, checkpoint: &TrainerCheckpoint) -> Result<()> {
    ensure!(
        !output.exists(),
        "refusing to replace existing checkpoint {}",
        output.display()
    );
    let temporary = temporary_path(output)?;
    let result = (|| -> Result<()> {
        let file = File::create(&temporary)
            .with_context(|| format!("failed to create {}", temporary.display()))?;
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
        bincode::serde::encode_into_std_write(checkpoint, &mut writer, bincode::config::standard())
            .context("failed to serialize trainer checkpoint")?;
        writer
            .flush()
            .context("failed to flush trainer checkpoint")?;
        writer
            .get_ref()
            .sync_all()
            .context("failed to sync trainer checkpoint")?;
        fs::rename(&temporary, output).with_context(|| {
            format!(
                "failed to publish checkpoint {} -> {}",
                temporary.display(),
                output.display()
            )
        })?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn read_checkpoint(path: &Path) -> Result<TrainerCheckpoint> {
    let file = File::open(path)
        .with_context(|| format!("failed to open checkpoint {}", path.display()))?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let checkpoint: TrainerCheckpoint =
        bincode::serde::decode_from_std_read(&mut reader, bincode::config::standard())
            .context("failed to deserialize trainer checkpoint")?;
    validate_checkpoint(&checkpoint)?;
    Ok(checkpoint)
}

fn write_tokenizer(output: &Path, tokenizer: &Tokenizer) -> Result<()> {
    ensure!(
        !output.exists(),
        "refusing to replace existing tokenizer {}",
        output.display()
    );
    let temporary = temporary_path(output)?;
    let result = (|| -> Result<()> {
        let file = File::create(&temporary)
            .with_context(|| format!("failed to create {}", temporary.display()))?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, tokenizer).context("failed to serialize tokenizer")?;
        writer.write_all(b"\n")?;
        writer.flush()?;
        writer.get_ref().sync_all()?;
        fs::rename(&temporary, output)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn feed(
    tokenizer_path: &Path,
    trainer_config_path: &Path,
    snapshot_path: &Path,
    output: &Path,
) -> Result<()> {
    ensure!(
        tokenizer_path.is_file(),
        "tokenizer not found: {}",
        tokenizer_path.display()
    );
    ensure!(
        trainer_config_path.is_file(),
        "trainer config not found: {}",
        trainer_config_path.display()
    );
    let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|error| {
        anyhow!(
            "failed to load tokenizer {}: {error}",
            tokenizer_path.display()
        )
    })?;
    ensure!(
        tokenizer.get_added_tokens_decoder().is_empty(),
        "feed tokenizer must not contain added tokens"
    );
    ensure!(
        tokenizer.get_post_processor().is_none(),
        "feed tokenizer must not contain a post-processor"
    );
    let trainer_file = File::open(trainer_config_path)?;
    let trainer_config: BpeTrainerConfig = serde_json::from_reader(BufReader::new(trainer_file))
        .context("failed to parse trainer config")?;
    let inner = trainer_config.build()?;
    let mut trainer = FeedOnlyTrainer { inner };
    let mut snapshot = SnapshotInput::open(snapshot_path)?;
    let started = Instant::now();
    tokenizer
        .train(&mut trainer, &mut snapshot)
        .map_err(|error| anyhow!("trainer feed failed: {error}"))?;
    let snapshot = snapshot.finish()?;
    eprintln!(
        "FEED complete records={} utf8_gib={:.3} elapsed_s={:.3}",
        snapshot.records,
        snapshot.utf8_bytes as f64 / 1024_f64.powi(3),
        started.elapsed().as_secs_f64()
    );

    let checkpoint = TrainerCheckpoint {
        schema_version: CHECKPOINT_SCHEMA_VERSION,
        tokenizers_engine_version: TOKENIZERS_ENGINE_VERSION.to_owned(),
        tokenizer_sha256: sha256_file(tokenizer_path)?,
        trainer_config_sha256: sha256_file(trainer_config_path)?,
        snapshot_sha256: sha256_file(snapshot_path)?,
        input_order_sha256: snapshot.input_order_sha256(),
        input_records: snapshot.records,
        input_utf8_bytes: snapshot.utf8_bytes,
        trainer: trainer.into_inner(),
    };
    write_checkpoint(output, &checkpoint)?;
    let bytes = fs::metadata(output)?.len();
    println!(
        "{}",
        serde_json::to_string(&summary(&checkpoint, Some(bytes)))?
    );
    Ok(())
}

fn train(
    tokenizer_path: &Path,
    checkpoint_path: &Path,
    output: &Path,
    vocab_size: usize,
    min_frequency: u64,
    limit_alphabet: usize,
) -> Result<()> {
    ensure!(vocab_size > 0, "vocab size must be positive");
    let mut tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|error| {
        anyhow!(
            "failed to load tokenizer {}: {error}",
            tokenizer_path.display()
        )
    })?;
    let mut checkpoint = read_checkpoint(checkpoint_path)?;
    ensure!(
        checkpoint.tokenizer_sha256 == sha256_file(tokenizer_path)?,
        "checkpoint tokenizer fingerprint does not match {}",
        tokenizer_path.display()
    );
    match &mut checkpoint.trainer {
        TrainerWrapper::BpeTrainer(trainer) => {
            trainer.vocab_size = vocab_size;
            trainer.min_frequency = min_frequency;
            trainer.limit_alphabet = Some(limit_alphabet);
        }
        _ => bail!("checkpoint does not contain a BPE trainer"),
    }
    let mut model: ModelWrapper = tokenizer.get_model().clone();
    let started = Instant::now();
    let special_tokens = checkpoint
        .trainer
        .train(&mut model)
        .map_err(|error| anyhow!("trainer merge stage failed: {error}"))?;
    tokenizer.with_model(model);
    tokenizer.add_special_tokens(&special_tokens);
    write_tokenizer(output, &tokenizer)?;
    eprintln!(
        "TRAIN complete vocab_size={} elapsed_s={:.3}",
        vocab_size,
        started.elapsed().as_secs_f64()
    );
    println!(
        "{}",
        serde_json::json!({
            "tokenizers_engine_version": TOKENIZERS_ENGINE_VERSION,
            "vocab_size": vocab_size,
            "min_frequency": min_frequency,
            "limit_alphabet": limit_alphabet,
            "checkpoint": checkpoint_path,
            "output": output,
        })
    );
    Ok(())
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Feed {
            tokenizer,
            trainer_config,
            snapshot,
            output,
        } => feed(&tokenizer, &trainer_config, &snapshot, &output),
        Command::Train {
            tokenizer,
            checkpoint,
            output,
            vocab_size,
            min_frequency,
            limit_alphabet,
        } => train(
            &tokenizer,
            &checkpoint,
            &output,
            vocab_size,
            min_frequency,
            limit_alphabet,
        ),
        Command::Inspect { checkpoint } => {
            let value = read_checkpoint(&checkpoint)?;
            let bytes = fs::metadata(checkpoint)?.len();
            println!("{}", serde_json::to_string(&summary(&value, Some(bytes)))?);
            Ok(())
        }
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("ERROR: {error:#}");
        std::process::exit(1);
    }
}
