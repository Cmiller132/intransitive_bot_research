//! Locked, ordered pair journal; resume validates the protocol before reading observations.
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::Pair;

pub(super) fn digest(value: &Value) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(value)?)))
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    protocol: Value,
    protocol_digest: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    pair: usize,
    games: Pair,
}

pub(super) struct Journal(File);

impl Journal {
    pub fn open(
        path: &Path,
        resume: bool,
        protocol: &Value,
        mut retire: impl FnMut(usize, Pair) -> Result<()>,
    ) -> Result<Self> {
        let mut options = OpenOptions::new();
        options.read(true).write(true);
        if !resume {
            options.create_new(true);
        }
        let mut file = options
            .open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        file.try_lock()
            .context("sequential journal is already in use")?;
        let protocol_digest = digest(protocol)?;
        if resume {
            let mut reader = BufReader::new(&file);
            let mut bytes = Vec::new();
            let mut offset = reader.read_until(b'\n', &mut bytes)? as u64;
            ensure!(bytes.last() == Some(&b'\n'), "incomplete journal header");
            let header: Header = serde_json::from_slice(&bytes)?;
            if header.protocol_digest != protocol_digest || header.protocol != *protocol {
                let bounds = |protocol: &Value| {
                    let test = &protocol["test"];
                    format!("s0 {} s1 {} cap {}", test["s0"], test["s1"], test["cap"])
                };
                let (recorded, requested) = (bounds(&header.protocol), bounds(protocol));
                ensure!(
                    recorded == requested,
                    "resume bounds differ: the journal has {recorded}, the options give {requested}"
                );
                bail!("resume protocol differs (settings, openings or artifact hashes)");
            }
            loop {
                bytes.clear();
                let size = reader.read_until(b'\n', &mut bytes)?;
                if size == 0 {
                    break;
                }
                if bytes.last() != Some(&b'\n') {
                    break;
                }
                let entry: Entry =
                    serde_json::from_slice(&bytes).context("invalid complete journal line")?;
                retire(entry.pair, entry.games)?;
                offset += size as u64;
            }
            drop(reader);
            file.set_len(offset)?;
            file.seek(SeekFrom::End(0))?;
        } else {
            serde_json::to_writer(
                &mut file,
                &Header {
                    protocol: protocol.clone(),
                    protocol_digest,
                },
            )?;
            file.write_all(b"\n")?;
            file.sync_data()?;
        }
        Ok(Self(file))
    }

    pub fn append(&mut self, pair: usize, games: &Pair) -> Result<()> {
        serde_json::to_writer(&mut self.0, &serde_json::json!({"pair":pair,"games":games}))?;
        self.0.write_all(b"\n")?;
        Ok(())
    }

    pub fn sync(&self) -> Result<()> {
        self.0.sync_data().context("sync sequential batch")
    }
}
