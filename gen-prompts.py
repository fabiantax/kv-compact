"""Generate realistic prompts for compaction benchmark."""
import json
import os

TMP = os.environ.get('TEMP', '/tmp')

# ~10K token prompt: distributed KV store guide with Rust code
PROMPT_10K = r"""# Distributed Key-Value Store Implementation Guide

## Chapter 1: Architecture Overview

A distributed key-value store is a fundamental building block of modern cloud infrastructure. Unlike traditional relational databases, KV stores optimize for simple read/write operations with horizontal scalability. This guide covers the complete implementation of a production-grade distributed KV store in Rust.

### 1.1 Design Principles

The system follows these core design principles:
- **Partition Tolerance**: The system continues to operate despite network partitions between nodes.
- **Eventual Consistency**: All replicas converge to the same state given sufficient time without new updates.
- **Horizontal Scalability**: Adding nodes linearly increases throughput and storage capacity.
- **Fault Tolerance**: The system tolerates up to f failures in a 2f+1 replica configuration.

### 1.2 System Components

The architecture consists of several interconnected components:

1. **Router Layer**: Receives client requests and routes them to the appropriate partition.
2. **Consensus Module**: Implements Raft consensus for leader election and log replication.
3. **Storage Engine**: LSM-tree based storage with write-ahead logging.
4. **Membership Service**: Tracks cluster topology and handles node joins/departures.
5. **Replication Manager**: Handles cross-datacenter async replication.

## Chapter 2: The Storage Engine

### 2.1 LSM-Tree Implementation

The Log-Structured Merge Tree provides excellent write throughput by converting random writes into sequential I/O.

```rust
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};
use std::fs::{File, OpenOptions};
use std::io::{Write, BufWriter, BufReader, Read};

/// In-memory component of the LSM tree
struct MemTable {
    data: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
    size_bytes: usize,
    max_size: usize,
}

impl MemTable {
    fn new(max_size: usize) -> Self {
        MemTable {
            data: BTreeMap::new(),
            size_bytes: 0,
            max_size,
        }
    }

    fn put(&mut self, key: Vec<u8>, value: Vec<u8>) -> bool {
        self.size_bytes += key.len() + value.len() + 16;
        self.data.insert(key, Some(value));
        self.size_bytes >= self.max_size
    }

    fn get(&self, key: &[u8]) -> Option<Option<&Vec<u8>>> {
        self.data.get(key).map(|v| v.as_ref())
    }

    fn delete(&mut self, key: Vec<u8>) -> bool {
        self.size_bytes += key.len() + 8;
        self.data.insert(key, None);
        self.size_bytes >= self.max_size
    }

    fn iter(&self) -> impl Iterator<Item = (&Vec<u8>, &Option<Vec<u8>>)> {
        self.data.iter()
    }
}

/// Sorted String Table - immutable on-disk component
struct SSTable {
    path: String,
    index: BTreeMap<Vec<u8>, u64>,
    bloom_filter: BloomFilter,
    min_key: Vec<u8>,
    max_key: Vec<u8>,
    level: u32,
}

impl SSTable {
    fn write_from_memtable(memtable: &MemTable, path: &str, level: u32) -> std::io::Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        let mut index = BTreeMap::new();
        let mut bloom = BloomFilter::new(memtable.data.len(), 0.01);
        let mut offset: u64 = 0;
        let mut min_key = Vec::new();
        let mut max_key = Vec::new();

        for (i, (key, value)) in memtable.data.iter().enumerate() {
            if i == 0 { min_key = key.clone(); }
            max_key = key.clone();

            bloom.insert(key);
            index.insert(key.clone(), offset);

            let key_len = key.len() as u32;
            writer.write_all(&key_len.to_le_bytes())?;
            writer.write_all(key)?;

            match value {
                Some(v) => {
                    let val_len = v.len() as u32;
                    writer.write_all(&val_len.to_le_bytes())?;
                    writer.write_all(v)?;
                    offset += 4 + key.len() as u64 + 4 + v.len() as u64;
                }
                None => {
                    writer.write_all(&u32::MAX.to_le_bytes())?;
                    offset += 4 + key.len() as u64 + 4;
                }
            }
        }
        writer.flush()?;

        Ok(SSTable { path: path.to_string(), index, bloom_filter: bloom, min_key, max_key, level })
    }
}

struct BloomFilter {
    bits: Vec<bool>,
    num_hashes: usize,
}

impl BloomFilter {
    fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        let num_bits = (-(expected_items as f64 * false_positive_rate.ln())
            / (2.0_f64.ln().powi(2))).ceil() as usize;
        let num_hashes = ((num_bits as f64 / expected_items as f64)
            * 2.0_f64.ln()).ceil() as usize;
        BloomFilter { bits: vec![false; num_bits.max(1)], num_hashes }
    }

    fn insert(&mut self, key: &[u8]) {
        for i in 0..self.num_hashes {
            let hash = self.hash(key, i) % self.bits.len();
            self.bits[hash] = true;
        }
    }

    fn may_contain(&self, key: &[u8]) -> bool {
        (0..self.num_hashes).all(|i| {
            let hash = self.hash(key, i) % self.bits.len();
            self.bits[hash]
        })
    }

    fn hash(&self, key: &[u8], seed: usize) -> usize {
        let mut hash: u64 = 14695981039346656037u64.wrapping_add(seed as u64);
        for &byte in key {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
        hash as usize
    }
}
```

### 2.2 Write-Ahead Log

The WAL ensures durability by persisting all mutations before applying them to the memtable.

```rust
use std::io::Seek;

struct WriteAheadLog {
    file: BufWriter<File>,
    path: String,
    sequence_number: u64,
}

enum WalEntry {
    Put { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
}

impl WriteAheadLog {
    fn new(path: &str) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(WriteAheadLog {
            file: BufWriter::new(file),
            path: path.to_string(),
            sequence_number: 0,
        })
    }

    fn append(&mut self, entry: &WalEntry) -> std::io::Result<u64> {
        self.sequence_number += 1;
        let seq = self.sequence_number;
        self.file.write_all(&seq.to_le_bytes())?;

        match entry {
            WalEntry::Put { key, value } => {
                self.file.write_all(&[0x01])?;
                self.file.write_all(&(key.len() as u32).to_le_bytes())?;
                self.file.write_all(key)?;
                self.file.write_all(&(value.len() as u32).to_le_bytes())?;
                self.file.write_all(value)?;
            }
            WalEntry::Delete { key } => {
                self.file.write_all(&[0x02])?;
                self.file.write_all(&(key.len() as u32).to_le_bytes())?;
                self.file.write_all(key)?;
            }
        }
        self.file.write_all(&[0u8; 4])?;
        self.file.flush()?;
        Ok(seq)
    }
}
```

## Chapter 3: Raft Consensus

### 3.1 State Machine

The Raft consensus algorithm ensures all nodes agree on the order of operations.

```rust
use std::time::{Duration, Instant};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
enum RaftState { Follower, Candidate, Leader }

#[derive(Clone, Debug)]
struct LogEntry { term: u64, index: u64, command: Command }

#[derive(Clone, Debug)]
enum Command {
    Put { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
    Noop,
    AddServer { id: String, address: String },
    RemoveServer { id: String },
}

struct RaftNode {
    id: String,
    state: RaftState,
    current_term: u64,
    voted_for: Option<String>,
    log: Vec<LogEntry>,
    commit_index: u64,
    last_applied: u64,
    next_index: HashMap<String, u64>,
    match_index: HashMap<String, u64>,
    election_timeout: Duration,
    last_heartbeat: Instant,
    peers: Vec<String>,
}

impl RaftNode {
    fn new(id: String, peers: Vec<String>) -> Self {
        RaftNode {
            id, state: RaftState::Follower, current_term: 0,
            voted_for: None,
            log: vec![LogEntry { term: 0, index: 0, command: Command::Noop }],
            commit_index: 0, last_applied: 0,
            next_index: HashMap::new(), match_index: HashMap::new(),
            election_timeout: Duration::from_millis(150),
            last_heartbeat: Instant::now(), peers,
        }
    }

    fn tick(&mut self) -> Vec<RaftMessage> {
        let mut messages = Vec::new();
        match self.state {
            RaftState::Follower | RaftState::Candidate => {
                if self.last_heartbeat.elapsed() > self.election_timeout {
                    self.start_election(&mut messages);
                }
            }
            RaftState::Leader => {
                for peer in &self.peers {
                    let next = self.next_index.get(peer).copied().unwrap_or(1);
                    let prev_index = next - 1;
                    let prev_term = self.log.get(prev_index as usize)
                        .map(|e| e.term).unwrap_or(0);
                    let entries: Vec<LogEntry> = self.log[next as usize..].to_vec();
                    messages.push(RaftMessage::AppendEntries {
                        to: peer.clone(), term: self.current_term,
                        leader_id: self.id.clone(),
                        prev_log_index: prev_index, prev_log_term: prev_term,
                        entries, leader_commit: self.commit_index,
                    });
                }
            }
        }
        messages
    }

    fn start_election(&mut self, messages: &mut Vec<RaftMessage>) {
        self.state = RaftState::Candidate;
        self.current_term += 1;
        self.voted_for = Some(self.id.clone());
        self.last_heartbeat = Instant::now();
        let last_log_index = self.log.last().map(|e| e.index).unwrap_or(0);
        let last_log_term = self.log.last().map(|e| e.term).unwrap_or(0);
        for peer in &self.peers {
            messages.push(RaftMessage::RequestVote {
                to: peer.clone(), term: self.current_term,
                candidate_id: self.id.clone(),
                last_log_index, last_log_term,
            });
        }
    }

    fn handle_append_entries(&mut self, term: u64, prev_log_index: u64,
                             prev_log_term: u64, entries: &[LogEntry],
                             leader_commit: u64) -> (u64, bool) {
        if term < self.current_term { return (self.current_term, false); }
        self.current_term = term;
        self.state = RaftState::Follower;
        self.last_heartbeat = Instant::now();

        if prev_log_index > 0 {
            if let Some(entry) = self.log.get(prev_log_index as usize) {
                if entry.term != prev_log_term {
                    self.log.truncate(prev_log_index as usize);
                    return (self.current_term, false);
                }
            } else { return (self.current_term, false); }
        }

        for entry in entries {
            let idx = entry.index as usize;
            if idx < self.log.len() {
                if self.log[idx].term != entry.term {
                    self.log.truncate(idx);
                    self.log.push(entry.clone());
                }
            } else { self.log.push(entry.clone()); }
        }

        if leader_commit > self.commit_index {
            self.commit_index = std::cmp::min(
                leader_commit,
                self.log.last().map(|e| e.index).unwrap_or(0)
            );
        }
        (self.current_term, true)
    }
}

#[derive(Debug)]
enum RaftMessage {
    RequestVote {
        to: String, term: u64, candidate_id: String,
        last_log_index: u64, last_log_term: u64,
    },
    AppendEntries {
        to: String, term: u64, leader_id: String,
        prev_log_index: u64, prev_log_term: u64,
        entries: Vec<LogEntry>, leader_commit: u64,
    },
}
```

### 3.2 Consistent Hashing

```rust
use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

struct ConsistentHash {
    ring: BTreeMap<u64, String>,
    virtual_nodes: usize,
}

impl ConsistentHash {
    fn new(virtual_nodes: usize) -> Self {
        ConsistentHash { ring: BTreeMap::new(), virtual_nodes }
    }

    fn add_node(&mut self, node: &str) {
        for i in 0..self.virtual_nodes {
            let key = format!("{}:{}", node, i);
            let hash = self.hash(&key);
            self.ring.insert(hash, node.to_string());
        }
    }

    fn get_node(&self, key: &str) -> Option<&String> {
        if self.ring.is_empty() { return None; }
        let hash = self.hash(key);
        self.ring.range(hash..).next()
            .or_else(|| self.ring.iter().next())
            .map(|(_, v)| v)
    }

    fn hash(&self, key: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}
```

## Chapter 4: Compaction and Garbage Collection

### 4.1 Level-Based Compaction

As data accumulates across multiple SSTables, compaction merges and removes deleted entries.

```rust
struct CompactionManager {
    levels: Vec<Vec<SSTable>>,
    max_level: usize,
    level_size_multiplier: usize,
    base_level_size: usize,
}

impl CompactionManager {
    fn new(max_level: usize) -> Self {
        CompactionManager {
            levels: (0..=max_level).map(|_| Vec::new()).collect(),
            max_level,
            level_size_multiplier: 10,
            base_level_size: 4,
        }
    }

    fn should_compact(&self, level: usize) -> bool {
        if level >= self.max_level { return false; }
        let max_tables = if level == 0 {
            self.base_level_size
        } else {
            self.base_level_size * self.level_size_multiplier.pow(level as u32)
        };
        self.levels[level].len() > max_tables
    }
}
```

### 4.2 Monitoring and Metrics

```rust
use std::sync::atomic::{AtomicU64, Ordering};

struct Metrics {
    reads_total: AtomicU64,
    writes_total: AtomicU64,
    deletes_total: AtomicU64,
    read_latency_sum_us: AtomicU64,
    write_latency_sum_us: AtomicU64,
    compaction_runs: AtomicU64,
    bloom_filter_hits: AtomicU64,
    bloom_filter_misses: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
}

impl Metrics {
    fn new() -> Self {
        Metrics {
            reads_total: AtomicU64::new(0),
            writes_total: AtomicU64::new(0),
            deletes_total: AtomicU64::new(0),
            read_latency_sum_us: AtomicU64::new(0),
            write_latency_sum_us: AtomicU64::new(0),
            compaction_runs: AtomicU64::new(0),
            bloom_filter_hits: AtomicU64::new(0),
            bloom_filter_misses: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }
    }

    fn record_read(&self, latency_us: u64) {
        self.reads_total.fetch_add(1, Ordering::Relaxed);
        self.read_latency_sum_us.fetch_add(latency_us, Ordering::Relaxed);
    }
}
```

## Chapter 5: Network Protocol

### 5.1 Binary Protocol

For efficiency, inter-node communication uses a custom binary protocol:

The protocol frame format is:
- Magic bytes: 0xKV01 (4 bytes)
- Message type: u8
- Payload length: u32 (big-endian)
- Request ID: u64
- Payload: variable length
- CRC32: u32

Message types include GET/PUT/DELETE requests and responses, Raft RPCs (AppendEntries, RequestVote), heartbeats, and cluster membership operations.

### 5.2 Connection Pooling

Maintaining persistent connections between nodes reduces latency by avoiding TCP handshake overhead. The pool pre-warms connections on startup and maintains between min_size and max_size connections per remote node. Idle connections are reaped after a configurable timeout to prevent resource leaks.

## Chapter 6: Testing and Benchmarking

### 6.1 Deterministic Simulation Testing

Following the approach pioneered by FoundationDB, we use deterministic simulation to test distributed behavior without actual network communication. All sources of nondeterminism (network, disk I/O, timers) are abstracted behind interfaces that can be replaced with deterministic implementations during testing. A seeded random number generator drives all scheduling decisions, making test runs perfectly reproducible.

### 6.2 Jepsen-Style Consistency Testing

For validating linearizability, the testing framework spawns a cluster, runs concurrent client operations, introduces failures (network partitions, node crashes, clock skew), and verifies that the operation history is linearizable using the Wing-Gong algorithm. Combined with property-based testing and chaos engineering, this provides high confidence in correctness.

### 6.3 Performance Benchmarks

Standard benchmarks include:
- **YCSB workloads**: Read-heavy (workload B), write-heavy (workload A), scan-heavy (workload E)
- **Latency distribution**: p50, p99, p99.9 under varying load
- **Throughput scaling**: Linear scaling verification as nodes are added
- **Recovery time**: How quickly a node catches up after rejoining
- **Compaction impact**: Read/write latency during background compaction

Continue with a detailed analysis of performance optimization strategies for distributed key-value stores, including cache hierarchies, prefetching, and adaptive compaction scheduling.
"""

# ~1K token prompt: use the existing prompt-profile.txt
with open(r'C:\Users\fabia\Projects\kv-compact\prompt-profile.txt') as f:
    base_prompt = f.read().strip()

PROMPT_1K = base_prompt  # already ~881 tokens

# Generate all prompt variants
# Pad 10K prompt to actually reach ~10K tokens (code is ~2.6 tok/word)
# Current code prompt: ~3800 tokens. Need ~10K. Repeat with chapter variation.
PROMPT_10K_PADDED = PROMPT_10K
for extra_ch in range(7, 12):
    PROMPT_10K_PADDED += f"\n\n## Chapter {extra_ch}: Extended Analysis (Part {extra_ch - 6})\n\n"
    PROMPT_10K_PADDED += PROMPT_10K.split("## Chapter 2")[1].split("## Chapter 3")[0]

# keep_ratio = 1/compression_ratio
# 1x=1.0, ~1.25x=0.8, ~1.67x=0.6, 2.5x=0.4, 5x=0.2, 10x=0.1, 20x=0.05, 50x=0.02
KEEP_RATIOS = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02]

for size_name, full_text in [('1k', PROMPT_1K), ('10k', PROMPT_10K_PADDED)]:
    full_words = full_text.split()
    print(f"\n{size_name}: {len(full_words)} words (full)")
    for keep in KEEP_RATIOS:
        n_words = max(10, int(len(full_words) * keep))
        text = ' '.join(full_words[:n_words])
        payload = {
            'prompt': text,
            'n_predict': 50,
            'temperature': 0.0,
            'cache_prompt': True,
        }
        fname = os.path.join(TMP, f'req_{size_name}_keep{int(keep*100)}.json')
        with open(fname, 'w') as f:
            json.dump(payload, f)
        print(f"  keep={keep:.0%}: {n_words} words -> {fname}")

print("\nDone.")
