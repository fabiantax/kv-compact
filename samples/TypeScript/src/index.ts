/**
 * KV Compact - TypeScript Entry Point
 *
 * Main export file for the kv-compact TypeScript/WASM library
 */

export {
  KvCompactor,
  StreamingCompactor,
  createCompactor,
  estimateOptimalRatio,
  formatMetrics,
  type CompactionConfig,
  type CompactionMetrics,
  type CompactionResult,
  type StreamingConfig,
  type CacheZone,
} from './compaction';

// Re-export types for convenience
export type { CompactionConfig, CompactionMetrics, CompactionResult, StreamingConfig, CacheZone } from './compaction';

// Version
export const VERSION = '0.1.0';

// Default export
export default {
  KvCompactor,
  StreamingCompactor,
  createCompactor,
  estimateOptimalRatio,
  formatMetrics,
  VERSION,
};
