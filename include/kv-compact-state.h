// KV Cache State Buffer Parser and Writer
//
// Parses the binary format produced by llama_state_seq_get_data() into
// structured per-layer, per-head K/V data (as float32), and writes
// compacted state buffers back in the same format.
//
// State format (per stream):
//   [n_stream:u32]
//   per stream:
//     [cell_count:u32]
//     per cell: [pos:i32] [n_seq_id:u32] [seq_ids:i32*n_seq_id]
//     [v_trans:u32] [n_layer:u32]
//     per layer: [k_type:i32] [k_size_row:u64] [k_data:u8*cell_count*k_size_row]
//     per layer (non-trans): [v_type:i32] [v_size_row:u64] [v_data:u8*cell_count*v_size_row]
//     per layer (trans): [v_type:i32] [v_size_el:u32] [n_embd_v_gqa:u32] [v_data:u8*n_embd_v_gqa*cell_count*v_size_el]

#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

// Forward declare ggml types we need (avoid full ggml.h dependency in header)
#ifndef GGML_TYPE_F32
#define KV_COMPACT_GGML_TYPE_F32   0
#define KV_COMPACT_GGML_TYPE_F16   1
#define KV_COMPACT_GGML_TYPE_Q4_0  2
#define KV_COMPACT_GGML_TYPE_Q4_1  3
#define KV_COMPACT_GGML_TYPE_Q5_0  6
#define KV_COMPACT_GGML_TYPE_Q5_1  7
#define KV_COMPACT_GGML_TYPE_Q8_0  8
#define KV_COMPACT_GGML_TYPE_Q8_1  9
#else
#define KV_COMPACT_GGML_TYPE_F32   GGML_TYPE_F32
#define KV_COMPACT_GGML_TYPE_F16   GGML_TYPE_F16
#define KV_COMPACT_GGML_TYPE_Q4_0  GGML_TYPE_Q4_0
#define KV_COMPACT_GGML_TYPE_Q4_1  GGML_TYPE_Q4_1
#define KV_COMPACT_GGML_TYPE_Q5_0  GGML_TYPE_Q5_0
#define KV_COMPACT_GGML_TYPE_Q5_1  GGML_TYPE_Q5_1
#define KV_COMPACT_GGML_TYPE_Q8_0  GGML_TYPE_Q8_0
#define KV_COMPACT_GGML_TYPE_Q8_1  GGML_TYPE_Q8_1
#endif

// Block quantization constants
// All ggml block-quantized types use blocks of 32 elements
#define KV_COMPACT_QK 32

// Block sizes in bytes:
//   Q4_0: f16 scale + 16 bytes (32 nibbles) = 18 bytes / 32 elements
//   Q4_1: f16 scale + f16 min + 16 bytes    = 20 bytes / 32 elements
//   Q5_0: f16 scale + 4 bytes (high bits) + 16 bytes = 22 bytes / 32 elements
//   Q5_1: f16 scale + f16 min + 4 bytes + 16 bytes   = 24 bytes / 32 elements
//   Q8_0: f16 scale + 32 int8              = 34 bytes / 32 elements
//   Q8_1: f32 scale + f32 sum + 32 int8    = 40 bytes / 32 elements
#define KV_COMPACT_Q4_0_BLOCK_SIZE 18
#define KV_COMPACT_Q4_1_BLOCK_SIZE 20
#define KV_COMPACT_Q5_0_BLOCK_SIZE 22
#define KV_COMPACT_Q5_1_BLOCK_SIZE 24
#define KV_COMPACT_Q8_0_BLOCK_SIZE 34
#define KV_COMPACT_Q8_1_BLOCK_SIZE 40

// ============================================================================
// Cell metadata
// ============================================================================

struct kv_cell_meta {
    int32_t  pos;
    int32_t  ext_x = 0;  // IMROPE: x position (M-RoPE spatial)
    int32_t  ext_y = 0;  // IMROPE: y position (M-RoPE spatial)
    std::vector<int32_t> seq_ids;
};

// ============================================================================
// Parsed layer data — all heads concatenated, converted to float32
// ============================================================================

struct kv_layer_data {
    int32_t  k_type;        // original ggml_type
    int32_t  v_type;
    uint64_t k_size_row;    // bytes per row in original format
    uint64_t v_size_row;    // bytes per row (non-transposed only)
    uint32_t v_size_el;     // bytes per element (transposed only)
    uint32_t n_embd_v_gqa;  // V embedding size (transposed only)

    // Float32 data: [cell_count, n_embd_k_gqa] and [cell_count, n_embd_v_gqa]
    // V is always stored as [cell_count, n_embd_v_gqa] regardless of v_trans
    std::vector<float> K;   // all heads concatenated, row = token
    std::vector<float> V;   // all heads concatenated, row = token (transposed from storage if needed)

    int n_embd_k_gqa() const { return K.empty() ? 0 : (int)(K.size() / cell_count); }
    int n_embd_v_gqa_computed() const { return V.empty() ? 0 : (int)(V.size() / cell_count); }

    uint32_t cell_count = 0;  // set during parsing
};

// ============================================================================
// Parsed KV state
// ============================================================================

struct parsed_kv_state {
    uint32_t n_stream = 0;

    // Per-stream data (usually just 1 stream for seq_id=0)
    struct stream_data {
        uint32_t cell_count = 0;
        std::vector<kv_cell_meta> cells;
        uint32_t v_trans = 0;
        uint32_t n_layer = 0;
        std::vector<kv_layer_data> layers;
    };
    std::vector<stream_data> streams;

    // Raw state buffer (kept for round-trip; overwritten by write_state)
    std::vector<uint8_t> raw;

    // Trailing data after KV section (e.g., recurrent section for hybrid SSM+MoE models)
    std::vector<uint8_t> trailing_data;

    // ---- Parsing ----

    // n_pos_per_embd: 1 for normal RoPE, 4 for M-RoPE/IMROPE
    //   When > 1, each cell has extra 8 bytes (llama_kv_cell_ext: x,y positions)
    //   after n_seq_id and before seq_ids
    bool parse(const uint8_t * data, size_t size, uint32_t n_pos_per_embd = 1) {
        raw.assign(data, data + size);

        const uint8_t * ptr = data;
        const uint8_t * end = data + size;

        if (!read_val(ptr, end, n_stream)) return false;
        streams.resize(n_stream);

        for (uint32_t s = 0; s < n_stream; s++) {
            auto & sd = streams[s];
            if (!read_val(ptr, end, sd.cell_count)) return false;

            if (sd.cell_count == 0) continue;

            // Parse cell metadata
            sd.cells.resize(sd.cell_count);
            for (uint32_t c = 0; c < sd.cell_count; c++) {
                auto & cell = sd.cells[c];
                if (!read_val(ptr, end, cell.pos)) return false;
                uint32_t n_seq_id;
                if (!read_val(ptr, end, n_seq_id)) return false;
                // Read llama_kv_cell_ext (x,y) for M-RoPE/IMROPE models
                if (n_pos_per_embd > 1) {
                    if (!read_val(ptr, end, cell.ext_x)) return false;
                    if (!read_val(ptr, end, cell.ext_y)) return false;
                }
                cell.seq_ids.resize(n_seq_id);
                for (uint32_t i = 0; i < n_seq_id; i++) {
                    if (!read_val(ptr, end, cell.seq_ids[i])) return false;
                }
            }

            // v_trans and n_layer
            if (!read_val(ptr, end, sd.v_trans)) return false;
            if (!read_val(ptr, end, sd.n_layer)) return false;

            sd.layers.resize(sd.n_layer);

            // Parse K data per layer
            for (uint32_t l = 0; l < sd.n_layer; l++) {
                auto & ld = sd.layers[l];
                ld.cell_count = sd.cell_count;

                if (!read_val(ptr, end, ld.k_type)) return false;
                if (!read_val(ptr, end, ld.k_size_row)) return false;

                const size_t k_data_size = sd.cell_count * ld.k_size_row;
                if (ptr + k_data_size > end) return false;

                // Convert to float32
                const int n_floats_per_row = elements_per_row(ld.k_type, ld.k_size_row);
                ld.K.resize((size_t)sd.cell_count * n_floats_per_row);
                convert_to_f32(ptr, ld.k_type, ld.K.data(), sd.cell_count * n_floats_per_row);

                ptr += k_data_size;
            }

            // Parse V data per layer
            if (!sd.v_trans) {
                for (uint32_t l = 0; l < sd.n_layer; l++) {
                    auto & ld = sd.layers[l];

                    if (!read_val(ptr, end, ld.v_type)) return false;
                    if (!read_val(ptr, end, ld.v_size_row)) return false;

                    const size_t v_data_size = sd.cell_count * ld.v_size_row;
                    if (ptr + v_data_size > end) return false;

                    const int n_floats_per_row = elements_per_row(ld.v_type, ld.v_size_row);
                    ld.V.resize((size_t)sd.cell_count * n_floats_per_row);
                    convert_to_f32(ptr, ld.v_type, ld.V.data(), sd.cell_count * n_floats_per_row);

                    ptr += v_data_size;
                }
            } else {
                // Transposed V
                for (uint32_t l = 0; l < sd.n_layer; l++) {
                    auto & ld = sd.layers[l];

                    if (!read_val(ptr, end, ld.v_type)) return false;
                    if (!read_val(ptr, end, ld.v_size_el)) return false;
                    if (!read_val(ptr, end, ld.n_embd_v_gqa)) return false;

                    // For non-block types: n_embd * cell_count * element_size
                    // For block types: n_embd * ceil(cell_count/QK) * block_size
                    const size_t v_data_size = is_quantized(ld.v_type)
                        ? (size_t)ld.n_embd_v_gqa * row_bytes_for(ld.v_type, sd.cell_count)
                        : (size_t)ld.n_embd_v_gqa * sd.cell_count * ld.v_size_el;
                    if (ptr + v_data_size > end) return false;

                    // Transpose from [embd][token] to [token][embd]
                    ld.V.resize((size_t)sd.cell_count * ld.n_embd_v_gqa);
                    transpose_v_to_f32(ptr, ld.v_type, ld.V.data(),
                                       sd.cell_count, ld.n_embd_v_gqa, ld.v_size_el);

                    ptr += v_data_size;
                }
            }
        }

        // Save any remaining data (e.g., recurrent section for hybrid SSM+MoE models)
        if (ptr < end) {
            trailing_data.assign(ptr, end);
        }

        return true;
    }

    // ---- Extract per-head data ----

    // Get K for a specific head: output [cell_count, d_k]
    void get_k_head(int stream, int layer, int head, int d_k, std::vector<float> & out) const {
        const auto & ld = streams[stream].layers[layer];
        const int n_embd = ld.n_embd_k_gqa();
        const int cc = ld.cell_count;
        out.resize(cc * d_k);
        for (int i = 0; i < cc; i++) {
            memcpy(out.data() + i * d_k,
                   ld.K.data() + i * n_embd + head * d_k,
                   d_k * sizeof(float));
        }
    }

    // Get V for a specific head: output [cell_count, d_v]
    void get_v_head(int stream, int layer, int head, int d_v, std::vector<float> & out) const {
        const auto & ld = streams[stream].layers[layer];
        const int n_embd = ld.n_embd_v_gqa_computed();
        const int cc = ld.cell_count;
        out.resize(cc * d_v);
        for (int i = 0; i < cc; i++) {
            memcpy(out.data() + i * d_v,
                   ld.V.data() + i * n_embd + head * d_v,
                   d_v * sizeof(float));
        }
    }

    // ---- Static utility functions (public for use by writer and tests) ----

    // Returns bytes per element for non-block types, or 0 for block-quantized types
    static int type_size(int32_t type) {
        if (type == KV_COMPACT_GGML_TYPE_F32) return 4;
        if (type == KV_COMPACT_GGML_TYPE_F16) return 2;
        return 0; // block-quantized — use block_size() and elements_per_row() instead
    }

    // Returns block size in bytes for quantized types, or element size for F32/F16
    static int block_size(int32_t type) {
        switch (type) {
            case KV_COMPACT_GGML_TYPE_F32:  return 4;
            case KV_COMPACT_GGML_TYPE_F16:  return 2;
            case KV_COMPACT_GGML_TYPE_Q4_0: return KV_COMPACT_Q4_0_BLOCK_SIZE;
            case KV_COMPACT_GGML_TYPE_Q4_1: return KV_COMPACT_Q4_1_BLOCK_SIZE;
            case KV_COMPACT_GGML_TYPE_Q5_0: return KV_COMPACT_Q5_0_BLOCK_SIZE;
            case KV_COMPACT_GGML_TYPE_Q5_1: return KV_COMPACT_Q5_1_BLOCK_SIZE;
            case KV_COMPACT_GGML_TYPE_Q8_0: return KV_COMPACT_Q8_0_BLOCK_SIZE;
            case KV_COMPACT_GGML_TYPE_Q8_1: return KV_COMPACT_Q8_1_BLOCK_SIZE;
            default: return 0;
        }
    }

    // Returns elements per block (1 for non-block types, QK for block types)
    static int elements_per_block(int32_t type) {
        switch (type) {
            case KV_COMPACT_GGML_TYPE_F32:  return 1;
            case KV_COMPACT_GGML_TYPE_F16:  return 1;
            default: return KV_COMPACT_QK;
        }
    }

    // Compute number of float elements from row byte size and type
    static int elements_per_row(int32_t type, uint64_t row_bytes) {
        int bs = block_size(type);
        int epb = elements_per_block(type);
        if (bs == 0) return 0;
        return (int)(row_bytes / bs) * epb;
    }

    // Is this a block-quantized type?
    static bool is_quantized(int32_t type) {
        return type != KV_COMPACT_GGML_TYPE_F32 && type != KV_COMPACT_GGML_TYPE_F16;
    }

    // Row byte size for a given number of float elements
    static uint64_t row_bytes_for(int32_t type, int n_elements) {
        if (type == KV_COMPACT_GGML_TYPE_F32)  return n_elements * 4;
        if (type == KV_COMPACT_GGML_TYPE_F16)  return n_elements * 2;
        // Block-quantized: n_elements must be a multiple of QK
        int n_blocks = n_elements / KV_COMPACT_QK;
        return (uint64_t)n_blocks * block_size(type);
    }

    // Convert raw data to float32 (supports F32, F16, and block-quantized types)
    static void convert_to_f32(const uint8_t * src, int32_t type, float * dst, size_t n) {
        if (type == KV_COMPACT_GGML_TYPE_F32) {
            memcpy(dst, src, n * sizeof(float));
        } else if (type == KV_COMPACT_GGML_TYPE_F16) {
            const uint16_t * f16 = (const uint16_t *) src;
            for (size_t i = 0; i < n; i++) {
                dst[i] = f16_to_f32(f16[i]);
            }
        } else if (type == KV_COMPACT_GGML_TYPE_Q4_0) {
            dequantize_q4_0(src, dst, n);
        } else if (type == KV_COMPACT_GGML_TYPE_Q4_1) {
            dequantize_q4_1(src, dst, n);
        } else if (type == KV_COMPACT_GGML_TYPE_Q5_0) {
            dequantize_q5_0(src, dst, n);
        } else if (type == KV_COMPACT_GGML_TYPE_Q5_1) {
            dequantize_q5_1(src, dst, n);
        } else if (type == KV_COMPACT_GGML_TYPE_Q8_0) {
            dequantize_q8_0(src, dst, n);
        } else if (type == KV_COMPACT_GGML_TYPE_Q8_1) {
            dequantize_q8_1(src, dst, n);
        } else {
            memset(dst, 0, n * sizeof(float));
        }
    }

    // ---- Dequantize block types to F32 ----

    // Q4_0: block = [f16 scale][16 bytes of 32 nibbles]
    static void dequantize_q4_0(const uint8_t * src, float * dst, size_t n) {
        const int nb = (int)(n / KV_COMPACT_QK);
        for (int i = 0; i < nb; i++) {
            const uint8_t * block = src + i * KV_COMPACT_Q4_0_BLOCK_SIZE;
            uint16_t scale_f16;
            memcpy(&scale_f16, block, sizeof(uint16_t));
            float scale = f16_to_f32(scale_f16);
            const uint8_t * qs = block + 2;

            for (int j = 0; j < KV_COMPACT_QK / 2; j++) {
                int v0 = (qs[j] & 0x0F) - 8;
                int v1 = (qs[j] >> 4)    - 8;
                dst[i * KV_COMPACT_QK + j]                    = scale * v0;
                dst[i * KV_COMPACT_QK + j + KV_COMPACT_QK / 2] = scale * v1;
            }
        }
    }

    // Q4_1: block = [f16 scale][f16 min][16 bytes of 32 nibbles]
    static void dequantize_q4_1(const uint8_t * src, float * dst, size_t n) {
        const int nb = (int)(n / KV_COMPACT_QK);
        for (int i = 0; i < nb; i++) {
            const uint8_t * block = src + i * KV_COMPACT_Q4_1_BLOCK_SIZE;
            uint16_t scale_f16, min_f16;
            memcpy(&scale_f16, block, sizeof(uint16_t));
            memcpy(&min_f16, block + 2, sizeof(uint16_t));
            float scale = f16_to_f32(scale_f16);
            float min   = f16_to_f32(min_f16);
            const uint8_t * qs = block + 4;

            for (int j = 0; j < KV_COMPACT_QK / 2; j++) {
                int v0 = (qs[j] & 0x0F);
                int v1 = (qs[j] >> 4);
                dst[i * KV_COMPACT_QK + j]                    = scale * v0 + min;
                dst[i * KV_COMPACT_QK + j + KV_COMPACT_QK / 2] = scale * v1 + min;
            }
        }
    }

    // Q5_0: block = [f16 scale][4 bytes high bits][16 bytes low nibbles]
    static void dequantize_q5_0(const uint8_t * src, float * dst, size_t n) {
        const int nb = (int)(n / KV_COMPACT_QK);
        for (int i = 0; i < nb; i++) {
            const uint8_t * block = src + i * KV_COMPACT_Q5_0_BLOCK_SIZE;
            uint16_t scale_f16;
            memcpy(&scale_f16, block, sizeof(uint16_t));
            float scale = f16_to_f32(scale_f16);
            const uint8_t * qh = block + 2;      // 4 bytes of high bits
            const uint8_t * qs = block + 2 + 4;   // 16 bytes of low nibbles

            uint32_t hbits;
            memcpy(&hbits, qh, sizeof(uint32_t));

            for (int j = 0; j < KV_COMPACT_QK / 2; j++) {
                int lo0 = (qs[j] & 0x0F);
                int lo1 = (qs[j] >> 4);
                int hi0 = (hbits >> j) & 1;
                int hi1 = (hbits >> (j + KV_COMPACT_QK / 2)) & 1;
                int v0 = (lo0 | (hi0 << 4)) - 16;
                int v1 = (lo1 | (hi1 << 4)) - 16;
                dst[i * KV_COMPACT_QK + j]                    = scale * v0;
                dst[i * KV_COMPACT_QK + j + KV_COMPACT_QK / 2] = scale * v1;
            }
        }
    }

    // Q5_1: block = [f16 scale][f16 min][4 bytes high bits][16 bytes low nibbles]
    static void dequantize_q5_1(const uint8_t * src, float * dst, size_t n) {
        const int nb = (int)(n / KV_COMPACT_QK);
        for (int i = 0; i < nb; i++) {
            const uint8_t * block = src + i * KV_COMPACT_Q5_1_BLOCK_SIZE;
            uint16_t scale_f16, min_f16;
            memcpy(&scale_f16, block, sizeof(uint16_t));
            memcpy(&min_f16, block + 2, sizeof(uint16_t));
            float scale = f16_to_f32(scale_f16);
            float min   = f16_to_f32(min_f16);
            const uint8_t * qh = block + 4;
            const uint8_t * qs = block + 4 + 4;

            uint32_t hbits;
            memcpy(&hbits, qh, sizeof(uint32_t));

            for (int j = 0; j < KV_COMPACT_QK / 2; j++) {
                int lo0 = (qs[j] & 0x0F);
                int lo1 = (qs[j] >> 4);
                int hi0 = (hbits >> j) & 1;
                int hi1 = (hbits >> (j + KV_COMPACT_QK / 2)) & 1;
                int v0 = (lo0 | (hi0 << 4));
                int v1 = (lo1 | (hi1 << 4));
                dst[i * KV_COMPACT_QK + j]                    = scale * v0 + min;
                dst[i * KV_COMPACT_QK + j + KV_COMPACT_QK / 2] = scale * v1 + min;
            }
        }
    }

    // Q8_0: block = [f16 scale][32 int8]
    static void dequantize_q8_0(const uint8_t * src, float * dst, size_t n) {
        const int nb = (int)(n / KV_COMPACT_QK);
        for (int i = 0; i < nb; i++) {
            const uint8_t * block = src + i * KV_COMPACT_Q8_0_BLOCK_SIZE;
            uint16_t scale_f16;
            memcpy(&scale_f16, block, sizeof(uint16_t));
            float scale = f16_to_f32(scale_f16);
            const int8_t * qs = (const int8_t *)(block + 2);

            for (int j = 0; j < KV_COMPACT_QK; j++) {
                dst[i * KV_COMPACT_QK + j] = scale * qs[j];
            }
        }
    }

    // Q8_1: block = [f32 scale][f32 sum][32 int8]
    static void dequantize_q8_1(const uint8_t * src, float * dst, size_t n) {
        const int nb = (int)(n / KV_COMPACT_QK);
        for (int i = 0; i < nb; i++) {
            const uint8_t * block = src + i * KV_COMPACT_Q8_1_BLOCK_SIZE;
            float scale;
            memcpy(&scale, block, sizeof(float));
            // skip f32 sum at block+4
            const int8_t * qs = (const int8_t *)(block + 8);

            for (int j = 0; j < KV_COMPACT_QK; j++) {
                dst[i * KV_COMPACT_QK + j] = scale * qs[j];
            }
        }
    }

    // Transpose V from [embd][cell] to [cell][embd] and convert to F32
    //
    // For non-block types (F32, F16): layout is [n_embd][cell_count] of individual elements.
    // For block-quantized types: each embedding dimension d has cell_count values stored
    // as consecutive blocks. The data for dimension d starts at offset d * row_bytes,
    // where row_bytes = ceil(cell_count / QK) * block_size.
    static void transpose_v_to_f32(const uint8_t * src, int32_t type, float * dst,
                                   uint32_t cell_count, uint32_t n_embd,
                                   uint32_t v_size_el = 0) {
        if (type == KV_COMPACT_GGML_TYPE_F32) {
            const float * f = (const float *) src;
            for (uint32_t d = 0; d < n_embd; d++) {
                for (uint32_t c = 0; c < cell_count; c++) {
                    dst[c * n_embd + d] = f[d * cell_count + c];
                }
            }
        } else if (type == KV_COMPACT_GGML_TYPE_F16) {
            const uint16_t * f16 = (const uint16_t *) src;
            for (uint32_t d = 0; d < n_embd; d++) {
                for (uint32_t c = 0; c < cell_count; c++) {
                    dst[c * n_embd + d] = f16_to_f32(f16[d * cell_count + c]);
                }
            }
        } else if (is_quantized(type)) {
            // For transposed quantized V: each embedding dimension stores cell_count
            // values as a sequence of blocks. v_size_el is the byte stride per cell
            // in the serialized format — the row for dimension d starts at d * cell_count * v_size_el.
            // But actually the state format stores v_data as n_embd_v_gqa * cell_count * v_size_el
            // contiguously with v_size_el being the element byte size.
            //
            // For block-quantized transposed V, llama.cpp stores the data such that
            // each dimension d has its cell values packed into blocks.
            // Total data per dimension = ceil(cell_count/QK) * block_size_bytes
            const uint64_t bytes_per_dim = row_bytes_for(type, cell_count);
            std::vector<float> tmp(cell_count);
            for (uint32_t d = 0; d < n_embd; d++) {
                const uint8_t * dim_data = src + d * bytes_per_dim;
                convert_to_f32(dim_data, type, tmp.data(), cell_count);
                for (uint32_t c = 0; c < cell_count; c++) {
                    dst[c * n_embd + d] = tmp[c];
                }
            }
        } else {
            memset(dst, 0, (size_t)cell_count * n_embd * sizeof(float));
        }
    }

    // Simple F16 → F32 conversion (IEEE 754)
    static float f16_to_f32(uint16_t h) {
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;

        if (exp == 0) {
            if (mant == 0) {
                // Zero
                uint32_t result = sign;
                float f;
                memcpy(&f, &result, 4);
                return f;
            }
            // Denormalized
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            exp = exp + (127 - 15);
        } else if (exp == 31) {
            // Inf / NaN
            uint32_t result = sign | 0x7F800000 | (mant << 13);
            float f;
            memcpy(&f, &result, 4);
            return f;
        } else {
            exp = exp + (127 - 15);
        }

        uint32_t result = sign | (exp << 23) | (mant << 13);
        float f;
        memcpy(&f, &result, 4);
        return f;
    }

private:
    template<typename T>
    static bool read_val(const uint8_t *& ptr, const uint8_t * end, T & val) {
        if (ptr + sizeof(T) > end) return false;
        memcpy(&val, ptr, sizeof(T));
        ptr += sizeof(T);
        return true;
    }
};

// ============================================================================
// State buffer writer — builds compacted state from compaction results
// ============================================================================

// Convert float32 to F16 (IEEE 754)
static uint16_t f32_to_f16(float val) {
    uint32_t f;
    memcpy(&f, &val, 4);

    uint32_t sign = (f >> 16) & 0x8000;
    int32_t  exp  = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (f >> 13) & 0x3FF;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign; // underflow to zero
        mant = (mant | 0x400) >> (1 - exp);
        return (uint16_t)(sign | mant);
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00); // overflow to inf
    }

    return (uint16_t)(sign | (exp << 10) | mant);
}

// ============================================================================
// Quantize F32 back to block types
// ============================================================================

// Quantize F32 to Q8_0: block = [f16 scale][32 int8]
static void quantize_q8_0(const float * src, uint8_t * dst, size_t n) {
    const int nb = (int)(n / KV_COMPACT_QK);
    for (int i = 0; i < nb; i++) {
        const float * block_src = src + i * KV_COMPACT_QK;
        uint8_t * block_dst = dst + i * KV_COMPACT_Q8_0_BLOCK_SIZE;

        // Find max absolute value for scale
        float amax = 0.0f;
        for (int j = 0; j < KV_COMPACT_QK; j++) {
            float av = fabsf(block_src[j]);
            if (av > amax) amax = av;
        }
        float scale = amax / 127.0f;
        float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

        // Write f16 scale
        uint16_t scale_f16 = f32_to_f16(scale);
        memcpy(block_dst, &scale_f16, sizeof(uint16_t));

        // Quantize values
        int8_t * qs = (int8_t *)(block_dst + 2);
        for (int j = 0; j < KV_COMPACT_QK; j++) {
            float v = block_src[j] * inv_scale;
            int vi = (int)roundf(v);
            if (vi < -128) vi = -128;
            if (vi >  127) vi =  127;
            qs[j] = (int8_t)vi;
        }
    }
}

// Quantize F32 to Q4_0: block = [f16 scale][16 bytes of 32 nibbles]
static void quantize_q4_0(const float * src, uint8_t * dst, size_t n) {
    const int nb = (int)(n / KV_COMPACT_QK);
    for (int i = 0; i < nb; i++) {
        const float * block_src = src + i * KV_COMPACT_QK;
        uint8_t * block_dst = dst + i * KV_COMPACT_Q4_0_BLOCK_SIZE;

        // Find max absolute value for scale
        float amax = 0.0f;
        for (int j = 0; j < KV_COMPACT_QK; j++) {
            float av = fabsf(block_src[j]);
            if (av > amax) amax = av;
        }
        float scale = amax / 8.0f;
        float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

        // Write f16 scale
        uint16_t scale_f16 = f32_to_f16(scale);
        memcpy(block_dst, &scale_f16, sizeof(uint16_t));

        // Quantize: first half goes to low nibbles, second half to high nibbles
        uint8_t * qs = block_dst + 2;
        for (int j = 0; j < KV_COMPACT_QK / 2; j++) {
            float v0 = block_src[j] * inv_scale;
            float v1 = block_src[j + KV_COMPACT_QK / 2] * inv_scale;
            int vi0 = (int)roundf(v0) + 8;
            int vi1 = (int)roundf(v1) + 8;
            if (vi0 < 0)  vi0 = 0;
            if (vi0 > 15) vi0 = 15;
            if (vi1 < 0)  vi1 = 0;
            if (vi1 > 15) vi1 = 15;
            qs[j] = (uint8_t)(vi0 | (vi1 << 4));
        }
    }
}

// Quantize F32 to Q4_1: block = [f16 scale][f16 min][16 bytes of 32 nibbles]
static void quantize_q4_1(const float * src, uint8_t * dst, size_t n) {
    const int nb = (int)(n / KV_COMPACT_QK);
    for (int i = 0; i < nb; i++) {
        const float * block_src = src + i * KV_COMPACT_QK;
        uint8_t * block_dst = dst + i * KV_COMPACT_Q4_1_BLOCK_SIZE;

        float vmin = block_src[0], vmax = block_src[0];
        for (int j = 1; j < KV_COMPACT_QK; j++) {
            if (block_src[j] < vmin) vmin = block_src[j];
            if (block_src[j] > vmax) vmax = block_src[j];
        }
        float scale = (vmax - vmin) / 15.0f;
        float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

        uint16_t scale_f16 = f32_to_f16(scale);
        uint16_t min_f16   = f32_to_f16(vmin);
        memcpy(block_dst, &scale_f16, sizeof(uint16_t));
        memcpy(block_dst + 2, &min_f16, sizeof(uint16_t));

        uint8_t * qs = block_dst + 4;
        for (int j = 0; j < KV_COMPACT_QK / 2; j++) {
            float v0 = (block_src[j] - vmin) * inv_scale;
            float v1 = (block_src[j + KV_COMPACT_QK / 2] - vmin) * inv_scale;
            int vi0 = (int)roundf(v0);
            int vi1 = (int)roundf(v1);
            if (vi0 < 0)  vi0 = 0;
            if (vi0 > 15) vi0 = 15;
            if (vi1 < 0)  vi1 = 0;
            if (vi1 > 15) vi1 = 15;
            qs[j] = (uint8_t)(vi0 | (vi1 << 4));
        }
    }
}

// Quantize F32 to Q5_0: block = [f16 scale][4 bytes high bits][16 bytes low nibbles]
static void quantize_q5_0(const float * src, uint8_t * dst, size_t n) {
    const int nb = (int)(n / KV_COMPACT_QK);
    for (int i = 0; i < nb; i++) {
        const float * block_src = src + i * KV_COMPACT_QK;
        uint8_t * block_dst = dst + i * KV_COMPACT_Q5_0_BLOCK_SIZE;

        float amax = 0.0f;
        for (int j = 0; j < KV_COMPACT_QK; j++) {
            float av = fabsf(block_src[j]);
            if (av > amax) amax = av;
        }
        float scale = amax / 16.0f;
        float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

        uint16_t scale_f16 = f32_to_f16(scale);
        memcpy(block_dst, &scale_f16, sizeof(uint16_t));

        uint8_t * qh = block_dst + 2;
        uint8_t * qs = block_dst + 2 + 4;
        uint32_t hbits = 0;

        for (int j = 0; j < KV_COMPACT_QK / 2; j++) {
            float v0 = block_src[j] * inv_scale;
            float v1 = block_src[j + KV_COMPACT_QK / 2] * inv_scale;
            int vi0 = (int)roundf(v0) + 16;
            int vi1 = (int)roundf(v1) + 16;
            if (vi0 < 0)  vi0 = 0;
            if (vi0 > 31) vi0 = 31;
            if (vi1 < 0)  vi1 = 0;
            if (vi1 > 31) vi1 = 31;
            qs[j] = (uint8_t)((vi0 & 0x0F) | ((vi1 & 0x0F) << 4));
            hbits |= ((uint32_t)(vi0 >> 4) << j);
            hbits |= ((uint32_t)(vi1 >> 4) << (j + KV_COMPACT_QK / 2));
        }
        memcpy(qh, &hbits, sizeof(uint32_t));
    }
}

// Quantize F32 to Q5_1: block = [f16 scale][f16 min][4 bytes high bits][16 bytes low nibbles]
static void quantize_q5_1(const float * src, uint8_t * dst, size_t n) {
    const int nb = (int)(n / KV_COMPACT_QK);
    for (int i = 0; i < nb; i++) {
        const float * block_src = src + i * KV_COMPACT_QK;
        uint8_t * block_dst = dst + i * KV_COMPACT_Q5_1_BLOCK_SIZE;

        float vmin = block_src[0], vmax = block_src[0];
        for (int j = 1; j < KV_COMPACT_QK; j++) {
            if (block_src[j] < vmin) vmin = block_src[j];
            if (block_src[j] > vmax) vmax = block_src[j];
        }
        float scale = (vmax - vmin) / 31.0f;
        float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

        uint16_t scale_f16 = f32_to_f16(scale);
        uint16_t min_f16   = f32_to_f16(vmin);
        memcpy(block_dst, &scale_f16, sizeof(uint16_t));
        memcpy(block_dst + 2, &min_f16, sizeof(uint16_t));

        uint8_t * qh = block_dst + 4;
        uint8_t * qs = block_dst + 4 + 4;
        uint32_t hbits = 0;

        for (int j = 0; j < KV_COMPACT_QK / 2; j++) {
            float v0 = (block_src[j] - vmin) * inv_scale;
            float v1 = (block_src[j + KV_COMPACT_QK / 2] - vmin) * inv_scale;
            int vi0 = (int)roundf(v0);
            int vi1 = (int)roundf(v1);
            if (vi0 < 0)  vi0 = 0;
            if (vi0 > 31) vi0 = 31;
            if (vi1 < 0)  vi1 = 0;
            if (vi1 > 31) vi1 = 31;
            qs[j] = (uint8_t)((vi0 & 0x0F) | ((vi1 & 0x0F) << 4));
            hbits |= ((uint32_t)(vi0 >> 4) << j);
            hbits |= ((uint32_t)(vi1 >> 4) << (j + KV_COMPACT_QK / 2));
        }
        memcpy(qh, &hbits, sizeof(uint32_t));
    }
}

// Quantize F32 to Q8_1: block = [f32 scale][f32 sum][32 int8]
static void quantize_q8_1(const float * src, uint8_t * dst, size_t n) {
    const int nb = (int)(n / KV_COMPACT_QK);
    for (int i = 0; i < nb; i++) {
        const float * block_src = src + i * KV_COMPACT_QK;
        uint8_t * block_dst = dst + i * KV_COMPACT_Q8_1_BLOCK_SIZE;

        float amax = 0.0f;
        for (int j = 0; j < KV_COMPACT_QK; j++) {
            float av = fabsf(block_src[j]);
            if (av > amax) amax = av;
        }
        float scale = amax / 127.0f;
        float inv_scale = (scale != 0.0f) ? 1.0f / scale : 0.0f;

        memcpy(block_dst, &scale, sizeof(float));

        int8_t * qs = (int8_t *)(block_dst + 8);
        float sum = 0.0f;
        for (int j = 0; j < KV_COMPACT_QK; j++) {
            float v = block_src[j] * inv_scale;
            int vi = (int)roundf(v);
            if (vi < -128) vi = -128;
            if (vi >  127) vi =  127;
            qs[j] = (int8_t)vi;
            sum += (float)qs[j] * scale;
        }
        memcpy(block_dst + 4, &sum, sizeof(float));
    }
}

// Generic F32 → quantized type conversion
// n must be a multiple of QK for block-quantized types
static void convert_from_f32(const float * src, int32_t type, uint8_t * dst, size_t n) {
    if (type == KV_COMPACT_GGML_TYPE_F32) {
        memcpy(dst, src, n * sizeof(float));
    } else if (type == KV_COMPACT_GGML_TYPE_F16) {
        uint16_t * f16 = (uint16_t *) dst;
        for (size_t i = 0; i < n; i++) {
            f16[i] = f32_to_f16(src[i]);
        }
    } else if (type == KV_COMPACT_GGML_TYPE_Q4_0) {
        quantize_q4_0(src, dst, n);
    } else if (type == KV_COMPACT_GGML_TYPE_Q4_1) {
        quantize_q4_1(src, dst, n);
    } else if (type == KV_COMPACT_GGML_TYPE_Q5_0) {
        quantize_q5_0(src, dst, n);
    } else if (type == KV_COMPACT_GGML_TYPE_Q5_1) {
        quantize_q5_1(src, dst, n);
    } else if (type == KV_COMPACT_GGML_TYPE_Q8_0) {
        quantize_q8_0(src, dst, n);
    } else if (type == KV_COMPACT_GGML_TYPE_Q8_1) {
        quantize_q8_1(src, dst, n);
    }
}

// Build a compacted state buffer from original parsed state + compaction results
//
// For each layer:
//   - K: copy original K rows for selected indices only
//   - V: write C_v (refitted values) for each head at selected positions
//
// selected_indices: [t] shared across all heads within a layer
// cv_per_head: [n_head_kv][t * d_v] refitted values per head
//
// Returns the new state buffer ready for llama_state_seq_set_data()
static std::vector<uint8_t> build_compacted_state(
        const parsed_kv_state & state,
        const std::vector<int> & selected_indices,
        // Per-layer, per-head C_v: cv_all[layer][head] = vector<float> of [t * d_v]
        const std::vector<std::vector<std::vector<float>>> & cv_all,
        int n_head_kv, int d_k, int d_v,
        uint32_t n_pos_per_embd = 1,
        // Optional merged keys (token merging): ck_all[layer][head] = vector<float> of [t * d_k]
        // When non-empty, writes these instead of original K at selected_indices
        const std::vector<std::vector<std::vector<float>>> * ck_all = nullptr) {

    const int t = (int) selected_indices.size();

    // Estimate output size (generous overestimate)
    std::vector<uint8_t> out;
    out.reserve(state.raw.size()); // at most same size as original

    auto write = [&](const void * data, size_t sz) {
        const uint8_t * p = (const uint8_t *) data;
        out.insert(out.end(), p, p + sz);
    };

    // Write n_stream
    write(&state.n_stream, sizeof(state.n_stream));

    for (uint32_t s = 0; s < state.n_stream; s++) {
        const auto & sd = state.streams[s];

        if (sd.cell_count == 0) {
            uint32_t zero = 0;
            write(&zero, sizeof(zero));
            continue;
        }

        // Write compacted cell count
        uint32_t new_cell_count = (uint32_t) t;
        write(&new_cell_count, sizeof(new_cell_count));

        // Write cell metadata for selected indices only
        for (int j = 0; j < t; j++) {
            const auto & cell = sd.cells[selected_indices[j]];
            write(&cell.pos, sizeof(cell.pos));
            uint32_t n_seq_id = (uint32_t) cell.seq_ids.size();
            write(&n_seq_id, sizeof(n_seq_id));
            // Write IMROPE ext data (x,y positions) if applicable
            if (n_pos_per_embd > 1) {
                write(&cell.ext_x, sizeof(cell.ext_x));
                write(&cell.ext_y, sizeof(cell.ext_y));
            }
            for (const auto & sid : cell.seq_ids) {
                write(&sid, sizeof(sid));
            }
        }

        // Write v_trans and n_layer
        write(&sd.v_trans, sizeof(sd.v_trans));
        write(&sd.n_layer, sizeof(sd.n_layer));

        // Write K data per layer — original K rows at selected indices
        for (uint32_t l = 0; l < sd.n_layer; l++) {
            const auto & ld = sd.layers[l];

            // Write type and row size (same as original)
            write(&ld.k_type, sizeof(ld.k_type));
            write(&ld.k_size_row, sizeof(ld.k_size_row));

            const int n_embd_k = ld.n_embd_k_gqa();

            // Write K rows: use merged keys (C_k) if available, otherwise original K at selected_indices
            const uint64_t k_row_bytes = parsed_kv_state::row_bytes_for(ld.k_type, n_embd_k);
            std::vector<uint8_t> k_buf(k_row_bytes);
            const bool has_merged_k = ck_all && l < ck_all->size() && !(*ck_all)[l].empty();
            for (int j = 0; j < t; j++) {
                std::vector<float> k_row_buf;
                const float * k_row;
                if (has_merged_k) {
                    // Build full K row from per-head merged keys
                    k_row_buf.resize(n_embd_k);
                    for (int h = 0; h < n_head_kv; h++) {
                        memcpy(k_row_buf.data() + h * d_k,
                               (*ck_all)[l][h].data() + j * d_k,
                               d_k * sizeof(float));
                    }
                    k_row = k_row_buf.data();
                } else {
                    int orig_idx = selected_indices[j];
                    k_row = ld.K.data() + orig_idx * n_embd_k;
                }
                convert_from_f32(k_row, ld.k_type, k_buf.data(), n_embd_k);
                write(k_buf.data(), k_row_bytes);
            }
        }

        // Write V data per layer — C_v (refitted values) at selected positions
        if (!sd.v_trans) {
            for (uint32_t l = 0; l < sd.n_layer; l++) {
                const auto & ld = sd.layers[l];

                write(&ld.v_type, sizeof(ld.v_type));
                // For quantized types, row size changes because t may differ from original cell_count
                const int n_embd_v = ld.n_embd_v_gqa_computed();
                uint64_t v_row_bytes = parsed_kv_state::row_bytes_for(ld.v_type, n_embd_v);
                write(&v_row_bytes, sizeof(v_row_bytes));

                std::vector<uint8_t> v_buf(v_row_bytes);

                // Build full V rows from per-head C_v
                for (int j = 0; j < t; j++) {
                    std::vector<float> v_row(n_embd_v);
                    for (int h = 0; h < n_head_kv; h++) {
                        const float * cv = cv_all[l][h].data() + j * d_v;
                        memcpy(v_row.data() + h * d_v, cv, d_v * sizeof(float));
                    }
                    convert_from_f32(v_row.data(), ld.v_type, v_buf.data(), n_embd_v);
                    write(v_buf.data(), v_row_bytes);
                }
            }
        } else {
            // Transposed V: write as [n_embd_v_gqa][t] per layer
            for (uint32_t l = 0; l < sd.n_layer; l++) {
                const auto & ld = sd.layers[l];

                write(&ld.v_type, sizeof(ld.v_type));
                write(&ld.v_size_el, sizeof(ld.v_size_el));
                uint32_t n_embd_v = (uint32_t)(n_head_kv * d_v);
                write(&n_embd_v, sizeof(n_embd_v));

                if (parsed_kv_state::is_quantized(ld.v_type)) {
                    // For block-quantized transposed V: each dimension stores t values
                    // packed into blocks. Collect values per dimension, then quantize.
                    const uint64_t bytes_per_dim = parsed_kv_state::row_bytes_for(ld.v_type, t);
                    std::vector<float> dim_vals(t);
                    std::vector<uint8_t> dim_buf(bytes_per_dim);
                    for (uint32_t d = 0; d < n_embd_v; d++) {
                        int h = d / d_v;
                        int di = d % d_v;
                        for (int j = 0; j < t; j++) {
                            dim_vals[j] = cv_all[l][h][j * d_v + di];
                        }
                        convert_from_f32(dim_vals.data(), ld.v_type, dim_buf.data(), t);
                        write(dim_buf.data(), bytes_per_dim);
                    }
                } else {
                    // For F32/F16: write element-by-element per dimension
                    for (uint32_t d = 0; d < n_embd_v; d++) {
                        int h = d / d_v;
                        int di = d % d_v;
                        for (int j = 0; j < t; j++) {
                            float val = cv_all[l][h][j * d_v + di];
                            if (ld.v_type == KV_COMPACT_GGML_TYPE_F32) {
                                write(&val, sizeof(float));
                            } else {
                                uint16_t f16 = f32_to_f16(val);
                                write(&f16, sizeof(uint16_t));
                            }
                        }
                    }
                }
            }
        }
    }

    // Append trailing data (e.g., recurrent section for hybrid SSM+MoE models)
    if (!state.trailing_data.empty()) {
        write(state.trailing_data.data(), state.trailing_data.size());
    }

    return out;
}
