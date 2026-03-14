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
#define KV_COMPACT_GGML_TYPE_Q8_0  8
#else
#define KV_COMPACT_GGML_TYPE_F32   GGML_TYPE_F32
#define KV_COMPACT_GGML_TYPE_F16   GGML_TYPE_F16
#define KV_COMPACT_GGML_TYPE_Q4_0  GGML_TYPE_Q4_0
#define KV_COMPACT_GGML_TYPE_Q4_1  GGML_TYPE_Q4_1
#define KV_COMPACT_GGML_TYPE_Q8_0  GGML_TYPE_Q8_0
#endif

// Quantization block size (elements per block)
static constexpr int KV_COMPACT_QK = 32;

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
                const int n_floats_per_row = n_elements_per_row(ld.k_type, ld.k_size_row);
                ld.K.resize((size_t)sd.cell_count * n_floats_per_row);
                if (is_quantized(ld.k_type)) {
                    // Dequantize row by row (each row is k_size_row bytes)
                    for (uint32_t c = 0; c < sd.cell_count; c++) {
                        convert_to_f32(ptr + c * ld.k_size_row, ld.k_type,
                                       ld.K.data() + c * n_floats_per_row, n_floats_per_row);
                    }
                } else {
                    convert_to_f32(ptr, ld.k_type, ld.K.data(), sd.cell_count * n_floats_per_row);
                }

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

                    const int n_floats_per_row = n_elements_per_row(ld.v_type, ld.v_size_row);
                    ld.V.resize((size_t)sd.cell_count * n_floats_per_row);
                    if (is_quantized(ld.v_type)) {
                        for (uint32_t c = 0; c < sd.cell_count; c++) {
                            convert_to_f32(ptr + c * ld.v_size_row, ld.v_type,
                                           ld.V.data() + c * n_floats_per_row, n_floats_per_row);
                        }
                    } else {
                        convert_to_f32(ptr, ld.v_type, ld.V.data(), sd.cell_count * n_floats_per_row);
                    }

                    ptr += v_data_size;
                }
            } else {
                // Transposed V
                for (uint32_t l = 0; l < sd.n_layer; l++) {
                    auto & ld = sd.layers[l];

                    if (!read_val(ptr, end, ld.v_type)) return false;
                    if (!read_val(ptr, end, ld.v_size_el)) return false;
                    if (!read_val(ptr, end, ld.n_embd_v_gqa)) return false;

                    const size_t v_data_size = (size_t)ld.n_embd_v_gqa * sd.cell_count * ld.v_size_el;
                    if (ptr + v_data_size > end) return false;

                    // Transpose from [embd][token] to [token][embd]
                    ld.V.resize((size_t)sd.cell_count * ld.n_embd_v_gqa);
                    transpose_v_to_f32(ptr, ld.v_type, ld.V.data(),
                                       sd.cell_count, ld.n_embd_v_gqa);

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

    // ---- Layer type detection for hybrid architectures ----

    // Check if a layer has valid KV cache data suitable for compaction.
    // Returns false for layers that:
    //   - Have zero-sized K or V data
    //   - Have K/V dimensions that don't divide evenly by n_head_kv
    //   - Have mismatched K and V cell counts
    //
    // For hybrid models (e.g., Qwen 3.5 with DeltaNet + attention), non-attention
    // layers may have no KV data or have differently-structured state.
    bool is_compactable_layer(int stream, int layer, int n_head_kv) const {
        if (stream < 0 || stream >= (int)n_stream) return false;
        const auto & sd = streams[stream];
        if (layer < 0 || layer >= (int)sd.n_layer) return false;
        const auto & ld = sd.layers[layer];

        // Must have K and V data
        if (ld.K.empty() || ld.V.empty()) return false;

        // K/V dimensions must divide evenly by n_head_kv
        int n_embd_k = ld.n_embd_k_gqa();
        int n_embd_v = ld.n_embd_v_gqa_computed();
        if (n_head_kv <= 0) return false;
        if (n_embd_k % n_head_kv != 0) return false;
        if (n_embd_v % n_head_kv != 0) return false;

        // Head dimensions must be > 0
        if (n_embd_k / n_head_kv == 0) return false;
        if (n_embd_v / n_head_kv == 0) return false;

        return true;
    }

    // Detect which layers are compactable and return a list of their indices.
    // Useful for auto-detecting attention layers in hybrid architectures.
    std::vector<int> get_compactable_layers(int stream, int n_head_kv) const {
        std::vector<int> result;
        if (stream < 0 || stream >= (int)n_stream) return result;
        const auto & sd = streams[stream];
        for (int l = 0; l < (int)sd.n_layer; l++) {
            if (is_compactable_layer(stream, l, n_head_kv)) {
                result.push_back(l);
            }
        }
        return result;
    }

    // Check if a model appears to be hybrid (not all layers have the same K/V geometry).
    // Returns true if any layer has different K dimensions than layer 0.
    bool is_hybrid_model(int stream = 0) const {
        if (stream < 0 || stream >= (int)n_stream) return false;
        const auto & sd = streams[stream];
        if (sd.n_layer <= 1) return false;

        uint64_t ref_k_size = sd.layers[0].k_size_row;
        for (uint32_t l = 1; l < sd.n_layer; l++) {
            if (sd.layers[l].k_size_row != ref_k_size) return true;
        }
        return false;
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

    // ---- Internal helpers ----

    template<typename T>
    static bool read_val(const uint8_t *& ptr, const uint8_t * end, T & val) {
        if (ptr + sizeof(T) > end) return false;
        memcpy(&val, ptr, sizeof(T));
        ptr += sizeof(T);
        return true;
    }

    // ---- Type utilities and (de)quantization (public for testing) ----

    // Bytes per element (for non-block types) or bytes per block (for block types)
    static int type_size(int32_t type) {
        if (type == KV_COMPACT_GGML_TYPE_F32)  return 4;
        if (type == KV_COMPACT_GGML_TYPE_F16)  return 2;
        if (type == KV_COMPACT_GGML_TYPE_Q8_0) return 2 + KV_COMPACT_QK;  // 34 bytes
        if (type == KV_COMPACT_GGML_TYPE_Q4_0) return 2 + KV_COMPACT_QK/2; // 18 bytes
        if (type == KV_COMPACT_GGML_TYPE_Q4_1) return 4 + KV_COMPACT_QK/2; // 20 bytes
        return 4; // fallback
    }

    // Is this a block-quantized type?
    static bool is_quantized(int32_t type) {
        return type == KV_COMPACT_GGML_TYPE_Q8_0 ||
               type == KV_COMPACT_GGML_TYPE_Q4_0 ||
               type == KV_COMPACT_GGML_TYPE_Q4_1;
    }

    // Number of float elements per row given type and row byte size
    static int n_elements_per_row(int32_t type, uint64_t row_bytes) {
        if (type == KV_COMPACT_GGML_TYPE_F32)  return (int)(row_bytes / 4);
        if (type == KV_COMPACT_GGML_TYPE_F16)  return (int)(row_bytes / 2);
        if (is_quantized(type)) {
            int block_bytes = type_size(type);
            int n_blocks = (int)(row_bytes / block_bytes);
            return n_blocks * KV_COMPACT_QK;
        }
        return (int)(row_bytes / 4); // fallback
    }

    // Dequantize a Q8_0 block (34 bytes → 32 floats)
    static void dequant_q8_0_block(const uint8_t * src, float * dst) {
        uint16_t d_f16;
        memcpy(&d_f16, src, 2);
        float d = f16_to_f32(d_f16);
        const int8_t * qs = (const int8_t *)(src + 2);
        for (int i = 0; i < KV_COMPACT_QK; i++) {
            dst[i] = d * (float)qs[i];
        }
    }

    // Dequantize a Q4_0 block (18 bytes → 32 floats)
    static void dequant_q4_0_block(const uint8_t * src, float * dst) {
        uint16_t d_f16;
        memcpy(&d_f16, src, 2);
        float d = f16_to_f32(d_f16);
        const uint8_t * qs = src + 2;
        for (int i = 0; i < KV_COMPACT_QK / 2; i++) {
            dst[2*i + 0] = d * ((float)(qs[i] & 0xF) - 8.0f);
            dst[2*i + 1] = d * ((float)(qs[i] >> 4)  - 8.0f);
        }
    }

    // Dequantize a Q4_1 block (20 bytes → 32 floats)
    static void dequant_q4_1_block(const uint8_t * src, float * dst) {
        uint16_t d_f16, m_f16;
        memcpy(&d_f16, src, 2);
        memcpy(&m_f16, src + 2, 2);
        float d = f16_to_f32(d_f16);
        float m = f16_to_f32(m_f16);
        const uint8_t * qs = src + 4;
        for (int i = 0; i < KV_COMPACT_QK / 2; i++) {
            dst[2*i + 0] = d * (float)(qs[i] & 0xF) + m;
            dst[2*i + 1] = d * (float)(qs[i] >> 4)  + m;
        }
    }

    // Convert raw data to float32 (supports F32, F16, Q8_0, Q4_0, Q4_1)
    static void convert_to_f32(const uint8_t * src, int32_t type, float * dst, size_t n) {
        if (type == KV_COMPACT_GGML_TYPE_F32) {
            memcpy(dst, src, n * sizeof(float));
        } else if (type == KV_COMPACT_GGML_TYPE_F16) {
            const uint16_t * f16 = (const uint16_t *) src;
            for (size_t i = 0; i < n; i++) {
                dst[i] = f16_to_f32(f16[i]);
            }
        } else if (type == KV_COMPACT_GGML_TYPE_Q8_0) {
            int block_bytes = type_size(type);
            size_t n_blocks = n / KV_COMPACT_QK;
            for (size_t b = 0; b < n_blocks; b++) {
                dequant_q8_0_block(src + b * block_bytes, dst + b * KV_COMPACT_QK);
            }
        } else if (type == KV_COMPACT_GGML_TYPE_Q4_0) {
            int block_bytes = type_size(type);
            size_t n_blocks = n / KV_COMPACT_QK;
            for (size_t b = 0; b < n_blocks; b++) {
                dequant_q4_0_block(src + b * block_bytes, dst + b * KV_COMPACT_QK);
            }
        } else if (type == KV_COMPACT_GGML_TYPE_Q4_1) {
            int block_bytes = type_size(type);
            size_t n_blocks = n / KV_COMPACT_QK;
            for (size_t b = 0; b < n_blocks; b++) {
                dequant_q4_1_block(src + b * block_bytes, dst + b * KV_COMPACT_QK);
            }
        } else {
            memset(dst, 0, n * sizeof(float));
        }
    }

    // Transpose V from [embd][cell] to [cell][embd] and convert to F32
    // For quantized types, each column of embd is stored as contiguous blocks
    static void transpose_v_to_f32(const uint8_t * src, int32_t type, float * dst,
                                   uint32_t cell_count, uint32_t n_embd) {
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
            // Transposed quantized: each embedding dimension d has cell_count values
            // stored as contiguous quantized blocks of size v_size_el * cell_count
            // Dequantize each column, then scatter to row-major
            int el_bytes = type_size(type) / KV_COMPACT_QK;
            // Actually for transposed V, v_size_el gives element size, not block size
            // The data layout is [n_embd][cell_count] elements in quantized format
            // Each contiguous run of cell_count elements is quantized in blocks of QK
            size_t col_bytes = (size_t)cell_count * type_size(type) / KV_COMPACT_QK;
            // Simpler: dequantize all, then transpose
            std::vector<float> tmp(n_embd * cell_count);
            convert_to_f32(src, type, tmp.data(), n_embd * cell_count);
            // tmp is now [n_embd][cell_count] in row-major, transpose to [cell_count][n_embd]
            for (uint32_t d = 0; d < n_embd; d++) {
                for (uint32_t c = 0; c < cell_count; c++) {
                    dst[c * n_embd + d] = tmp[d * cell_count + c];
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

// Quantize a block of 32 floats to Q8_0 format (34 bytes)
static void quant_q8_0_block(const float * src, uint8_t * dst) {
    // Find max absolute value
    float amax = 0.0f;
    for (int i = 0; i < KV_COMPACT_QK; i++) {
        float a = fabsf(src[i]);
        if (a > amax) amax = a;
    }
    float d = amax / 127.0f;
    uint16_t d_f16 = f32_to_f16(d);
    memcpy(dst, &d_f16, 2);

    float id = (d != 0.0f) ? 127.0f / amax : 0.0f;
    int8_t * qs = (int8_t *)(dst + 2);
    for (int i = 0; i < KV_COMPACT_QK; i++) {
        float v = src[i] * id;
        qs[i] = (int8_t)roundf(std::max(-128.0f, std::min(127.0f, v)));
    }
}

// Quantize a block of 32 floats to Q4_0 format (18 bytes)
static void quant_q4_0_block(const float * src, uint8_t * dst) {
    float amax = 0.0f;
    for (int i = 0; i < KV_COMPACT_QK; i++) {
        float a = fabsf(src[i]);
        if (a > amax) amax = a;
    }
    float d = amax / 7.0f;
    uint16_t d_f16 = f32_to_f16(d);
    memcpy(dst, &d_f16, 2);

    float id = (d != 0.0f) ? 7.0f / amax : 0.0f;
    uint8_t * qs = dst + 2;
    for (int i = 0; i < KV_COMPACT_QK / 2; i++) {
        float v0 = src[2*i + 0] * id + 8.0f;
        float v1 = src[2*i + 1] * id + 8.0f;
        uint8_t q0 = (uint8_t)std::max(0.0f, std::min(15.0f, roundf(v0)));
        uint8_t q1 = (uint8_t)std::max(0.0f, std::min(15.0f, roundf(v1)));
        qs[i] = q0 | (q1 << 4);
    }
}

// Quantize a block of 32 floats to Q4_1 format (20 bytes)
static void quant_q4_1_block(const float * src, uint8_t * dst) {
    // Find min and max
    float fmin = src[0], fmax = src[0];
    for (int i = 1; i < KV_COMPACT_QK; i++) {
        if (src[i] < fmin) fmin = src[i];
        if (src[i] > fmax) fmax = src[i];
    }
    float d = (fmax - fmin) / 15.0f;
    uint16_t d_f16 = f32_to_f16(d);
    uint16_t m_f16 = f32_to_f16(fmin);
    memcpy(dst, &d_f16, 2);
    memcpy(dst + 2, &m_f16, 2);

    float id = (d != 0.0f) ? 15.0f / (fmax - fmin) : 0.0f;
    uint8_t * qs = dst + 4;
    for (int i = 0; i < KV_COMPACT_QK / 2; i++) {
        float v0 = (src[2*i + 0] - fmin) * id;
        float v1 = (src[2*i + 1] - fmin) * id;
        uint8_t q0 = (uint8_t)std::max(0.0f, std::min(15.0f, roundf(v0)));
        uint8_t q1 = (uint8_t)std::max(0.0f, std::min(15.0f, roundf(v1)));
        qs[i] = q0 | (q1 << 4);
    }
}

// Convert a row of floats to the target type, writing to dst
// Returns number of bytes written
static size_t convert_from_f32(const float * src, int32_t type, uint8_t * dst, int n_elements) {
    if (type == KV_COMPACT_GGML_TYPE_F32) {
        memcpy(dst, src, n_elements * sizeof(float));
        return n_elements * sizeof(float);
    } else if (type == KV_COMPACT_GGML_TYPE_F16) {
        uint16_t * f16 = (uint16_t *)dst;
        for (int i = 0; i < n_elements; i++) {
            f16[i] = f32_to_f16(src[i]);
        }
        return n_elements * sizeof(uint16_t);
    } else if (type == KV_COMPACT_GGML_TYPE_Q8_0) {
        int n_blocks = n_elements / KV_COMPACT_QK;
        int block_bytes = 2 + KV_COMPACT_QK;  // 34
        for (int b = 0; b < n_blocks; b++) {
            quant_q8_0_block(src + b * KV_COMPACT_QK, dst + b * block_bytes);
        }
        return n_blocks * block_bytes;
    } else if (type == KV_COMPACT_GGML_TYPE_Q4_0) {
        int n_blocks = n_elements / KV_COMPACT_QK;
        int block_bytes = 2 + KV_COMPACT_QK / 2;  // 18
        for (int b = 0; b < n_blocks; b++) {
            quant_q4_0_block(src + b * KV_COMPACT_QK, dst + b * block_bytes);
        }
        return n_blocks * block_bytes;
    } else if (type == KV_COMPACT_GGML_TYPE_Q4_1) {
        int n_blocks = n_elements / KV_COMPACT_QK;
        int block_bytes = 4 + KV_COMPACT_QK / 2;  // 20
        for (int b = 0; b < n_blocks; b++) {
            quant_q4_1_block(src + b * KV_COMPACT_QK, dst + b * block_bytes);
        }
        return n_blocks * block_bytes;
    }
    return 0;
}

// Build a compacted state buffer from original parsed state + compaction results
//
// For each layer:
//   - K: copy original K rows for selected indices, optionally modified with beta
//   - V: write C_v (refitted values) for each head at selected positions
//
// selected_indices: [t] shared across all heads within a layer
// cv_all:  [layer][head] = vector<float> of [t * d_v] (refitted values)
// beta_all: [layer][head] = vector<float> of [t] (attention biases)
//           If empty, no beta modification is applied to K.
// beta_dirs: [layer][head] = vector<float> of [d_k] (beta encoding directions)
//           Required when beta_all is non-empty.
//
// Returns the new state buffer ready for llama_state_seq_set_data()
static std::vector<uint8_t> build_compacted_state(
        const parsed_kv_state & state,
        const std::vector<int> & selected_indices,
        const std::vector<std::vector<std::vector<float>>> & cv_all,
        int n_head_kv, int d_k, int d_v,
        uint32_t n_pos_per_embd = 1,
        const std::vector<std::vector<std::vector<float>>> & beta_all = {},
        const std::vector<std::vector<std::vector<float>>> & beta_dirs = {}) {

    const bool has_beta = !beta_all.empty() && !beta_dirs.empty();

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

            // Write selected K rows, optionally with beta folded in
            std::vector<uint8_t> row_buf(ld.k_size_row);
            for (int j = 0; j < t; j++) {
                int orig_idx = selected_indices[j];
                std::vector<float> k_row(ld.K.data() + orig_idx * n_embd_k,
                                         ld.K.data() + (orig_idx + 1) * n_embd_k);

                if (has_beta && l < (uint32_t)beta_all.size()) {
                    const float scale = sqrtf((float) d_k);
                    for (int h = 0; h < n_head_kv; h++) {
                        const float b_scaled = beta_all[l][h][j] * scale;
                        const float * dir = beta_dirs[l][h].data();
                        for (int d = 0; d < d_k; d++) {
                            k_row[h * d_k + d] += b_scaled * dir[d];
                        }
                    }
                }

                size_t bytes = convert_from_f32(k_row.data(), ld.k_type,
                                                row_buf.data(), n_embd_k);
                write(row_buf.data(), bytes);
            }
        }

        // Write V data per layer — C_v (refitted values) at selected positions
        if (!sd.v_trans) {
            for (uint32_t l = 0; l < sd.n_layer; l++) {
                const auto & ld = sd.layers[l];

                write(&ld.v_type, sizeof(ld.v_type));
                write(&ld.v_size_row, sizeof(ld.v_size_row));

                const int n_embd_v = ld.n_embd_v_gqa_computed();

                // Build full V rows from per-head C_v and requantize
                std::vector<uint8_t> v_row_buf(ld.v_size_row);
                for (int j = 0; j < t; j++) {
                    std::vector<float> v_row(n_embd_v);
                    for (int h = 0; h < n_head_kv; h++) {
                        const float * cv = cv_all[l][h].data() + j * d_v;
                        memcpy(v_row.data() + h * d_v, cv, d_v * sizeof(float));
                    }

                    size_t bytes = convert_from_f32(v_row.data(), ld.v_type,
                                                    v_row_buf.data(), n_embd_v);
                    write(v_row_buf.data(), bytes);
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

                // For each embedding dimension d, write t values (transposed)
                // Collect t values per embedding dim, then convert
                std::vector<float> col_vals(t);
                for (uint32_t d = 0; d < n_embd_v; d++) {
                    int h = d / d_v;
                    int di = d % d_v;
                    for (int j = 0; j < t; j++) {
                        col_vals[j] = cv_all[l][h][j * d_v + di];
                    }
                    // For transposed, each "column" is t elements
                    // Use per-element writing for non-block types,
                    // or block writing for quantized types
                    if (parsed_kv_state::is_quantized(ld.v_type)) {
                        // Quantized transposed: write t values as blocks
                        std::vector<uint8_t> buf(t * 4); // generous
                        size_t bytes = convert_from_f32(col_vals.data(), ld.v_type,
                                                        buf.data(), t);
                        write(buf.data(), bytes);
                    } else {
                        for (int j = 0; j < t; j++) {
                            if (ld.v_type == KV_COMPACT_GGML_TYPE_F32) {
                                write(&col_vals[j], sizeof(float));
                            } else if (ld.v_type == KV_COMPACT_GGML_TYPE_F16) {
                                uint16_t f16 = f32_to_f16(col_vals[j]);
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
