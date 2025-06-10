#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <emmintrin.h>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

// For SIMD intrinsics (SSSE3)
#include <immintrin.h>
#include <xmmintrin.h>

// --- Define core parameters ---
const int D = 128; // vector dimension
const int M = 8;   // Number of subvectors (Number of sub-vectors, 128 / 16 = 8)
const int K_STAR = 256;   // Number of centroids per sub-quantizer (Number of
                          // centroids per sub-quantizer, 16 * 16 = 256)
const int D_STAR = D / M; // Dimension of each sub-vector

const int NUM_DB_VECTORS = 1'000'000; // Number of vectors in the database
const int GROUPING_COMPONENTS = 4; // Number of subvectors for vector grouping
const float KEEP_PERCENT = 0.01f;  // Scan ratio for finding qmax

// --- Data structure definitions ---
using vec = std::vector<float>;
using vec_u8 = std::vector<uint8_t>;
using centroids_t = std::vector<std::vector<vec>>; // Centroids: [m][k*][d*]
using db_pqcodes_t = std::vector<vec_u8>;          // Database PQ codes: [N][m]

// --- Utility functions ---

// Calculate L2 squared distance between two sub-vectors
float l2_squared(const vec &v1, const vec &v2) {
  float dist = 0.0f;
  for (size_t i = 0; i < v1.size(); ++i) {
    float diff = v1[i] - v2[i];
    dist += diff * diff;
  }
  return dist;
}

// Simple timer
class Timer {
public:
  void start() { m_start = std::chrono::high_resolution_clock::now(); }
  double stop() {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - m_start).count();
  }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
};

// --- PQ core functions ---

/**
 * @brief Generate centroids and simulate the optimized assignment strategy
 * mentioned in the paper. To make centroids with similar indices spatially
 * close, we generate a cluster core for each portion (16 centroids) and
 * then generate 16 centroids around this cluster core.
 */
centroids_t generate_centroids_optimized() {
  std::cout << "Generating optimized centroids..." << std::endl;
  centroids_t centroids(M, std::vector<vec>(K_STAR, vec(D_STAR)));
  std::random_device rd;
  std::mt19937 gen(rd());
  // Distribution range for cluster cores
  std::uniform_real_distribution<> dis_core(-5.0, 5.0);
  // Noise around cluster cores
  std::normal_distribution<> dis_noise(0.0, 0.5);

  for (int m = 0; m < M; ++m) {
    // Each sub-quantizer has 16 clusters
    for (int portion_idx = 0; portion_idx < (K_STAR / 16); ++portion_idx) {
      // Generate a cluster core for this portion
      vec cluster_core(D_STAR);
      for (int d = 0; d < D_STAR; ++d) {
        cluster_core[d] = dis_core(gen);
      }

      // 2. Generate 16 centroids around the cluster core
      int start_k_in_portion = portion_idx * 16;
      for (int i = 0; i < 16; ++i) {
        int current_k = start_k_in_portion + i;
        for (int d = 0; d < D_STAR; ++d) {
          centroids[m][current_k][d] = cluster_core[d] + dis_noise(gen);
        }
      }
    }
  }
  return centroids;
}

// Encode a single 128-dim vector into a PQ Code
vec_u8 encode_vector(const vec &v, const centroids_t &centroids) {
  vec_u8 pqcode(M);
  for (int m_idx = 0; m_idx < M; ++m_idx) {
    vec sub_vec(v.begin() + m_idx * D_STAR, v.begin() + (m_idx + 1) * D_STAR);

    float min_dist = std::numeric_limits<float>::max();
    uint8_t best_k = 0;
    for (int k = 0; k < K_STAR; ++k) {
      float dist = l2_squared(sub_vec, centroids[m_idx][k]);
      if (dist < min_dist) {
        min_dist = dist;
        best_k = static_cast<uint8_t>(k);
      }
    }
    pqcode[m_idx] = best_k;
  }
  return pqcode;
}

// Compute distance tables for a given query
std::vector<std::vector<float>>
compute_distance_tables(const vec &query, const centroids_t &centroids) {
  std::vector<std::vector<float>> dist_tables(M, std::vector<float>(K_STAR));
  for (int m_idx = 0; m_idx < M; ++m_idx) {
    vec sub_query(query.begin() + m_idx * D_STAR,
                  query.begin() + (m_idx + 1) * D_STAR);
    for (int k = 0; k < K_STAR; ++k) {
      dist_tables[m_idx][k] = l2_squared(sub_query, centroids[m_idx][k]);
    }
  }
  return dist_tables;
}

// --- Original PQ Scan method ---
inline float
pq_distance_original(const vec_u8 &pqcode,
                     const std::vector<std::vector<float>> &dist_tables) {
  float total_dist = 0.0f;
  for (int m = 0; m < M; ++m) {
    total_dist += dist_tables[m][pqcode[m]];
  }
  return total_dist;
}

std::priority_queue<std::pair<float, int>>
run_pq_scan_original(const db_pqcodes_t &db,
                     const std::vector<std::vector<float>> &dist_tables,
                     int topK = 100) {
  std::cout << "\n--- PQ Scan (Running Original PQ Scan) ---" << std::endl;
  std::priority_queue<std::pair<float, int>> top_results;
  Timer timer;
  timer.start();

  for (int i = 0; i < db.size(); ++i) {
    const auto &pqcode = db[i];
    float dist = pq_distance_original(pqcode, dist_tables);
    if (top_results.size() < topK) {
      top_results.emplace(dist, i);
    } else if (dist < top_results.top().first) {
      top_results.pop();
      top_results.emplace(dist, i);
    }
  }

  double elapsed = timer.stop();
  std::cout << "Original PQ Scan finished in: " << elapsed << " ms"
            << std::endl;
  return top_results;
}

// --- PQ Fast Scan method ---

/**
 * @brief Quantize float distance to 8-bit unsigned integer.
 * (According to paper Figure 12: q_min -> 0, q_max -> 126, >q_max -> 127)
 */
uint8_t quantize_dist(float dist, float q_min, float q_max,
                      float inv_range_0_126) {
  if (dist <= q_min)
    return 0;
  // (Paper mentions values >= q_max are
  // quantized to 127)
  if (dist >= q_max)
    return 127;
  float scaled = (dist - q_min) * inv_range_0_126;
  return static_cast<uint8_t>(std::min(std::max(0.0f, scaled), 126.0f));
}

// --- SIMD processing functions for PQ Fast Scan ---

// Processes 1 vector
inline __m128i process_one(size_t vec_idx_base,
                           const std::vector<const vec_u8 *> &group_vectors,
                           const __m128i small_tables[M],
                           int8_t lower_bound_signed) {
  uint8_t shuffle_indices_buffer[16];
  __m128i accumulated_sums = _mm_setzero_si128();

  for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) & 0x0F;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  for (int m = GROUPING_COMPONENTS; m < M; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) >> 4;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  __m128i lower_bound_vec = _mm_set1_epi8(lower_bound_signed);
  __m128i mask1 = _mm_cmpgt_epi8(lower_bound_vec, accumulated_sums);
  __m128i mask2 = _mm_cmpeq_epi8(lower_bound_vec, accumulated_sums);
  return _mm_or_si128(mask1, mask2);
}

// Processes 2 vectors
inline __m128i process_two(size_t vec_idx_base,
                           const std::vector<const vec_u8 *> &group_vectors,
                           const __m128i small_tables[M],
                           int8_t lower_bound_signed) {
  uint8_t shuffle_indices_buffer[16];
  __m128i accumulated_sums = _mm_setzero_si128();

  for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) & 0x0F;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) & 0x0F;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  for (int m = GROUPING_COMPONENTS; m < M; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) >> 4;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) >> 4;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  __m128i lower_bound_vec = _mm_set1_epi8(lower_bound_signed);
  __m128i mask1 = _mm_cmpgt_epi8(lower_bound_vec, accumulated_sums);
  __m128i mask2 = _mm_cmpeq_epi8(lower_bound_vec, accumulated_sums);
  return _mm_or_si128(mask1, mask2);
}

// Processes 3 vectors
inline __m128i process_three(size_t vec_idx_base,
                             const std::vector<const vec_u8 *> &group_vectors,
                             const __m128i small_tables[M],
                             int8_t lower_bound_signed) {
  uint8_t shuffle_indices_buffer[16];
  __m128i accumulated_sums = _mm_setzero_si128();

  for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) & 0x0F;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) & 0x0F;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) & 0x0F;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  for (int m = GROUPING_COMPONENTS; m < M; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) >> 4;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) >> 4;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) >> 4;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  __m128i lower_bound_vec = _mm_set1_epi8(lower_bound_signed);
  __m128i mask1 = _mm_cmpgt_epi8(lower_bound_vec, accumulated_sums);
  __m128i mask2 = _mm_cmpeq_epi8(lower_bound_vec, accumulated_sums);
  return _mm_or_si128(mask1, mask2);
}

// Processes 4 vectors
inline __m128i process_four(size_t vec_idx_base,
                            const std::vector<const vec_u8 *> &group_vectors,
                            const __m128i small_tables[M],
                            int8_t lower_bound_signed) {
  uint8_t shuffle_indices_buffer[16];
  __m128i accumulated_sums = _mm_setzero_si128();

  for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) & 0x0F;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) & 0x0F;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) & 0x0F;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) & 0x0F;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  for (int m = GROUPING_COMPONENTS; m < M; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) >> 4;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) >> 4;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) >> 4;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) >> 4;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  __m128i lower_bound_vec = _mm_set1_epi8(lower_bound_signed);
  __m128i mask1 = _mm_cmpgt_epi8(lower_bound_vec, accumulated_sums);
  __m128i mask2 = _mm_cmpeq_epi8(lower_bound_vec, accumulated_sums);
  return _mm_or_si128(mask1, mask2);
}

// Processes 5 vectors
inline __m128i process_five(size_t vec_idx_base,
                            const std::vector<const vec_u8 *> &group_vectors,
                            const __m128i small_tables[M],
                            int8_t lower_bound_signed) {
  uint8_t shuffle_indices_buffer[16];
  __m128i accumulated_sums = _mm_setzero_si128();

  for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) & 0x0F;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) & 0x0F;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) & 0x0F;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) & 0x0F;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) & 0x0F;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  for (int m = GROUPING_COMPONENTS; m < M; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) >> 4;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) >> 4;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) >> 4;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) >> 4;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) >> 4;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  __m128i lower_bound_vec = _mm_set1_epi8(lower_bound_signed);
  __m128i mask1 = _mm_cmpgt_epi8(lower_bound_vec, accumulated_sums);
  __m128i mask2 = _mm_cmpeq_epi8(lower_bound_vec, accumulated_sums);
  return _mm_or_si128(mask1, mask2);
}

// Processes 6 vectors
inline __m128i process_six(size_t vec_idx_base,
                           const std::vector<const vec_u8 *> &group_vectors,
                           const __m128i small_tables[M],
                           int8_t lower_bound_signed) {
  uint8_t shuffle_indices_buffer[16];
  __m128i accumulated_sums = _mm_setzero_si128();

  for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) & 0x0F;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) & 0x0F;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) & 0x0F;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) & 0x0F;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) & 0x0F;
    shuffle_indices_buffer[5] = group_vectors[vec_idx_base + 5]->at(m) & 0x0F;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  for (int m = GROUPING_COMPONENTS; m < M; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) >> 4;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) >> 4;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) >> 4;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) >> 4;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) >> 4;
    shuffle_indices_buffer[5] = group_vectors[vec_idx_base + 5]->at(m) >> 4;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  __m128i lower_bound_vec = _mm_set1_epi8(lower_bound_signed);
  __m128i mask1 = _mm_cmpgt_epi8(lower_bound_vec, accumulated_sums);
  __m128i mask2 = _mm_cmpeq_epi8(lower_bound_vec, accumulated_sums);
  return _mm_or_si128(mask1, mask2);
}

// Processes 7 vectors
inline __m128i process_seven(size_t vec_idx_base,
                             const std::vector<const vec_u8 *> &group_vectors,
                             const __m128i small_tables[M],
                             int8_t lower_bound_signed) {
  uint8_t shuffle_indices_buffer[16];
  __m128i accumulated_sums = _mm_setzero_si128();

  for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) & 0x0F;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) & 0x0F;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) & 0x0F;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) & 0x0F;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) & 0x0F;
    shuffle_indices_buffer[5] = group_vectors[vec_idx_base + 5]->at(m) & 0x0F;
    shuffle_indices_buffer[6] = group_vectors[vec_idx_base + 6]->at(m) & 0x0F;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  for (int m = GROUPING_COMPONENTS; m < M; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) >> 4;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) >> 4;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) >> 4;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) >> 4;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) >> 4;
    shuffle_indices_buffer[5] = group_vectors[vec_idx_base + 5]->at(m) >> 4;
    shuffle_indices_buffer[6] = group_vectors[vec_idx_base + 6]->at(m) >> 4;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  __m128i lower_bound_vec = _mm_set1_epi8(lower_bound_signed);
  __m128i mask1 = _mm_cmpgt_epi8(lower_bound_vec, accumulated_sums);
  __m128i mask2 = _mm_cmpeq_epi8(lower_bound_vec, accumulated_sums);
  return _mm_or_si128(mask1, mask2);
}

// Processes 8 vectors
inline __m128i process_eight(size_t vec_idx_base,
                             const std::vector<const vec_u8 *> &group_vectors,
                             const __m128i small_tables[M],
                             int8_t lower_bound_signed) {
  uint8_t shuffle_indices_buffer[16];
  __m128i accumulated_sums = _mm_setzero_si128();

  for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) & 0x0F;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) & 0x0F;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) & 0x0F;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) & 0x0F;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) & 0x0F;
    shuffle_indices_buffer[5] = group_vectors[vec_idx_base + 5]->at(m) & 0x0F;
    shuffle_indices_buffer[6] = group_vectors[vec_idx_base + 6]->at(m) & 0x0F;
    shuffle_indices_buffer[7] = group_vectors[vec_idx_base + 7]->at(m) & 0x0F;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  for (int m = GROUPING_COMPONENTS; m < M; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) >> 4;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) >> 4;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) >> 4;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) >> 4;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) >> 4;
    shuffle_indices_buffer[5] = group_vectors[vec_idx_base + 5]->at(m) >> 4;
    shuffle_indices_buffer[6] = group_vectors[vec_idx_base + 6]->at(m) >> 4;
    shuffle_indices_buffer[7] = group_vectors[vec_idx_base + 7]->at(m) >> 4;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  __m128i lower_bound_vec = _mm_set1_epi8(lower_bound_signed);
  __m128i mask1 = _mm_cmpgt_epi8(lower_bound_vec, accumulated_sums);
  __m128i mask2 = _mm_cmpeq_epi8(lower_bound_vec, accumulated_sums);
  return _mm_or_si128(mask1, mask2);
}

// Processes 9 vectors
inline __m128i process_nine(size_t vec_idx_base,
                            const std::vector<const vec_u8 *> &group_vectors,
                            const __m128i small_tables[M],
                            int8_t lower_bound_signed) {
  uint8_t shuffle_indices_buffer[16];
  __m128i accumulated_sums = _mm_setzero_si128();

  for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) & 0x0F;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) & 0x0F;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) & 0x0F;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) & 0x0F;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) & 0x0F;
    shuffle_indices_buffer[5] = group_vectors[vec_idx_base + 5]->at(m) & 0x0F;
    shuffle_indices_buffer[6] = group_vectors[vec_idx_base + 6]->at(m) & 0x0F;
    shuffle_indices_buffer[7] = group_vectors[vec_idx_base + 7]->at(m) & 0x0F;
    shuffle_indices_buffer[8] = group_vectors[vec_idx_base + 8]->at(m) & 0x0F;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  for (int m = GROUPING_COMPONENTS; m < M; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) >> 4;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) >> 4;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) >> 4;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) >> 4;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) >> 4;
    shuffle_indices_buffer[5] = group_vectors[vec_idx_base + 5]->at(m) >> 4;
    shuffle_indices_buffer[6] = group_vectors[vec_idx_base + 6]->at(m) >> 4;
    shuffle_indices_buffer[7] = group_vectors[vec_idx_base + 7]->at(m) >> 4;
    shuffle_indices_buffer[8] = group_vectors[vec_idx_base + 8]->at(m) >> 4;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  __m128i lower_bound_vec = _mm_set1_epi8(lower_bound_signed);
  __m128i mask1 = _mm_cmpgt_epi8(lower_bound_vec, accumulated_sums);
  __m128i mask2 = _mm_cmpeq_epi8(lower_bound_vec, accumulated_sums);
  return _mm_or_si128(mask1, mask2);
}

// Processes 10 vectors
inline __m128i process_ten(size_t vec_idx_base,
                           const std::vector<const vec_u8 *> &group_vectors,
                           const __m128i small_tables[M],
                           int8_t lower_bound_signed) {
  uint8_t shuffle_indices_buffer[16];
  __m128i accumulated_sums = _mm_setzero_si128();

  for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) & 0x0F;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) & 0x0F;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) & 0x0F;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) & 0x0F;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) & 0x0F;
    shuffle_indices_buffer[5] = group_vectors[vec_idx_base + 5]->at(m) & 0x0F;
    shuffle_indices_buffer[6] = group_vectors[vec_idx_base + 6]->at(m) & 0x0F;
    shuffle_indices_buffer[7] = group_vectors[vec_idx_base + 7]->at(m) & 0x0F;
    shuffle_indices_buffer[8] = group_vectors[vec_idx_base + 8]->at(m) & 0x0F;
    shuffle_indices_buffer[9] = group_vectors[vec_idx_base + 9]->at(m) & 0x0F;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  for (int m = GROUPING_COMPONENTS; m < M; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) >> 4;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) >> 4;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) >> 4;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) >> 4;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) >> 4;
    shuffle_indices_buffer[5] = group_vectors[vec_idx_base + 5]->at(m) >> 4;
    shuffle_indices_buffer[6] = group_vectors[vec_idx_base + 6]->at(m) >> 4;
    shuffle_indices_buffer[7] = group_vectors[vec_idx_base + 7]->at(m) >> 4;
    shuffle_indices_buffer[8] = group_vectors[vec_idx_base + 8]->at(m) >> 4;
    shuffle_indices_buffer[9] = group_vectors[vec_idx_base + 9]->at(m) >> 4;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  __m128i lower_bound_vec = _mm_set1_epi8(lower_bound_signed);
  __m128i mask1 = _mm_cmpgt_epi8(lower_bound_vec, accumulated_sums);
  __m128i mask2 = _mm_cmpeq_epi8(lower_bound_vec, accumulated_sums);
  return _mm_or_si128(mask1, mask2);
}

// Processes 11 vectors
inline __m128i process_eleven(size_t vec_idx_base,
                              const std::vector<const vec_u8 *> &group_vectors,
                              const __m128i small_tables[M],
                              int8_t lower_bound_signed) {
  uint8_t shuffle_indices_buffer[16];
  __m128i accumulated_sums = _mm_setzero_si128();

  for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) & 0x0F;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) & 0x0F;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) & 0x0F;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) & 0x0F;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) & 0x0F;
    shuffle_indices_buffer[5] = group_vectors[vec_idx_base + 5]->at(m) & 0x0F;
    shuffle_indices_buffer[6] = group_vectors[vec_idx_base + 6]->at(m) & 0x0F;
    shuffle_indices_buffer[7] = group_vectors[vec_idx_base + 7]->at(m) & 0x0F;
    shuffle_indices_buffer[8] = group_vectors[vec_idx_base + 8]->at(m) & 0x0F;
    shuffle_indices_buffer[9] = group_vectors[vec_idx_base + 9]->at(m) & 0x0F;
    shuffle_indices_buffer[10] = group_vectors[vec_idx_base + 10]->at(m) & 0x0F;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  for (int m = GROUPING_COMPONENTS; m < M; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) >> 4;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) >> 4;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) >> 4;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) >> 4;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) >> 4;
    shuffle_indices_buffer[5] = group_vectors[vec_idx_base + 5]->at(m) >> 4;
    shuffle_indices_buffer[6] = group_vectors[vec_idx_base + 6]->at(m) >> 4;
    shuffle_indices_buffer[7] = group_vectors[vec_idx_base + 7]->at(m) >> 4;
    shuffle_indices_buffer[8] = group_vectors[vec_idx_base + 8]->at(m) >> 4;
    shuffle_indices_buffer[9] = group_vectors[vec_idx_base + 9]->at(m) >> 4;
    shuffle_indices_buffer[10] = group_vectors[vec_idx_base + 10]->at(m) >> 4;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  __m128i lower_bound_vec = _mm_set1_epi8(lower_bound_signed);
  __m128i mask1 = _mm_cmpgt_epi8(lower_bound_vec, accumulated_sums);
  __m128i mask2 = _mm_cmpeq_epi8(lower_bound_vec, accumulated_sums);
  return _mm_or_si128(mask1, mask2);
}

// Processes 12 vectors
inline __m128i process_twelve(size_t vec_idx_base,
                              const std::vector<const vec_u8 *> &group_vectors,
                              const __m128i small_tables[M],
                              int8_t lower_bound_signed) {
  uint8_t shuffle_indices_buffer[16];
  __m128i accumulated_sums = _mm_setzero_si128();

  for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) & 0x0F;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) & 0x0F;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) & 0x0F;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) & 0x0F;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) & 0x0F;
    shuffle_indices_buffer[5] = group_vectors[vec_idx_base + 5]->at(m) & 0x0F;
    shuffle_indices_buffer[6] = group_vectors[vec_idx_base + 6]->at(m) & 0x0F;
    shuffle_indices_buffer[7] = group_vectors[vec_idx_base + 7]->at(m) & 0x0F;
    shuffle_indices_buffer[8] = group_vectors[vec_idx_base + 8]->at(m) & 0x0F;
    shuffle_indices_buffer[9] = group_vectors[vec_idx_base + 9]->at(m) & 0x0F;
    shuffle_indices_buffer[10] = group_vectors[vec_idx_base + 10]->at(m) & 0x0F;
    shuffle_indices_buffer[11] = group_vectors[vec_idx_base + 11]->at(m) & 0x0F;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  for (int m = GROUPING_COMPONENTS; m < M; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) >> 4;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) >> 4;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) >> 4;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) >> 4;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) >> 4;
    shuffle_indices_buffer[5] = group_vectors[vec_idx_base + 5]->at(m) >> 4;
    shuffle_indices_buffer[6] = group_vectors[vec_idx_base + 6]->at(m) >> 4;
    shuffle_indices_buffer[7] = group_vectors[vec_idx_base + 7]->at(m) >> 4;
    shuffle_indices_buffer[8] = group_vectors[vec_idx_base + 8]->at(m) >> 4;
    shuffle_indices_buffer[9] = group_vectors[vec_idx_base + 9]->at(m) >> 4;
    shuffle_indices_buffer[10] = group_vectors[vec_idx_base + 10]->at(m) >> 4;
    shuffle_indices_buffer[11] = group_vectors[vec_idx_base + 11]->at(m) >> 4;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  __m128i lower_bound_vec = _mm_set1_epi8(lower_bound_signed);
  __m128i mask1 = _mm_cmpgt_epi8(lower_bound_vec, accumulated_sums);
  __m128i mask2 = _mm_cmpeq_epi8(lower_bound_vec, accumulated_sums);
  return _mm_or_si128(mask1, mask2);
}

// Processes 13 vectors
inline __m128i
process_thirteen(size_t vec_idx_base,
                 const std::vector<const vec_u8 *> &group_vectors,
                 const __m128i small_tables[M], int8_t lower_bound_signed) {
  uint8_t shuffle_indices_buffer[16];
  __m128i accumulated_sums = _mm_setzero_si128();

  for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) & 0x0F;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) & 0x0F;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) & 0x0F;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) & 0x0F;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) & 0x0F;
    shuffle_indices_buffer[5] = group_vectors[vec_idx_base + 5]->at(m) & 0x0F;
    shuffle_indices_buffer[6] = group_vectors[vec_idx_base + 6]->at(m) & 0x0F;
    shuffle_indices_buffer[7] = group_vectors[vec_idx_base + 7]->at(m) & 0x0F;
    shuffle_indices_buffer[8] = group_vectors[vec_idx_base + 8]->at(m) & 0x0F;
    shuffle_indices_buffer[9] = group_vectors[vec_idx_base + 9]->at(m) & 0x0F;
    shuffle_indices_buffer[10] = group_vectors[vec_idx_base + 10]->at(m) & 0x0F;
    shuffle_indices_buffer[11] = group_vectors[vec_idx_base + 11]->at(m) & 0x0F;
    shuffle_indices_buffer[12] = group_vectors[vec_idx_base + 12]->at(m) & 0x0F;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  for (int m = GROUPING_COMPONENTS; m < M; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) >> 4;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) >> 4;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) >> 4;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) >> 4;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) >> 4;
    shuffle_indices_buffer[5] = group_vectors[vec_idx_base + 5]->at(m) >> 4;
    shuffle_indices_buffer[6] = group_vectors[vec_idx_base + 6]->at(m) >> 4;
    shuffle_indices_buffer[7] = group_vectors[vec_idx_base + 7]->at(m) >> 4;
    shuffle_indices_buffer[8] = group_vectors[vec_idx_base + 8]->at(m) >> 4;
    shuffle_indices_buffer[9] = group_vectors[vec_idx_base + 9]->at(m) >> 4;
    shuffle_indices_buffer[10] = group_vectors[vec_idx_base + 10]->at(m) >> 4;
    shuffle_indices_buffer[11] = group_vectors[vec_idx_base + 11]->at(m) >> 4;
    shuffle_indices_buffer[12] = group_vectors[vec_idx_base + 12]->at(m) >> 4;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  __m128i lower_bound_vec = _mm_set1_epi8(lower_bound_signed);
  __m128i mask1 = _mm_cmpgt_epi8(lower_bound_vec, accumulated_sums);
  __m128i mask2 = _mm_cmpeq_epi8(lower_bound_vec, accumulated_sums);
  return _mm_or_si128(mask1, mask2);
}

// Processes 14 vectors
inline __m128i
process_fourteen(size_t vec_idx_base,
                 const std::vector<const vec_u8 *> &group_vectors,
                 const __m128i small_tables[M], int8_t lower_bound_signed) {
  uint8_t shuffle_indices_buffer[16];
  __m128i accumulated_sums = _mm_setzero_si128();

  for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) & 0x0F;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) & 0x0F;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) & 0x0F;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) & 0x0F;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) & 0x0F;
    shuffle_indices_buffer[5] = group_vectors[vec_idx_base + 5]->at(m) & 0x0F;
    shuffle_indices_buffer[6] = group_vectors[vec_idx_base + 6]->at(m) & 0x0F;
    shuffle_indices_buffer[7] = group_vectors[vec_idx_base + 7]->at(m) & 0x0F;
    shuffle_indices_buffer[8] = group_vectors[vec_idx_base + 8]->at(m) & 0x0F;
    shuffle_indices_buffer[9] = group_vectors[vec_idx_base + 9]->at(m) & 0x0F;
    shuffle_indices_buffer[10] = group_vectors[vec_idx_base + 10]->at(m) & 0x0F;
    shuffle_indices_buffer[11] = group_vectors[vec_idx_base + 11]->at(m) & 0x0F;
    shuffle_indices_buffer[12] = group_vectors[vec_idx_base + 12]->at(m) & 0x0F;
    shuffle_indices_buffer[13] = group_vectors[vec_idx_base + 13]->at(m) & 0x0F;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  for (int m = GROUPING_COMPONENTS; m < M; ++m) {
    shuffle_indices_buffer[0] = group_vectors[vec_idx_base]->at(m) >> 4;
    shuffle_indices_buffer[1] = group_vectors[vec_idx_base + 1]->at(m) >> 4;
    shuffle_indices_buffer[2] = group_vectors[vec_idx_base + 2]->at(m) >> 4;
    shuffle_indices_buffer[3] = group_vectors[vec_idx_base + 3]->at(m) >> 4;
    shuffle_indices_buffer[4] = group_vectors[vec_idx_base + 4]->at(m) >> 4;
    shuffle_indices_buffer[5] = group_vectors[vec_idx_base + 5]->at(m) >> 4;
    shuffle_indices_buffer[6] = group_vectors[vec_idx_base + 6]->at(m) >> 4;
    shuffle_indices_buffer[7] = group_vectors[vec_idx_base + 7]->at(m) >> 4;
    shuffle_indices_buffer[8] = group_vectors[vec_idx_base + 8]->at(m) >> 4;
    shuffle_indices_buffer[9] = group_vectors[vec_idx_base + 9]->at(m) >> 4;
    shuffle_indices_buffer[10] = group_vectors[vec_idx_base + 10]->at(m) >> 4;
    shuffle_indices_buffer[11] = group_vectors[vec_idx_base + 11]->at(m) >> 4;
    shuffle_indices_buffer[12] = group_vectors[vec_idx_base + 12]->at(m) >> 4;
    shuffle_indices_buffer[13] = group_vectors[vec_idx_base + 13]->at(m) >> 4;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  __m128i lower_bound_vec = _mm_set1_epi8(lower_bound_signed);
  __m128i mask1 = _mm_cmpgt_epi8(lower_bound_vec, accumulated_sums);
  __m128i mask2 = _mm_cmpeq_epi8(lower_bound_vec, accumulated_sums);
  return _mm_or_si128(mask1, mask2);
}

// Processes 15 vectors
inline __m128i process_fifteen(size_t vec_idx_base,
                               const std::vector<const vec_u8 *> &group_vectors,
                               const __m128i small_tables[M],
                               int8_t lower_bound_signed) {
  uint8_t shuffle_indices_buffer[16];
  __m128i accumulated_sums = _mm_setzero_si128();

  for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
    for (int i = 0; i < 15; ++i)
      shuffle_indices_buffer[i] = group_vectors[vec_idx_base + i]->at(m) & 0x0F;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  for (int m = GROUPING_COMPONENTS; m < M; ++m) {
    for (int i = 0; i < 15; ++i)
      shuffle_indices_buffer[i] = group_vectors[vec_idx_base + i]->at(m) >> 4;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  __m128i lower_bound_vec = _mm_set1_epi8(lower_bound_signed);
  __m128i mask1 = _mm_cmpgt_epi8(lower_bound_vec, accumulated_sums);
  __m128i mask2 = _mm_cmpeq_epi8(lower_bound_vec, accumulated_sums);
  return _mm_or_si128(mask1, mask2);
}

// Processes 16 vectors
inline __m128i process_sixteen(size_t vec_idx_base,
                               const std::vector<const vec_u8 *> &group_vectors,
                               const __m128i small_tables[M],
                               int8_t lower_bound_signed) {
  uint8_t shuffle_indices_buffer[16];
  __m128i accumulated_sums = _mm_setzero_si128();

  for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
    for (int i = 0; i < 16; ++i)
      shuffle_indices_buffer[i] = group_vectors[vec_idx_base + i]->at(m) & 0x0F;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  for (int m = GROUPING_COMPONENTS; m < M; ++m) {
    for (int i = 0; i < 16; ++i)
      shuffle_indices_buffer[i] = group_vectors[vec_idx_base + i]->at(m) >> 4;
    __m128i shuffle_mask = _mm_loadu_si128(
        reinterpret_cast<const __m128i *>(shuffle_indices_buffer));
    __m128i shuffled_values = _mm_shuffle_epi8(small_tables[m], shuffle_mask);
    accumulated_sums = _mm_adds_epu8(accumulated_sums, shuffled_values);
  }

  __m128i lower_bound_vec = _mm_set1_epi8(lower_bound_signed);
  __m128i mask1 = _mm_cmpgt_epi8(lower_bound_vec, accumulated_sums);
  __m128i mask2 = _mm_cmpeq_epi8(lower_bound_vec, accumulated_sums);
  return _mm_or_si128(mask1, mask2);
}

// PQ Fast Scan implementation
std::priority_queue<std::pair<float, int>>
run_pq_scan_fast(db_pqcodes_t &db,
                 const std::vector<std::vector<float>> &dist_tables,
                 int topK = 100) {
  std::cout << "\n--- Running PQ Fast Scan ---" << std::endl;
  std::priority_queue<std::pair<float, int>> top_results;
  Timer timer;
  timer.start();
  // 1. Group database vectors by GROUPING_COMPONENTS
  std::unordered_map<std::uint16_t, std::vector<const vec_u8 *>> grouped_db;
  for (int id = 0; id < db.size(); ++id) {
    auto &pqcode = db[id];
    std::uint16_t group_id = 0;
    for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
      group_id = (group_id << 4) | (pqcode[m] >> 4); // MSB 4 bits for group ID
    }
    pqcode.push_back(id); // Store the original index in the vector
    grouped_db[group_id].push_back(&pqcode);
  }
  // get the q_min and q_max for quantization
  float qmin = std::numeric_limits<float>::max();
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K_STAR; ++k) {
      qmin = std::min(qmin, dist_tables[m][k]);
    }
  }

  float qmax = std::numeric_limits<float>::min();
  int keep_scan_count = static_cast<int>(NUM_DB_VECTORS * KEEP_PERCENT);
  keep_scan_count = std::max(topK, keep_scan_count);
  for (int i = 0; i < keep_scan_count && i < db.size(); ++i) {
    float dist = pq_distance_original(db[i], dist_tables);
    if (top_results.size() < topK) {
      top_results.emplace(dist, i);
    } else if (dist < top_results.top().first) {
      top_results.pop();
      top_results.emplace(dist, i);
    }
  }
  qmax = top_results.top().first;

  float inv_range_0_126 = 126.0 / (qmax - qmin);
  // Create quantization tables
  std::vector<std::vector<uint8_t>> quantization_distances(
      M, std::vector<uint8_t>(K_STAR));
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K_STAR; ++k) {
      quantization_distances[m][k] =
          quantize_dist(dist_tables[m][k], qmin, qmax, inv_range_0_126);
    }
  }

  std::vector<std::vector<uint8_t>> minimum_tables(
      M - GROUPING_COMPONENTS, std::vector<uint8_t>(K_STAR / 16));
  // Calculate minimum quantized distances for each sub-quantizer
  for (int m = GROUPING_COMPONENTS; m < M; ++m) {
    for (int k = 0; k < K_STAR / 16; ++k) {
      auto smallest =
          std::min_element(quantization_distances[m].begin() + k * 16,
                           quantization_distances[m].begin() + (k + 1) * 16);
      minimum_tables[m - GROUPING_COMPONENTS][k] = *smallest;
    }
  }

  // Start Scanning
  __m128i small_tables[M];
  // Load the minimum tables into SIMD registers
  for (int m = GROUPING_COMPONENTS; m < M; ++m) {
    small_tables[m] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
        minimum_tables[m - GROUPING_COMPONENTS].data()));
  }

  for (const auto &[group_id, group_vectors] : grouped_db) {
    // Prepare small_tables for the current group (group-specific part)
    for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
      uint8_t portion_id =
          (group_id >> (4 * (GROUPING_COMPONENTS - 1 - m))) & 0x0F;
      const uint8_t *small_table_ptr_const = // Use const uint8_t*
          quantization_distances[m].data() + portion_id * 16;
      small_tables[m] = _mm_loadu_si128(
          reinterpret_cast<const __m128i *>(small_table_ptr_const));
    }
    // small_tables for m >= GROUPING_COMPONENTS are already loaded from
    // minimum_tables and are group-independent.

    uint8_t current_pruning_threshold =
        quantize_dist(top_results.top().first, qmin, qmax, inv_range_0_126);
    __m128i result_mask;
    for (int vid = 0; vid + 16 <= group_vectors.size(); vid += 16) {
      result_mask = process_sixteen(vid, group_vectors, small_tables,
                                    (int8_t)current_pruning_threshold);
      for (int i = 0; i < 16; ++i) {
        if (((uint8_t *)&result_mask)[i] ==
            0xFF) { // Check if this vector is a candidate
          const vec_u8 &pqcode = *group_vectors[vid + i];
          float dist = pq_distance_original(pqcode, dist_tables);
          int original_db_idx =
              pqcode.back(); // Assuming original index is stored
          if (top_results.size() < topK) {
            top_results.emplace(dist, original_db_idx);
            current_pruning_threshold = quantize_dist(
                top_results.top().first, qmin, qmax, inv_range_0_126);
          } else if (dist < top_results.top().first) {
            top_results.pop();
            top_results.emplace(dist, original_db_idx);
            current_pruning_threshold = quantize_dist(
                top_results.top().first, qmin, qmax, inv_range_0_126);
          }
        }
      }
    }

    // Process remaining vectors in the group
    size_t remaining = group_vectors.size() % 16;
    size_t current_idx_in_group = group_vectors.size() - remaining;
    if (remaining > 0) {
      switch (remaining) {
      case 15:
        result_mask =
            process_fifteen(current_idx_in_group, group_vectors, small_tables,
                            (int8_t)current_pruning_threshold);
        break;
      case 14:
        result_mask =
            process_fourteen(current_idx_in_group, group_vectors, small_tables,
                             (int8_t)current_pruning_threshold);
        break;
      case 13:
        result_mask =
            process_thirteen(current_idx_in_group, group_vectors, small_tables,
                             (int8_t)current_pruning_threshold);
        break;
      case 12:
        result_mask =
            process_twelve(current_idx_in_group, group_vectors, small_tables,
                           (int8_t)current_pruning_threshold);
        break;
      case 11:
        result_mask =
            process_eleven(current_idx_in_group, group_vectors, small_tables,
                           (int8_t)current_pruning_threshold);
        break;
      case 10:
        result_mask =
            process_ten(current_idx_in_group, group_vectors, small_tables,
                        (int8_t)current_pruning_threshold);
        break;
      case 9:
        result_mask =
            process_nine(current_idx_in_group, group_vectors, small_tables,
                         (int8_t)current_pruning_threshold);
        break;
      case 8:
        result_mask =
            process_eight(current_idx_in_group, group_vectors, small_tables,
                          (int8_t)current_pruning_threshold);
        break;
      case 7:
        result_mask =
            process_seven(current_idx_in_group, group_vectors, small_tables,
                          (int8_t)current_pruning_threshold);
        break;
      case 6:
        result_mask =
            process_six(current_idx_in_group, group_vectors, small_tables,
                        (int8_t)current_pruning_threshold);
        break;
      case 5:
        result_mask =
            process_five(current_idx_in_group, group_vectors, small_tables,
                         (int8_t)current_pruning_threshold);
        break;
      case 4:
        result_mask =
            process_four(current_idx_in_group, group_vectors, small_tables,
                         (int8_t)current_pruning_threshold);
        break;
      case 3:
        result_mask =
            process_three(current_idx_in_group, group_vectors, small_tables,
                          (int8_t)current_pruning_threshold);
        break;
      case 2:
        result_mask =
            process_two(current_idx_in_group, group_vectors, small_tables,
                        (int8_t)current_pruning_threshold);
        break;
      case 1:
        result_mask =
            process_one(current_idx_in_group, group_vectors, small_tables,
                        (int8_t)current_pruning_threshold);
        break;
      }
    }
    for (int i = 0; i < remaining; ++i) {
      if (((uint8_t *)&result_mask)[i] ==
          0xFF) { // Check if this vector is a candidate
        const vec_u8 &pqcode = *group_vectors[current_idx_in_group + i];
        float dist = pq_distance_original(pqcode, dist_tables);
        int original_db_idx =
            pqcode.back(); // Assuming original index is stored
        if (top_results.size() < topK) {
          top_results.emplace(dist, original_db_idx);
          current_pruning_threshold = quantize_dist(
              top_results.top().first, qmin, qmax, inv_range_0_126);
        } else if (dist < top_results.top().first) {
          top_results.pop();
          top_results.emplace(dist, original_db_idx);
          current_pruning_threshold = quantize_dist(
              top_results.top().first, qmin, qmax, inv_range_0_126);
        }
      }
    }
  }

  auto elapsed = timer.stop();
  std::cout << "PQ Fast Scan completed in " << elapsed << " seconds."
            << std::endl;
  std::cout << "Top " << topK << " results:" << std::endl;
  return top_results;
}

int main() {
  // 1. Generate optimized centroids based on section 4.3 of the paper
  centroids_t centroids = generate_centroids_optimized();

  // 2. Generate and encode database vectors
  std::cout << "Generating " << NUM_DB_VECTORS << " database vectors..."
            << std::endl;
  db_pqcodes_t db_pqcodes;
  db_pqcodes.reserve(NUM_DB_VECTORS);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis_vec_val(
      -10.0, 10.0); // Range for database vector values

  for (int i = 0; i < NUM_DB_VECTORS; ++i) {
    vec v(D);
    for (int d_idx = 0; d_idx < D; ++d_idx)
      v[d_idx] = dis_vec_val(gen);
    db_pqcodes.push_back(encode_vector(v, centroids));
  }

  // 3. Generate a random query vector
  std::cout << "Generating random query vector..." << std::endl;
  vec query(D);
  for (int d_idx = 0; d_idx < D; ++d_idx)
    query[d_idx] = dis_vec_val(gen);

  // 4. Compute distance tables for the query
  std::vector<std::vector<float>> dist_tables =
      compute_distance_tables(query, centroids);

  // 5. Execute and compare both methods
  int topK = 100;
  run_pq_scan_original(db_pqcodes, dist_tables, topK);
  run_pq_scan_fast(db_pqcodes, dist_tables, topK);

  return 0;
}
