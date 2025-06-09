#include <algorithm>
#include <chrono>
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

// PQ Fast Scan implementation
std::priority_queue<std::pair<float, int>>
run_pq_scan_fast(const db_pqcodes_t &db,
                 const std::vector<std::vector<float>> &dist_tables,
                 int topK = 100) {
  std::cout << "\n--- Running PQ Fast Scan ---" << std::endl;
  std::priority_queue<std::pair<float, int>> top_results;
  Timer timer;
  timer.start();
  // 1. Group database vectors by GROUPING_COMPONENTS
  std::unordered_map<std::uint16_t, std::vector<const vec_u8 *>> grouped_db;
  for (const auto &pqcode : db) {
    std::uint16_t group_id = 0;
    for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
      group_id = (group_id << 4) | (pqcode[m] >> 4); // MSB 4 bits for group ID
    }
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
      M, std::vector<uint8_t>(K_STAR / 16));
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
  for (const auto &[group_id, group_vectors] : grouped_db) {
    for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
      small_tables[m] = _mm_loadu_si128(const __m128i_u *p)
    }
  }

  // // 4. Iterate through all groups for scanning
  // long pruned_count = 0;
  // long total_processed_for_pruning = 0;

  // for (const auto &pair : grouped_db) {
  //   const auto &group_id_key = pair.first; // Though unused, kept for
  //   structure const auto &group_vectors = pair.second;

  //   // 4.1 Create S_0 to S_GROUPING_COMPONENTS-1 for the current group
  //   std::vector<vec_u8> S_group_specific_tables(GROUPING_COMPONENTS,
  //                                               vec_u8(16));
  //   for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
  //     // Extract portion_id for this component from group_id_key
  //     // MSB of group_id_key corresponds to m=0, LSB to
  //     m=GROUPING_COMPONENTS-1 uint8_t portion_id =
  //         (group_id_key >> (4 * (GROUPING_COMPONENTS - 1 - m))) & 0x0F;
  //     int start_k_in_dist_table = portion_id * 16;
  //     for (int i = 0; i < 16; ++i) { // Each portion has 16 elements
  //       S_group_specific_tables[m][i] =
  //           quantize_dist(dist_tables[m][start_k_in_dist_table + i],
  //                         q_min_fixed, q_max_fixed, inv_range_0_126_fixed);
  //     }
  //   }

  //   // 4.2 Load all 8 small tables into SIMD registers
  //   __m128i s_reg[M];
  //   for (int m = 0; m < GROUPING_COMPONENTS; ++m) {
  //     s_reg[m] = _mm_loadu_si128(
  //         reinterpret_cast<const __m128i
  //         *>(S_group_specific_tables[m].data()));
  //   }
  //   for (int m = GROUPING_COMPONENTS; m < M; ++m) {
  //     s_reg[m] = _mm_loadu_si128(reinterpret_cast<const __m128i *>(
  //         S_min_tables[m - GROUPING_COMPONENTS].data()));
  //   }

  //   // 4.3 Perform SIMD scan within the group
  //   for (size_t vec_idx = 0; vec_idx < group_vectors.size(); ++vec_idx) {
  //     const vec_u8 &pqcode = *group_vectors[vec_idx];
  //     total_processed_for_pruning++;

  //     // Calculate lower bound using SIMD
  //     __m128i accumulated_sum_reg = _mm_setzero_si128();
  //     for (int m = 0; m < M; ++m) {
  //       uint8_t index_in_s_table;
  //       if (m < GROUPING_COMPONENTS) {
  //         index_in_s_table = pqcode[m] & 0x0F; // LSB 4 bits for S0-S3
  //       } else {
  //         index_in_s_table = pqcode[m] >> 4; // MSB 4 bits for S4-S7
  //       }

  //       __m128i shuffle_mask =
  //           _mm_set1_epi8(static_cast<char>(index_in_s_table));
  //       __m128i looked_up_val_broadcasted =
  //           _mm_shuffle_epi8(s_reg[m], shuffle_mask);
  //       accumulated_sum_reg =
  //           _mm_adds_epu8(accumulated_sum_reg, looked_up_val_broadcasted);
  //     }

  //     // Extract lower bound result (only take the first lane as all lanes
  //     are
  //     // the same)
  //     uint8_t lower_bound_quantized =
  //         static_cast<uint8_t>(_mm_extract_epi8(accumulated_sum_reg, 0));

  //     if (lower_bound_quantized >= current_min_dist_quantized) {
  //       pruned_count++;
  //       continue;
  //     }

  //     // If not pruned, calculate exact distance.
  //     // Find original index in db to compare results.
  //     // Here we assume pqcode_ptr can uniquely identify the vector,
  //     // but for updating min_idx, a global index is needed.
  //     // For simplicity, we don't track the global min_idx here, only update
  //     the
  //     // distance.
  //     float dist_float = pq_distance_original(pqcode, dist_tables);
  //     if (dist_float < current_min_dist_float) {
  //       current_min_dist_float = dist_float;
  //       // Update quantized nearest distance for subsequent pruning
  //       current_min_dist_quantized =
  //           quantize_dist(current_min_dist_float, q_min_fixed, q_max_fixed,
  //                         inv_range_0_126_fixed);
  //     }
  //   }
  // }

  // double elapsed = timer.stop();
  // std::cout << "PQ Fast Scan finished in: " << elapsed << " ms" << std::endl;
  // if (total_processed_for_pruning > 0) {
  //   std::cout << "Pruned " << pruned_count << " / "
  //             << total_processed_for_pruning << " vectors ("
  //             << (100.0 * pruned_count / total_processed_for_pruning) << "%)"
  //             << std::endl;
  // } else {
  //   std::cout << "No vectors processed for pruning." << std::endl;
  // }
  // std::cout << "Min distance found: " << current_min_dist_float << std::endl;
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
