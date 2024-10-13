#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include "inc/main.hxx"

using namespace std;




#pragma region CONFIGURATION
#ifndef TYPE
/** Type of edge weights. */
#define TYPE float
#endif
#ifndef MAX_THREADS
/** Maximum number of threads to use. */
#define MAX_THREADS 64
#endif
#ifndef REPEAT_BATCH
/** Number of times to repeat each batch. */
#define REPEAT_BATCH 5
#endif
#ifndef REPEAT_METHOD
/** Number of times to repeat each method. */
#define REPEAT_METHOD 5
#endif
#pragma endregion




#pragma region METHODS
#pragma region HELPERS
/**
 * Obtain the modularity of community structure on a graph.
 * @param x original graph
 * @param a louvain result
 * @param M sum of edge weights
 * @returns modularity
 */
template <class G, class K>
inline double getModularity(const G& x, const LouvainResult<K>& a, double M) {
  auto fc = [&](auto u) { return a.membership[u]; };
  return modularityBy(x, fc, M, 1.0);
}

/**
 * Obtain the modularity of community structure on a graph.
 * @param x original graph
 * @param a leiden result
 * @param M sum of edge weights
 * @returns modularity
 */
template <class G, class K>
inline double getModularity(const G& x, const LeidenResult<K>& a, double M) {
  auto fc = [&](auto u) { return a.membership[u]; };
  return modularityBy(x, fc, M, 1.0);
}

/**
 * Get the refinement time of an algorithm result.
 * @param a louvain result
 * @returns refinement time
 */
template <class K, class W>
inline float refinementTime(const LouvainResult<K, W>& a) {
  return 0;
}

/**
 * Get the refinement time of an algorithm result.
 * @param a leiden result
 * @returns refinement time
 */
template <class K, class W>
inline float refinementTime(const LeidenResult<K, W>& a) {
  return a.refinementTime;
}

/**
 * Get the splitting time of an algorithm result.
 * @param a louvain result
 * @returns splitting time
 */
template <class K, class W>
inline float splittingTime(const LouvainResult<K, W>& a) {
  return 0;
}

/**
 * Get the splitting time of an algorithm result.
 * @param a leiden result
 * @returns splitting time
 */
template <class K, class W>
inline float splittingTime(const LeidenResult<K, W>& a) {
  return a.splittingTime;
}

/**
 * Get the tracking time of an algorithm result.
 * @param a louvain result
 * @returns tracking time
 */
template <class K, class W>
inline float trackingTime(const LouvainResult<K, W>& a) {
  return a.trackingTime;
}


/**
 * Get the tracking time of an algorithm result.
 * @param a leiden result
 * @returns tracking time
 */
template <class K, class W>
inline float trackingTime(const LeidenResult<K, W>& a) {
  return a.trackingTime;
}
#pragma endregion




#pragma region PERFORM EXPERIMENT
/**
 * Perform the experiment.
 * @param x input graph
 * @param fstream input file stream
 * @param rows number of rows/vetices in the graph
 * @param size number of lines/edges (temporal) in the graph
 * @param batchFraction fraction of edges to use in each batch
 */
template <class G>
void runExperiment(G& x, istream& fstream, size_t rows, size_t size, double batchFraction) {
  using K = typename G::key_type;
  using V = typename G::edge_value_type;
  using W = LOUVAIN_WEIGHT_TYPE;
  int repeat     = REPEAT_METHOD;
  int numThreads = MAX_THREADS;
  double M = edgeWeightOmp(x)/2;
  // Follow a specific result logging format, which can be easily parsed later.
  auto glog = [&](const auto& ans, const char *technique, int numThreads, const auto& y, auto M, auto deletionsf, auto insertionsf, double Q=0.6) {
    printf(
      "{-%.3e/+%.3e batchf, %03d threads, %.1f refinetol} -> "
      "{%09.1fms, %09.1fms mark, %09.1fms init, %09.1fms firstpass, %09.1fms locmove, %09.1fms split, %09.1fms refine, %09.1fms aggr, %09.1fms track, %.3e aff, %04d iters, %03d passes, %01.9f modularity, %zu/%zu disconnected} %s\n",
      double(deletionsf), double(insertionsf), numThreads, Q,
      ans.time, ans.markingTime, ans.initializationTime, ans.firstPassTime, ans.localMoveTime, splittingTime(ans), refinementTime(ans), ans.aggregationTime, trackingTime(ans),
      double(ans.affectedVertices), ans.iterations, ans.passes, getModularity(y, ans, M),
      countValue(communitiesDisconnectedOmp(y, ans.membership), char(1)),
      communities(y, ans.membership).size(), technique
    );
  };
  vector<tuple<K, K, V>> deletions;
  vector<tuple<K, K, V>> insertions;
  // Get community memberships on original graph (static).
  auto c0 = leidenStaticOmp(x, {5});
  glog(c0, "leidenStaticOmpOriginal", MAX_THREADS, x, M, 0.0, 0.0);
  auto CM2 = c0.membership;
  auto CV2 = c0.vertexWeight;
  auto CC2 = c0.communityWeight;
  auto CD2 = c0.communityWeightChanged;
  auto CM3 = c0.membership;
  auto CV3 = c0.vertexWeight;
  auto CC3 = c0.communityWeight;
  auto CD3 = c0.communityWeightChanged;
  auto CM4 = c0.membership;
  auto CV4 = c0.vertexWeight;
  auto CC4 = c0.communityWeight;
  auto CD4 = c0.communityWeightChanged;
  // Get community memberships on updated graph (dynamic).
  for (int batchIndex=0; batchIndex<BATCH_LENGTH; ++batchIndex) {
    auto y = duplicate(x);
    insertions.clear();
    auto fb = [&](auto u, auto v, auto w) {
      insertions.push_back({u, v, w});
    };
    readTemporalDo(fstream, false, true, rows, size_t(batchFraction * size), fb);
    tidyBatchUpdateU(deletions, insertions, y);
    applyBatchUpdateOmpU(y, deletions, insertions);
    LOG(""); print(y); printf(" (insertions=%zu)\n", insertions.size());
    double  M = edgeWeightOmp(y)/2;
    auto flog = [&](const auto& ans, const char *technique) {
      glog(ans, technique, numThreads, y, M, 0.0, batchFraction);
    };
    // Find static Leiden.
    auto c1 = leidenStaticOmp(y, {repeat});
    flog(c1, "leidenStaticOmp");
    // Find naive-dynamic Leiden.
    auto c2 = leidenNaiveDynamicOmp<true>(y, deletions, insertions, CM2, CV2, CC2, CD2, {repeat});
    flog(c2, "leidenNaiveDynamicOmpSelsplit");
    // Find delta-screening based dynamic Leiden.
    auto c3 = leidenDynamicDeltaScreeningOmp<true>(y, deletions, insertions, CM3, CV3, CC3, CD3, {repeat});
    flog(c3, "leidenDynamicDeltaScreeningOmpSelsplit");
    // Find frontier based dynamic Leiden.
    auto c4 = leidenDynamicFrontierOmp<true>(y, deletions, insertions, CM4, CV4, CC4, CD4, {repeat});
    flog(c4, "leidenDynamicFrontierOmpSelsplit");
    copyValuesOmpW(CM2, c2.membership);
    copyValuesOmpW(CV2, c2.vertexWeight);
    copyValuesOmpW(CC2, c2.communityWeight);
    copyValuesOmpW(CD2, c2.communityWeightChanged);
    copyValuesOmpW(CM3, c3.membership);
    copyValuesOmpW(CV3, c3.vertexWeight);
    copyValuesOmpW(CC3, c3.communityWeight);
    copyValuesOmpW(CD3, c3.communityWeightChanged);
    copyValuesOmpW(CM4, c4.membership);
    copyValuesOmpW(CV4, c4.vertexWeight);
    copyValuesOmpW(CC4, c4.communityWeight);
    copyValuesOmpW(CD4, c4.communityWeightChanged);
    swap(x, y);
  }
}


/**
 * Main function.
 * @param argc argument count
 * @param argv argument values
 * @returns zero on success, non-zero on failure
 */
int main(int argc, char **argv) {
  using K = uint32_t;
  using V = TYPE;
  install_sigsegv();
  char *file     = argv[1];
  size_t rows = strtoull(argv[2], nullptr, 10);
  size_t size = strtoull(argv[3], nullptr, 10);
  double batchFraction = strtod(argv[5], nullptr);
  omp_set_num_threads(MAX_THREADS);
  LOG("OMP_NUM_THREADS=%d\n", MAX_THREADS);
  LOG("Loading graph %s ...\n", file);
  DiGraph<K, None, V> x;
  ifstream fstream(file);
  readTemporalOmpW(x, fstream, false, true, rows, size_t(0.90 * size)); LOG(""); print(x); printf(" (90%%)\n");
  symmetrizeOmpU(x); LOG(""); print(x); printf(" (symmetrize)\n");
  runExperiment(x, fstream, rows, size, batchFraction);
  printf("\n");
  return 0;
}
#pragma endregion
#pragma endregion
