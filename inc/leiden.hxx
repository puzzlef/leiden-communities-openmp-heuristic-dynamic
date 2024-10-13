#pragma once
#include <utility>
#include <limits>
#include <tuple>
#include <vector>
#include <algorithm>
#include "_main.hxx"
#include "Graph.hxx"
#include "properties.hxx"
#include "csr.hxx"
#include "bfs.hxx"
#include <omp.h>

using std::numeric_limits;
using std::tuple;
using std::vector;
using std::make_pair;
using std::move;
using std::swap;
using std::get;
using std::min;
using std::max;




#pragma region TYPES
/**
 * Options for Leiden algorithm.
 */
struct LeidenOptions {
  #pragma region DATA
  /** Number of times to repeat the algorithm [1]. */
  int repeat;
  /** Resolution parameter for modularity [1]. */
  double resolution;
  /** Tolerance for convergence [1e-2]. */
  double tolerance;
  /** Tolerance for aggregation [0.8]. */
  double aggregationTolerance;
  /** Tolerance for refinement [0.6]. */
  double refinementTolerance;
  /** Tolerance drop factor after each pass [10]. */
  double toleranceDrop;
  /** Maximum number of iterations per pass [20]. */
  int maxIterations;
  /** Maximum number of passes [10]. */
  int maxPasses;
  #pragma endregion


  #pragma region CONSTRUCTORS
  /**
   * Define options for Leiden algorithm.
   * @param repeat number of times to repeat the algorithm [1]
   * @param resolution resolution parameter for modularity [1]
   * @param tolerance tolerance for convergence [1e-2]
   * @param aggregationTolerance tolerance for aggregation [0.8]
   * @param refinementTolerance tolerance for refinement [0.6]
   * @param toleranceDrop tolerance drop factor after each pass [10]
   * @param maxIterations maximum number of iterations per pass [20]
   * @param maxPasses maximum number of passes [10]
   */
  LeidenOptions(int repeat=1, double resolution=1, double tolerance=1e-2, double aggregationTolerance=1.0, double refinementTolerance=0.6, double toleranceDrop=10, int maxIterations=20, int maxPasses=10) :
  repeat(repeat), resolution(resolution), tolerance(tolerance), aggregationTolerance(aggregationTolerance), refinementTolerance(refinementTolerance), toleranceDrop(toleranceDrop), maxIterations(maxIterations), maxPasses(maxPasses) {}
  #pragma endregion
};


/** Weight to be used in hashtable. */
#define LEIDEN_WEIGHT_TYPE double




/**
 * Result of Leiden algorithm.
 * @tparam K key type (vertex-id)
 * @tparam W weight type
 */
template <class K, class W=LEIDEN_WEIGHT_TYPE>
struct LeidenResult {
  #pragma region DATA
  /** Community membership each vertex belongs to. */
  vector<K> membership;
  /** Total edge weight of each vertex. */
  vector<W> vertexWeight;
  /** Total edge weight of each community. */
  vector<W> communityWeight;
  /** Change in Total edge weight of each community, since it was last refined. */
  vector<W> communityWeightChanged;
  /** Number of iterations performed. */
  int iterations;
  /** Number of passes performed. */
  int passes;
  /** Time spent in milliseconds. */
  float time;
  /** Time spent in milliseconds for initial marking of affected vertices. */
  float markingTime;
  /** Time spent in initializing community memberships and total vertex/community weights. */
  float initializationTime;
  /** Time spent in milliseconds in first pass. */
  float firstPassTime;
  /** Time spent in milliseconds in local-moving phase. */
  float localMoveTime;
  /** Time spent in milliseconds in splitting phase. */
  float splittingTime;
  /** Time spent in milliseconds in refinement phase. */
  float refinementTime;
  /** Time spent in milliseconds in aggregation phase. */
  float aggregationTime;
  /** Time spent in milliseconds in tracking phase. */
  float trackingTime;
  /** Number of vertices initially marked as affected. */
  size_t affectedVertices;
  /** Number of changed communities. */
  size_t changedCommunities;
  /** Number of split communities. */
  size_t splitCommunities;
  /** Number of bottom-level communities. */
  size_t bottomLevelCommunities;
  #pragma endregion


  #pragma region CONSTRUCTORS
  /**
   * Result of Leiden algorithm.
   * @param membership community membership each vertex belongs to
   * @param vertexWeight total edge weight of each vertex
   * @param communityWeight total edge weight of each community
   * @param communityWeightChanged change in total edge weight of each community, since it was last refined
   * @param iterations number of iterations performed
   * @param passes number of passes performed
   * @param time time spent in milliseconds
   * @param markingTime time spent in milliseconds for initial marking of affected vertices
   * @param initializationTime time spent in initializing community memberships and total vertex/community weights
   * @param firstPassTime time spent in milliseconds in first pass
   * @param localMoveTime time spent in milliseconds in local-moving phase
   * @param splittingTime time spent in milliseconds in splitting phase
   * @param refinementTime time spent in milliseconds in refinement phase
   * @param aggregationTime time spent in milliseconds in aggregation phase
   * @param trackingTime time spent in milliseconds in tracking phase
   * @param affectedVertices number of vertices initially marked as affected
   * @param changedCommunities number of changed communities
   * @param splitCommunities number of split communities
   * @param bottomLevelCommunities number of bottom-level communities
   */
  LeidenResult(vector<K>&& membership, vector<W>&& vertexWeight, vector<W>&& communityWeight, vector<W>&& communityWeightChanged, int iterations=0, int passes=0, float time=0, float markingTime=0, float initializationTime=0, float firstPassTime=0, float localMoveTime=0, float splittingTime=0, float refinementTime=0, float aggregationTime=0, float trackingTime=0, size_t affectedVertices=0, size_t changedCommunities=0, size_t splitCommunities=0, size_t bottomLevelCommunities=0) :
  membership(membership), vertexWeight(vertexWeight), communityWeight(communityWeight), communityWeightChanged(communityWeightChanged), iterations(iterations), passes(passes), time(time), markingTime(markingTime), initializationTime(initializationTime), firstPassTime(firstPassTime), localMoveTime(localMoveTime), splittingTime(splittingTime), refinementTime(refinementTime), aggregationTime(aggregationTime), trackingTime(trackingTime), affectedVertices(affectedVertices), changedCommunities(changedCommunities), splitCommunities(splitCommunities), bottomLevelCommunities(bottomLevelCommunities) {}


  /**
   * Result of Leiden algorithm.
   * @param membership community membership each vertex belongs to (moved)
   * @param vertexWeight total edge weight of each vertex (moved)
   * @param communityWeight total edge weight of each community (moved)
   * @param communityWeightChanged change in total edge weight of each community, since it was last refined (moved)
   * @param iterations number of iterations performed
   * @param passes number of passes performed
   * @param time time spent in milliseconds
   * @param markingTime time spent in milliseconds for initial marking of affected vertices
   * @param initializationTime time spent in initializing community memberships and total vertex/community weights
   * @param firstPassTime time spent in milliseconds in first pass
   * @param localMoveTime time spent in milliseconds in local-moving phase
   * @param splittingTime time spent in milliseconds in splitting phase
   * @param refinementTime time spent in milliseconds in refinement phase
   * @param aggregationTime time spent in milliseconds in aggregation phase
   * @param trackingTime time spent in milliseconds in tracking phase
   * @param affectedVertices number of vertices initially marked as affected
   * @param changedCommunities number of changed communities
   * @param splitCommunities number of split communities
   * @param bottomLevelCommunities number of bottom-level communities
   */
  LeidenResult(vector<K>& membership, vector<W>& vertexWeight, vector<W>& communityWeight, vector<W>& communityWeightChanged, int iterations=0, int passes=0, float time=0, float markingTime=0, float initializationTime=0, float firstPassTime=0, float localMoveTime=0, float splittingTime=0, float refinementTime=0, float aggregationTime=0, float trackingTime=0, size_t affectedVertices=0, size_t changedCommunities=0, size_t splitCommunities=0, size_t bottomLevelCommunities=0) :
  membership(move(membership)), vertexWeight(move(vertexWeight)), communityWeight(move(communityWeight)), communityWeightChanged(move(communityWeightChanged)), iterations(iterations), passes(passes), time(time), markingTime(markingTime), initializationTime(initializationTime), firstPassTime(firstPassTime), localMoveTime(localMoveTime), splittingTime(splittingTime), refinementTime(refinementTime), aggregationTime(aggregationTime), trackingTime(trackingTime), affectedVertices(affectedVertices), changedCommunities(changedCommunities), splitCommunities(splitCommunities), bottomLevelCommunities(bottomLevelCommunities) {}
  #pragma endregion
};
#pragma endregion




#pragma region METHODS
#pragma region HASHTABLES
/**
 * Allocate a number of hashtables.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 * @param S size of each hashtable
 */
template <class K, class W>
inline void leidenAllocateHashtablesW(vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, size_t S) {
  size_t N = vcs.size();
  for (size_t i=0; i<N; ++i) {
    vcs[i]   = new vector<K>();
    vcout[i] = new vector<W>(S);
  }
}


/**
 * Free a number of hashtables.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 */
template <class K, class W>
inline void leidenFreeHashtablesW(vector<vector<K>*>& vcs, vector<vector<W>*>& vcout) {
  size_t N = vcs.size();
  for (size_t i=0; i<N; ++i) {
    delete vcs[i];
    delete vcout[i];
  }
}
#pragma endregion




#pragma region INITIALIZE
/**
 * Find the total edge weight of each vertex.
 * @param vtot total edge weight of each vertex (updated, must be initialized)
 * @param x original graph
 */
template <class G, class W>
inline void leidenVertexWeightsOmpW(vector<W>& vtot, const G& x) {
  using  K = typename G::key_type;
  size_t S = x.span();
  #pragma omp parallel for schedule(dynamic, 2048)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    x.forEachEdge(u, [&](auto v, auto w) { vtot[u] += w; });
  }
}


/**
 * Find the total edge weight of each community.
 * @param ctot total edge weight of each community (updated, must be initialized)
 * @param x original graph
 * @param vcom community each vertex belongs to
 * @param vtot total edge weight of each vertex
 */
template <class G, class K, class W>
inline void leidenCommunityWeightsOmpW(vector<W>& ctot, const G& x, const vector<K>& vcom, const vector<W>& vtot) {
  size_t S = x.span();
  #pragma omp parallel for schedule(static, 2048)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    K c = vcom[u];
    #pragma omp atomic
    ctot[c] += vtot[u];
  }
}


/**
 * Obtain any vertex from each community.
 * @param cvtx any vertex from each community (MAX => empty)
 * @param x original graph
 * @param vcom community each vertex belongs to
 */
template <class G, class K>
inline void leidenCommunityAnyVertexOmpW(vector<K>& cvtx, const G& x, const vector<K>& vcom) {
  const  K EMPTY = numeric_limits<K>::max();
  size_t S = x.span();
  fillValueOmpU(cvtx, EMPTY);
  #pragma omp parallel for schedule(static, 2048)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    K c = vcom[u];
    if (cvtx[c]==EMPTY) cvtx[c] = u;
  }
}


/**
 * Rename communities based on the ID of any vertex within each community.
 * @param vcom community each vertex belongs to (output)
 * @param ctot total edge weight of each community (output)
 * @param cchg communities that have changed (output)
 * @param cspt communities that are to be split (output)
 * @param cvtx any vertex from each community (scratch)
 * @param x original graph
 * @param vdom old community each vertex belongs to
 * @param dtot total edge weight of each old community
 * @param dchg old communities that have changed
 * @param dspt old communities that are to be split
 */
template <bool SELSPLIT=false, class G, class K, class W, class B>
inline void leidenSubsetRenameCommunitiesOmpW(vector<K>& vcom, vector<W>& ctot, vector<B>& cchg, vector<B>& cspt, vector<K>& cvtx, const G& x, const vector<K>& vdom, const vector<W>& dtot, const vector<B>& dchg, const vector<B>& dspt) {
  const  K EMPTY = numeric_limits<K>::max();
  size_t S = x.span();
  // Find any vertex from each community.
  leidenCommunityAnyVertexOmpW(cvtx, x, vdom);
  // Update community weights.
  #pragma omp parallel for schedule(static, 2048)
  for (K d=0; d<S; ++d) {
    K c = cvtx[d];
    if (c==EMPTY) continue;
    ctot[c] = dtot[d];
    cchg[c] = dchg[d];
    if (SELSPLIT) cspt[c] = dspt[d];
  }
  // Update community memberships.
  #pragma omp parallel for schedule(static, 2048)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    K d = vdom[u];
    K c = cvtx[d];
    vcom[u] = c;
  }
}


/**
 * Initialize communities such that each vertex is its own community.
 * @param vcom community each vertex belongs to (updated, must be initialized)
 * @param ctot total edge weight of each community (updated, must be initialized)
 * @param x original graph
 * @param vtot total edge weight of each vertex
 * @param fr does vertex need to be reset?
 */
template <class G, class K, class W, class FR>
inline void leidenInitializeOmpW(vector<K>& vcom, vector<W>& ctot, const G& x, const vector<W>& vtot, FR fr) {
  size_t S = x.span();
  #pragma omp parallel for schedule(static, 2048)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    if (!fr(u)) continue;
    vcom[u] = u;
    ctot[u] = vtot[u];
  }
}


/**
 * Initialize communities such that each vertex is its own community.
 * @param vcom community each vertex belongs to (updated, must be initialized)
 * @param ctot total edge weight of each community (updated, must be initialized)
 * @param x original graph
 * @param vtot total edge weight of each vertex
 */
template <class G, class K, class W>
inline void leidenInitializeOmpW(vector<K>& vcom, vector<W>& ctot, const G& x, const vector<W>& vtot) {
  auto fr = [](auto u) { return true; };
  leidenInitializeOmpW(vcom, ctot, x, vtot, fr);
}


/**
 * Initialize communities from given initial communities.
 * @param vcom community each vertex belongs to (updated, must be initialized)
 * @param ctot total edge weight of each community (updated, must be initialized)
 * @param y updated graph
 * @param vtot total edge weight of each vertex
 * @param q initial community each vertex belongs to
 */
template <class G, class K, class W>
inline void leidenInitializeFromOmpW(vector<K>& vcom, vector<W>& ctot, const G& y, const vector<W>& vtot, const vector<K>& q) {
  size_t S = y.span();
  #pragma omp parallel for schedule(static, 2048)
  for (K u=0; u<S; ++u) {
    if (!y.hasVertex(u)) continue;
    K c = q[u];
    vcom[u]  = c;
    #pragma omp atomic
    ctot[c] += vtot[u];
  }
}


/**
 * Update weights using given edge deletions and insertions.
 * @param vtot total edge weight of each vertex (updated)
 * @param ctot total edge weight of each community (updated)
 * @param cdwt change in total edge weight of each community (updated)
 * @param y updated graph
 * @param deletions edge deletions in batch update
 * @param insertions edge insertions in batch update
 * @param vcom community each vertex belongs to
 */
template <class G, class K, class V, class W>
inline void leidenUpdateWeightsFromOmpU(vector<W>& vtot, vector<W>& ctot, vector<W>& cdwt, const G& y, const vector<tuple<K, K, V>>& deletions, const vector<tuple<K, K, V>>& insertions, const vector<K>& vcom) {
  #pragma omp parallel
  {
    for (auto [u, v, w] : deletions) {
      K c = vcom[u];
      if (belongsOmp(u)) vtot[u] -= w;
      if (belongsOmp(c)) { ctot[c] -= w; cdwt[c] += w; }
    }
    for (auto [u, v, w] : insertions) {
      K c = vcom[u];
      if (belongsOmp(u)) vtot[u] += w;
      if (belongsOmp(c)) { ctot[c] += w; cdwt[c] += w; }
    }
  }
}
#pragma endregion




#pragma region CHANGE COMMUNITY
/**
 * Scan an edge community connected to a vertex.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 * @param u given vertex
 * @param v outgoing edge vertex
 * @param w outgoing edge weight
 * @param vcom community each vertex belongs to
 * @param vcob community bound each vertex belongs to
 */
template <bool SELF=false, bool REFINE=false, class K, class V, class W>
inline void leidenScanCommunityW(vector<K>& vcs, vector<W>& vcout, K u, K v, V w, const vector<K>& vcom, const vector<K>& vcob) {
  if (!SELF && u==v) return;
  if (REFINE && vcob[u]!=vcob[v]) return;
  K c = vcom[v];
  if (!vcout[c]) vcs.push_back(c);
  vcout[c] += w;
}


/**
 * Scan an edge community connected to a vertex.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 * @param u given vertex
 * @param v outgoing edge vertex
 * @param w outgoing edge weight
 * @param vcom community each vertex belongs to
 */
template <bool SELF=false, class K, class V, class W>
inline void leidenScanCommunityW(vector<K>& vcs, vector<W>& vcout, K u, K v, V w, const vector<K>& vcom) {
  leidenScanCommunityW<SELF, false>(vcs, vcout, u, v, w, vcom, vcom);
}


/**
 * Scan communities connected to a vertex.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 * @param x original graph
 * @param u given vertex
 * @param vcom community each vertex belongs to
 * @param vcob community bound each vertex belongs to
 */
template <bool SELF=false, bool REFINE=false, class G, class K, class W>
inline void leidenScanCommunitiesW(vector<K>& vcs, vector<W>& vcout, const G& x, K u, const vector<K>& vcom, const vector<K>& vcob) {
  x.forEachEdge(u, [&](auto v, auto w) { leidenScanCommunityW<SELF, REFINE>(vcs, vcout, u, v, w, vcom, vcob); });
}


/**
 * Scan communities connected to a vertex.
 * @param vcs communities vertex u is linked to (updated)
 * @param vcout total edge weight from vertex u to community C (updated)
 * @param x original graph
 * @param u given vertex
 * @param vcom community each vertex belongs to
 */
template <bool SELF=false, class G, class K, class W>
inline void leidenScanCommunitiesW(vector<K>& vcs, vector<W>& vcout, const G& x, K u, const vector<K>& vcom) {
  leidenScanCommunitiesW<SELF, false>(vcs, vcout, x, u, vcom, vcom);
}


/**
 * Clear communities scan data.
 * @param vcs total edge weight from vertex u to community C (updated)
 * @param vcout communities vertex u is linked to (updated)
 */
template <class K, class W>
inline void leidenClearScanW(vector<K>& vcs, vector<W>& vcout) {
  for (K c : vcs)
    vcout[c] = W();
  vcs.clear();
}


/**
 * Choose connected community with best delta modularity.
 * @param x original graph
 * @param u given vertex
 * @param d previous community of vertex u
 * @param vtot total edge weight of each vertex
 * @param ctot total edge weight of each community
 * @param vcs communities vertex u is linked to
 * @param vcout total edge weight from vertex u to community C
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @returns [best community, delta modularity]
 */
template <bool SELF=false, class G, class K, class W>
inline auto leidenChooseCommunity(const G& x, K u, K d, const vector<W>& vtot, const vector<W>& ctot, const vector<K>& vcs, const vector<W>& vcout, double M, double R) {
  K cmax = K();
  W emax = W();
  for (K c : vcs) {
    if (!SELF && c==d) continue;
    W e = deltaModularity(vcout[c], vcout[d], vtot[u], ctot[c], ctot[d], M, R);
    if (e>emax) { emax = e; cmax = c; }
  }
  return make_pair(cmax, emax);
}


/**
 * Move vertex to another community C.
 * @param vcom community each vertex belongs to (updated)
 * @param ctot total edge weight of each community (updated)
 * @param x original graph
 * @param u given vertex
 * @param d previous community of vertex u
 * @param c community to move to
 * @param vtot total edge weight of each vertex
 */
template <bool REFINE=false, class G, class K, class W>
inline bool leidenChangeCommunityOmpW(vector<K>& vcom, vector<W>& ctot, const G& x, K u, K d, K c, const vector<W>& vtot) {
  if (REFINE) {
    bool ok = false;
    W ctotd = ctot[d];
    W ctotc = ctot[c];
    W btotd = ctotd - vtot[u];
    W btotc = ctotc + vtot[u];
    if (ctotd != vtot[u] || vcom[c] != c) return false;
    ok = __atomic_compare_exchange((int64_t*) &ctot[d], (int64_t*) &ctotd, (int64_t*) &btotd, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    if (!ok) return false;
    ok = __atomic_compare_exchange((int64_t*) &ctot[c], (int64_t*) &ctotc, (int64_t*) &btotc, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    if (!ok) {
      #pragma omp atomic
      ctot[d] += vtot[u];
      return false;
    }
    vcom[u] = c;
  }
  else {
    #pragma omp atomic
    ctot[d] -= vtot[u];
    #pragma omp atomic
    ctot[c] += vtot[u];
    vcom[u] = c;
  }
  return true;
}
#pragma endregion




#pragma region LOCAL-MOVING PHASE
/**
 * Leiden algorithm's local moving phase.
 * @param vcom community each vertex belongs to (initial, updated)
 * @param ctot total edge weight of each community (precalculated, updated)
 * @param vaff is vertex affected flag (updated)
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param vcob community bound each vertex belongs to
 * @param vtot total edge weight of each vertex
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @param L max iterations
 * @param fc has local moving phase converged?
 * @param fa is vertex allowed to be updated?
 * @param fb track communities that need to be broken
 * @returns iterations performed (0 if converged already)
 */
template <bool REFINE=false, class G, class K, class W, class B, class FC, class FA, class FB, class FZ>
inline int leidenMoveOmpW(vector<K>& vcom, vector<W>& ctot, vector<B>& vaff, vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, const G& x, const vector<K>& vcob, const vector<W>& vtot, double M, double R, int L, FC fc, FA fa, FB fb, FZ fz) {
  size_t S = x.span();
  int l = 0;
  W  el = W();
  for (; l<L;) {
    el = W();
    #pragma omp parallel for schedule(dynamic, 2048) reduction(+:el)
    for (K u=0; u<S; ++u) {
      int t = omp_get_thread_num();
      K   d = vcom[u];
      if (!x.hasVertex(u)) continue;
      if (!fa(u)) continue;
      if (!REFINE && !vaff[u]) continue;
      if ( REFINE && (d!=u || ctot[d]>vtot[u])) continue;
      leidenClearScanW(*vcs[t], *vcout[t]);
      leidenScanCommunitiesW<false, REFINE>(*vcs[t], *vcout[t], x, u, vcom, vcob);
      auto [c, e] = leidenChooseCommunity(x, u, d, vtot, ctot, *vcs[t], *vcout[t], M, R);
      if (e && leidenChangeCommunityOmpW<REFINE>(vcom, ctot, x, u, d, c, vtot)) {
        if (!REFINE) fb(d);
        if (!REFINE) x.forEachEdgeKey(u, [&](auto v) { vaff[v] = B(1); fz(v); });
      }
      if (!REFINE) vaff[u] = B();
      el += e;  // l1-norm
    }
    if (REFINE || fc(el, l++)) break;
  }
  return l>1 || el? l : 0;
}


/**
 * Leiden algorithm's local moving phase.
 * @param vcom community each vertex belongs to (initial, updated)
 * @param ctot total edge weight of each community (precalculated, updated)
 * @param vaff is vertex affected flag (updated)
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param vcob community bound each vertex belongs to
 * @param vtot total edge weight of each vertex
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @param L max iterations
 * @param fc has local moving phase converged?
 * @param fa is vertex allowed to be updated?
 * @returns iterations performed (0 if converged already)
 */
template <bool REFINE=false, class G, class K, class W, class B, class FC, class FA>
inline int leidenMoveOmpW(vector<K>& vcom, vector<W>& ctot, vector<B>& vaff, vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, const G& x, const vector<K>& vcob, const vector<W>& vtot, double M, double R, int L, FC fc, FA fa) {
  auto fb = [](auto u) {};
  return leidenMoveOmpW<REFINE>(vcom, ctot, vaff, vcs, vcout, x, vcob, vtot, M, R, L, fc, fa, fb, fb);
}


/**
 * Leiden algorithm's local moving phase.
 * @param vcom community each vertex belongs to (initial, updated)
 * @param ctot total edge weight of each community (precalculated, updated)
 * @param vaff is vertex affected flag (updated)
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param vcob community bound each vertex belongs to
 * @param vtot total edge weight of each vertex
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param R resolution (0, 1]
 * @param L max iterations
 * @param fc has local moving phase converged?
 * @returns iterations performed (0 if converged already)
 */
template <bool REFINE=false, class G, class K, class W, class B, class FC>
inline int leidenMoveOmpW(vector<K>& vcom, vector<W>& ctot, vector<B>& vaff, vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, const G& x, const vector<K>& vcob, const vector<W>& vtot, double M, double R, int L, FC fc) {
  auto fa = [](auto u) { return true; };
  return leidenMoveOmpW<REFINE>(vcom, ctot, vaff, vcs, vcout, x, vcob, vtot, M, R, L, fc, fa);
}
#pragma endregion




#pragma region COMMUNITY PROPERTIES
/**
 * Examine if each community exists.
 * @param a does each community exist (updated)
 * @param x original graph
 * @param vcom community each vertex belongs to
 * @returns number of communities
 */
template <class G, class K, class A>
inline size_t leidenCommunityExistsOmpW(vector<A>& a, const G& x, const vector<K>& vcom) {
  size_t S = x.span();
  size_t C = 0;
  fillValueOmpU(a, A());
  #pragma omp parallel for schedule(static, 2048) reduction(+:C)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    K c = vcom[u];
    A m = A();
    #pragma omp atomic capture
    { m = a[c]; a[c] = A(1); }
    if (!m) ++C;
  }
  return C;
}


/**
 * Find the total degree of each community.
 * @param a total degree of each community (updated)
 * @param x original graph
 * @param vcom community each vertex belongs to
 */
template <class G, class K, class A>
inline void leidenCommunityTotalDegreeOmpW(vector<A>& a, const G& x, const vector<K>& vcom) {
  size_t S = x.span();
  fillValueOmpU(a, A());
  #pragma omp parallel for schedule(static, 2048)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    K c = vcom[u];
    #pragma omp atomic
    a[c] += x.degree(u);
  }
}


/**
 * Find the number of vertices in each community.
 * @param a number of vertices belonging to each community (updated)
 * @param x original graph
 * @param vcom community each vertex belongs to
 */
template <class G, class K, class A>
inline void leidenCountCommunityVerticesOmpW(vector<A>& a, const G& x, const vector<K>& vcom) {
  size_t S = x.span();
  fillValueOmpU(a, A());
  #pragma omp parallel for schedule(static, 2048)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    K c = vcom[u];
    #pragma omp atomic
    ++a[c];
  }
}


/**
 * Find the vertices in each community.
 * @param coff csr offsets for vertices belonging to each community (updated)
 * @param cdeg number of vertices in each community (updated)
 * @param cedg vertices belonging to each community (updated)
 * @param bufk buffer for exclusive scan of size |threads| (scratch)
 * @param x original graph
 * @param vcom community each vertex belongs to
 */
template <class G, class K>
inline void leidenCommunityVerticesOmpW(vector<K>& coff, vector<K>& cdeg, vector<K>& cedg, vector<K>& bufk, const G& x, const vector<K>& vcom) {
  size_t S = x.span();
  size_t C = coff.size() - 1;
  leidenCountCommunityVerticesOmpW(coff, x, vcom);
  coff[C] = exclusiveScanOmpW(coff.data(), bufk.data(), coff.data(), C);
  fillValueOmpU(cdeg, K());
  #pragma omp parallel for schedule(static, 2048)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    K c = vcom[u];
    csrAddEdgeOmpU(cdeg, cedg, coff, c, u);
  }
}
#pragma endregion




#pragma region LOOKUP COMMUNITIES
/**
 * Update community membership in a tree-like fashion (to handle aggregation).
 * @param a output community each vertex belongs to (updated)
 * @param vcom community each vertex belongs to (at this aggregation level)
 */
template <class K>
inline void leidenLookupCommunitiesOmpU(vector<K>& a, const vector<K>& vcom) {
  size_t S = a.size();
  #pragma omp parallel for schedule(static, 2048)
  for (size_t u=0; u<S; ++u)
    a[u] = vcom[a[u]];
}
#pragma endregion




#pragma region AGGREGATION PHASE
/**
 * Aggregate outgoing edges of each community.
 * @param ydeg degree of each community (updated)
 * @param yedg vertex ids of outgoing edges of each community (updated)
 * @param ywei weights of outgoing edges of each community (updated)
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param vcom community each vertex belongs to
 * @param coff offsets for vertices belonging to each community
 * @param cedg vertices belonging to each community
 * @param yoff offsets for vertices belonging to each community
 */
template <int CHUNK_SIZE=2048, class G, class K, class W>
inline void leidenAggregateEdgesOmpW(vector<K>& ydeg, vector<K>& yedg, vector<W>& ywei, vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, const G& x, const vector<K>& vcom, const vector<K>& coff, const vector<K>& cedg, const vector<size_t>& yoff) {
  size_t C = coff.size() - 1;
  fillValueOmpU(ydeg, K());
  #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
  for (K c=0; c<C; ++c) {
    int t = omp_get_thread_num();
    K   n = csrDegree(coff, c);
    if (n==0) continue;
    leidenClearScanW(*vcs[t], *vcout[t]);
    csrForEachEdgeKey(coff, cedg, c, [&](auto u) {
      leidenScanCommunitiesW<true>(*vcs[t], *vcout[t], x, u, vcom);
    });
    for (auto d : *vcs[t])
      csrAddEdgeU(ydeg, yedg, ywei, yoff, c, d, (*vcout[t])[d]);
  }
}


/**
 * Re-number communities such that they are numbered 0, 1, 2, ...
 * @param vcom community each vertex belongs to (updated)
 * @param cext does each community exist (updated)
 * @param bufk buffer for exclusive scan of size |threads| (scratch)
 * @param x original graph
 * @returns number of communities
 */
template <class G, class K>
inline size_t leidenRenumberCommunitiesOmpW(vector<K>& vcom, vector<K>& cext, vector<K>& bufk, const G& x) {
  size_t C = exclusiveScanOmpW(cext, bufk, cext);
  leidenLookupCommunitiesOmpU(vcom, cext);
  return C;
}


/**
 * Leiden algorithm's community aggregation phase.
 * @param yoff offsets for vertices belonging to each community (updated)
 * @param ydeg degree of each community (updated)
 * @param yedg vertex ids of outgoing edges of each community (updated)
 * @param ywei weights of outgoing edges of each community (updated)
 * @param bufs buffer for exclusive scan of size |threads| (scratch)
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param x original graph
 * @param vcom community each vertex belongs to
 * @param coff offsets for vertices belonging to each community
 * @param cedg vertices belonging to each community
 */
template <int CHUNK_SIZE=2048, class G, class K, class W>
inline void leidenAggregateOmpW(vector<size_t>& yoff, vector<K>& ydeg, vector<K>& yedg, vector<W>& ywei, vector<size_t>& bufs, vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, const G& x, const vector<K>& vcom, vector<K>& coff, vector<K>& cedg) {
  size_t C = coff.size() - 1;
  leidenCommunityTotalDegreeOmpW(yoff, x, vcom);
  yoff[C] = exclusiveScanOmpW(yoff.data(), bufs.data(), yoff.data(), C);
  leidenAggregateEdgesOmpW<CHUNK_SIZE>(ydeg, yedg, ywei, vcs, vcout, x, vcom, coff, cedg, yoff);
}
#pragma endregion




#pragma region SPLIT DISCONNECTED COMMUNITIES
/**
 * Split disconnected communities using BFS.
 * @param vcom label/subcommunity each vertex belongs to (output)
 * @param cbsy community busy flags (scratch)
 * @param vis vertex visited flags (scratch)
 * @param us per-thread start vertices for BFS (scratch)
 * @param vs per-thread frontier vertices for BFS (scratch)
 * @param x given graph
 * @param vdom community each vertex belongs to
 * @param fs does community need to be split?
 */
template <class B, class G, class K, class FS>
inline void splitDisconnectedCommunitiesBfsOmpW(vector<K>& vcom, vector<B>& cbsy, vector<B>& vis, vector<vector<K>*>& us, vector<vector<K>*>& vs, const G& x, const vector<K>& vdom, FS fs) {
  size_t S = x.span();
  // Initialize each vertex to its own label/subcommunity.
  #pragma omp parallel for schedule(auto)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    cbsy[u] = B();
    if (fs(vdom[u])) {
      vcom[u] = u;
      vis[u]  = B();
    }
    else {
      vcom[u] = vdom[u];
      vis[u]  = B(1);
    }
  }
  // Perform DFS from an untouched vertex, within each community (each thread picks a community atomically).
  #pragma omp parallel
  {
    int t = omp_get_thread_num();
    int T = omp_get_num_threads();
    auto fl = [&](auto u) {
      if (!x.hasVertex(u) || vis[u]) return;
      K d = vdom[u], c = vcom[u];
      if (cbsy[d] || __atomic_exchange_n(&cbsy[d], B(1), __ATOMIC_SEQ_CST)!=B()) return;
      auto ft = [&](auto v, auto _) { return vdom[v]==d; };
      auto fp = [&](auto v, auto _) { vcom[v] = c; };
      (*us[t]).clear(); (*vs[t]).clear(); (*us[t]).push_back(u);
      bfsVisitedForEachU(vis, *us[t], *vs[t], x, ft, fp);
      cbsy[d] = B();
    };
    K ub = K((double(t)/T) * S);
    for (K u=ub; u<S; ++u) fl(u);
    for (K u=0; u<ub; ++u) fl(u);
  }
}
#pragma endregion




#pragma region TRACK COMMUNITIES
/**
 * Re-number communities such that they are tracked by their old IDs.
 * @param vcom community each vertex belongs to (updated)
 * @param cdid community ID each community is assigned to (updated)
 * @param cdwt total weight associated with community IDs (updated)
 * @param dcid community ID each old community is assigned to (updated)
 * @param dcwt total weight associated with old community IDs (updated)
 * @param x original graph
 * @param vdom community each vertex belonged to
 * @param vtot total weight of each vertex
 */
template <class G, class K, class W>
inline void leidenTrackCommunitiesOmpU(vector<K>& vcom, vector<K>& cdid, vector<W>& cdwt, vector<K>& dcid, vector<W>& dcwt, const G& x, const vector<K>& vdom, const vector<W>& vtot) {
  size_t S = x.span();
  // Clean up buffers.
  #pragma omp parallel for schedule(auto)
  for (K c=0; c<S; ++c) {
    cdid[c] = K();
    cdwt[c] = W();
    dcid[c] = K();
    dcwt[c] = W();
  }
  // For each old community, find the most overlapping new community,
  // using Boyer–Moore majority vote algorithm.
  #pragma omp parallel for schedule(auto)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    K d = vdom[u];
    K c = vcom[u];
    W w = vtot[u];
    if (dcid[d] == c+1) dcwt[d] += w;
    else if (dcwt[d] > w) dcwt[d] -= w;
    else {
      dcid[d] = c+1;
      dcwt[d] = w;
    }
  }
  // Now find the best old community for each new community, by flipping.
  #pragma omp parallel for schedule(auto)
  for (K d=0; d<S; ++d) {
    if (!dcid[d]) continue;
    K c = dcid[d] - 1;
    W w = dcwt[d];
    if (cdwt[c] > w) continue;
    cdid[c] = d+1;
    cdwt[c] = w;
  }
  // Mark all new communities.
  #pragma omp parallel for schedule(auto)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    K c = vcom[u];
    if (!cdwt[c]) cdwt[c] = W(1);
  }
  // Assign a free community ID to unassigned communities.
  #pragma omp parallel
  {
    int T = omp_get_num_threads();
    int t = omp_get_thread_num();
    K d   = (S / T) * t;
    #pragma omp for schedule(auto)
    for (K c=0; c<S; ++c) {
      if (!cdwt[c]) continue;
      if ( cdid[c]) continue;
      while (1) {
        while (dcid[d++])
          if (d>=S) d = K();
        if (__atomic_exchange_n(&dcid[d], c+1, __ATOMIC_SEQ_CST)==K()) break;
      }
      cdid[c] = d+1;
    }
  }
  // Now assign the new community IDs.
  #pragma omp parallel for schedule(auto)
  for (K u=0; u<S; ++u) {
    if (!x.hasVertex(u)) continue;
    K c = vcom[u];
    vcom[u] = cdid[c] - 1;
  }
}
#pragma endregion




#pragma region ENVIRONMENT SETUP
/**
 * Setup and perform the Leiden algorithm.
 * @param x original graph
 * @param o leiden options
 * @param fi initializing community membership and total vertex/community weights (vcom, vtot, ctot)
 * @param fm marking affected vertices (vaff, vcs, vcout, vcom, vtot, ctot)
 * @param fa is vertex allowed to be updated? (u)
 * @returns leiden result
 */
template <bool DYNAMIC=false, bool SELSPLIT=false, int CHUNK_SIZE=2048, class G, class FI, class FM, class FA>
inline auto leidenInvokeOmp(const G& x, const LeidenOptions& o, FI fi, FM fm, FA fa) {
  using  K = typename G::key_type;
  using  W = LEIDEN_WEIGHT_TYPE;
  using  B = char;
  // Options.
  double R = o.resolution;
  int    L = o.maxIterations, l = 0;
  int    P = o.maxPasses, p = 0;
  // Get graph properties.
  size_t X = x.size();
  size_t S = x.span();
  double M = edgeWeightOmp(x)/2;
  // Allocate buffers.
  int    T = omp_get_max_threads();
  vector<B> vaff(S), _vaff(S);  // Affected vertex flag (any pass)
  vector<B> cchg, cspt(S);  // Community changed/split flag (first pass)
  vector<B> bufb;           // Buffer for splitting communities
  vector<K> bufc;           // Buffer for obtaining a vertex from each community
  vector<K> ucom, vcom(S);  // Community membership (first pass, current pass)
  vector<K> udom, vcob(S);  // Old community membership (first pass), Community bound (any pass)
  vector<W> utot, vtot(S);  // Total vertex weights (first pass, current pass)
  vector<W> ctot, dtot;     // Total community weights (any pass)
  vector<W> cdwt;           // Change in total weight of each community
  vector<K> bufk(T);        // Buffer for exclusive scan
  vector<size_t> bufs(T);   // Buffer for exclusive scan
  vector<vector<K>*> vcs(T);    // Hashtable keys
  vector<vector<W>*> vcout(T);  // Hashtable values
  vector<vector<K>*> us(T), vs(T);  // BFS scratch space
  for (int t=0; t<T; ++t) {
    us[t] = new vector<K>();
    vs[t] = new vector<K>();
    us[t]->reserve(S);
    vs[t]->reserve(S);
  }
  if (!DYNAMIC) ucom.resize(S);
  if (!DYNAMIC) utot.resize(S);
  if (!DYNAMIC) ctot.resize(S);
  if (!DYNAMIC) cdwt.resize(S);
  if ( DYNAMIC) udom.resize(S);
  if ( DYNAMIC) cchg.resize(S);
  if ( DYNAMIC) bufb.resize(S);
  if ( DYNAMIC) bufc.resize(S);
  if ( DYNAMIC) dtot.resize(S);
  leidenAllocateHashtablesW(vcs, vcout, S);
  size_t _VAFF = 0, _CCHG = 0, _CSPT = 0, _UCOM = 0;
  size_t Z = max(size_t(o.aggregationTolerance * X), X);
  size_t Y = max(size_t(o.aggregationTolerance * Z), Z);
  DiGraphCsr<K, None, None, K> cv(S, S);  // CSR for community vertices
  DiGraphCsr<K, None, W> y(S, Y);         // CSR for aggregated graph (input);  y(S, X)
  DiGraphCsr<K, None, W> z(S, Z);         // CSR for aggregated graph (output); z(S, X)
  // Perform Leiden algorithm.
  float tm = 0, ti = 0, tp = 0, tl = 0, ts = 0, tr = 0, ta = 0, tt = 0;  // Time spent in different phases
  float t  = measureDurationMarked([&](auto mark) {
    double E  = o.tolerance;
    auto   fc = [&](double el, int l) { return el<=E; };
    // Reset buffers, in case of multiple runs.
    fillValueOmpU(vaff, B());
    fillValueOmpU(_vaff, B());
    fillValueOmpU(cchg, B());
    fillValueOmpU(cspt, B());
    fillValueOmpU(ucom, K());
    fillValueOmpU(vcom, K());
    fillValueOmpU(udom, K());
    fillValueOmpU(vcob, K());
    fillValueOmpU(utot, W());
    fillValueOmpU(vtot, W());
    fillValueOmpU(ctot, W());
    fillValueOmpU(cdwt, W());
    cv.respan(S);
    y .respan(S);
    z .respan(S);
    // Time the algorithm.
    mark([&]() {
      size_t CCHG = 0;
      // Initialize community membership and total vertex/community weights.
      ti += measureDuration([&]() {
        fi(ucom, utot, ctot, cdwt);
        if (DYNAMIC) copyValuesOmpW(udom, ucom);
      });
      // Mark affected vertices.
      tm += measureDuration([&]() {
        CCHG = fm(vaff, cchg, cspt, cdwt, vcs, vcout, ucom, utot, ctot);
        copyValuesOmpW(_vaff, vaff);
      });
      // Start timing first pass.
      auto t0 = timeNow(), t1 = t0;
      // Start local-moving, refinement, aggregation phases.
      // NOTE: In first pass, the input graph is a DiGraph.
      // NOTE: For subsequent passes, the input graph is a DiGraphCsr (optimization).
      for (l=0, p=0; M>0 && P>0;) {
        if (p==1) t1 = timeNow();
        bool isFirst = p==0;
        int m = 0;
        if (isFirst) _UCOM = communities(x, ucom).size();
        tl += measureDuration([&]() {
          auto fz = [&](auto u) { _vaff[u] = B(1); };
          auto fb = [&](auto c) { if (SELSPLIT) cspt[c] = B(1); };
          if (isFirst) m += leidenMoveOmpW(ucom, ctot, vaff, vcs, vcout, x, vcob, utot, M, R, L, fc, fa, fb, fz);
          else         m += leidenMoveOmpW(vcom, ctot, vaff, vcs, vcout, y, vcob, vtot, M, R, L, fc);
        });
        if (isFirst) _VAFF = countValueOmp(_vaff, B(1));
        if (isFirst) _CCHG = countValueOmp(cchg, B(1));
        if (isFirst) _CSPT = countValueOmp(cspt, B(1));
        size_t CSPT = SELSPLIT && isFirst? countValueOmp(cspt, B(1)) : 0;
        // Adjust community IDs.
        if (DYNAMIC && isFirst && (CSPT || CCHG)) {
          swap(ctot, dtot); swap(ucom, vcob); swap(cchg, vaff); swap(cspt, bufb);
          leidenSubsetRenameCommunitiesOmpW<SELSPLIT>(ucom, ctot, cchg, cspt, bufc, x, vcob, dtot, vaff, bufb);
        }
        ts += measureDuration([&]() {
          if (DYNAMIC && isFirst && (!SELSPLIT || CSPT)) {
            auto fs = [&](auto c) { return (!SELSPLIT || cspt[c]) && !cchg[c]; };
            splitDisconnectedCommunitiesBfsOmpW(vcom, bufb, vaff, us, vs, x, ucom, fs);
            swap(ucom, vcom);
          }
        });
        tr += measureDuration([&]() {
          if (!isFirst || !DYNAMIC || CCHG) {
            auto fr = [&](auto u) { return DYNAMIC? cchg[vcob[u]] : B(1); };
            if (isFirst) copyValuesOmpW(vcob.data(), ucom.data(), x.span());  // swap(vcob, ucom);
            else         copyValuesOmpW(vcob.data(), vcom.data(), y.span());  // swap(vcob, vcom);
            if (isFirst) leidenInitializeOmpW(ucom, ctot, x, utot, fr);
            else         leidenInitializeOmpW(vcom, ctot, y, vtot);
            // if (isFirst) fillValueOmpU(vaff.data(), x.order(), B(1));
            // else         fillValueOmpU(vaff.data(), y.order(), B(1));
            if (isFirst) m += leidenMoveOmpW<true>(ucom, ctot, vaff, vcs, vcout, x, vcob, utot, M, R, L, fc, fr);
            else         m += leidenMoveOmpW<true>(vcom, ctot, vaff, vcs, vcout, y, vcob, vtot, M, R, L, fc);
          }
        });
        l += max(m, 1); ++p;
        if ((m<=1 || p>=P) && (!isFirst || !CCHG)) break;
        size_t GN = isFirst? x.order() : y.order();
        size_t CN = 0;
        if (isFirst) CN = leidenCommunityExistsOmpW(cv.degrees, x, ucom);
        else         CN = leidenCommunityExistsOmpW(cv.degrees, y, vcom);
        if (double(CN)/GN >= o.aggregationTolerance) break;
        if (isFirst) leidenRenumberCommunitiesOmpW(ucom, cv.degrees, bufk, x);
        else         leidenRenumberCommunitiesOmpW(vcom, cv.degrees, bufk, y);
        if (isFirst) {}
        else         leidenLookupCommunitiesOmpU(ucom, vcom);
        ta += measureDuration([&]() {
          cv.respan(CN); z.respan(CN);
          if (isFirst) leidenCommunityVerticesOmpW(cv.offsets, cv.degrees, cv.edgeKeys, bufk, x, ucom);
          else         leidenCommunityVerticesOmpW(cv.offsets, cv.degrees, cv.edgeKeys, bufk, y, vcom);
          if (isFirst) leidenAggregateOmpW<CHUNK_SIZE>(z.offsets, z.degrees, z.edgeKeys, z.edgeValues, bufs, vcs, vcout, x, ucom, cv.offsets, cv.edgeKeys);
          else         leidenAggregateOmpW<CHUNK_SIZE>(z.offsets, z.degrees, z.edgeKeys, z.edgeValues, bufs, vcs, vcout, y, vcom, cv.offsets, cv.edgeKeys);
        });
        swap(y, z);
        // fillValueOmpU(vcob.data(), CN, K());
        // fillValueOmpU(vcom.data(), CN, K());
        // fillValueOmpU(ctot.data(), CN, W());
        fillValueOmpU(vtot.data(), CN, W());
        fillValueOmpU(vaff.data(), CN, B(1));
        leidenVertexWeightsOmpW(vtot, y);
        leidenInitializeOmpW(vcom, ctot, y, vtot);
        E /= o.toleranceDrop;
      }
      if (p<=1) {}
      else      leidenLookupCommunitiesOmpU(ucom, vcom);
      if (p<=1) t1 = timeNow();
      tp += duration(t0, t1);
      tt += measureDuration([&]() {
        if (DYNAMIC) leidenTrackCommunitiesOmpU(ucom, bufc, vtot, vcob, dtot, x, udom, utot);
      });
    });
  }, o.repeat);
  leidenFreeHashtablesW(vcs, vcout);
  for (int t=0; t<T; ++t) {
    delete us[t];
    delete vs[t];
  }
  return LeidenResult<K>(ucom, utot, ctot, cdwt, l, p, t, tm/o.repeat, ti/o.repeat, tp/o.repeat, tl/o.repeat, ts/o.repeat, tr/o.repeat, ta/o.repeat, tt/o.repeat, _VAFF, _CCHG, _CSPT, _UCOM);
}
#pragma endregion




#pragma region REPEAT SETUP (DYNAMIC)
/**
 * Setup the Dynamic Leiden algorithm for multiple runs.
 * @param qs initial community membership for each run (updated)
 * @param qvtots initial total vertex weights for each run (updated)
 * @param qctots initial total community weights for each run (updated)
 * @param qcdwts initial change in total weight of each community for each run (updated)
 * @param q initial community membership
 * @param qvtot initial total vertex weights
 * @param qctot initial total community weights
 * @param qcdwt initial change in total weight of each community
 * @param repeat number of runs
 */
template <class K, class W>
inline void leidenSetupInitialsW(vector2d<K>& qs, vector2d<W>& qvtots, vector2d<W>& qctots, vector2d<W>& qcdwts, const vector<K>& q, const vector<W>& qvtot, const vector<W>& qctot, const vector<W>& qcdwt, int repeat) {
  qs    .resize(repeat);
  qvtots.resize(repeat);
  qctots.resize(repeat);
  qcdwts.resize(repeat);
  for (int r=0; r<repeat; ++r) {
    qs[r]     = q;
    qvtots[r] = qvtot;
    qctots[r] = qctot;
    qcdwts[r] = qcdwt;
  }
}
#pragma endregion




#pragma region STATIC APPROACH
/**
 * Obtain the community membership of each vertex with Static Leiden.
 * @param x original graph
 * @param o leiden options
 * @returns leiden result
 */
template <class G>
inline auto leidenStaticOmp(const G& x, const LeidenOptions& o={}) {
  using B = char;
  using W = LEIDEN_WEIGHT_TYPE;
  auto fi = [&](auto& vcom, auto& vtot, auto& ctot, auto& cdwt) {
    leidenVertexWeightsOmpW(vtot, x);
    leidenInitializeOmpW(vcom, ctot, x, vtot);
    fillValueOmpU(cdwt, W());
  };
  auto fm = [ ](auto& vaff, auto& cchg, auto& cspt, auto& cdwt, auto& vcs, auto& vcout, const auto& vcom, const auto& vtot, const auto& ctot) {
    fillValueOmpU(vaff, B(1));
    return size_t(1);
  };
  auto fa = [ ](auto u) { return true; };
  return leidenInvokeOmp(x, o, fi, fm, fa);
}
#pragma endregion




#pragma region NAIVE-DYNAMIC APPROACH
/**
 * Mark changed communities due to edge deletions and insertions.
 * @param cchg community changed flag (updated)
 * @param cspt community split flag (updated)
 * @param cdwt change in total edge weight of each community (updated)
 * @param y updated graph
 * @param deletions edge deletions for this batch update (undirected)
 * @param insertions edge insertions for this batch update (undirected)
 * @param vcom community each vertex belongs to
 * @param ctot total edge weight of each community
 * @param Q refinement tolerance
 * @returns number of communities marked as changed
 */
template <bool SELSPLIT=false, class B, class G, class K, class V, class W>
inline size_t leidenChangedCommunitiesOmpU(vector<B>& cchg, vector<B>& cspt, vector<W>& cdwt, const G& y, const vector<tuple<K, K, V>>& deletions, const vector<tuple<K, K, V>>& insertions, const vector<K>& vcom, const vector<W>& ctot, double Q) {
  size_t S = y.span();
  size_t D = deletions.size();
  size_t I = insertions.size();
  size_t CCHG = 0;
  #pragma omp parallel for schedule(auto) reduction(+:CCHG)
  for (size_t i=0; i<D; ++i) {
    K u = get<0>(deletions[i]);
    K v = get<1>(deletions[i]);
    if (vcom[u] != vcom[v]) continue;
    K c = vcom[u];
    if (SELSPLIT) cspt[c] = B(1);
    if (cdwt[c] / ctot[c] < Q) continue;
    cchg[c] = B(1);
    cdwt[c] = W();
    ++CCHG;
  }
  #pragma omp parallel for schedule(auto) reduction(+:CCHG)
  for (size_t i=0; i<I; ++i) {
    K u = get<0>(insertions[i]);
    K v = get<1>(insertions[i]);
    if (vcom[u] != vcom[v]) continue;
    K c = vcom[u];
    if (cdwt[c] / ctot[c] < Q) continue;
    cchg[c] = B(1);
    cdwt[c] = W();
    ++CCHG;
  }
  return CCHG;
}


/**
 * Obtain the community membership of each vertex with Naive-dynamic Leiden.
 * @param y updated graph
 * @param deletions edge deletions for this batch update (undirected)
 * @param insertions edge insertions for this batch update (undirected)
 * @param q initial community each vertex belongs to
 * @param qvtot initial total edge weight of each vertex
 * @param qctot initial total edge weight of each community
 * @param qcdwt initial change in total edge weight of each community
 * @param o leiden options
 * @returns leiden result
 */
template <bool SELSPLIT=false, class G, class K, class V, class W>
inline auto leidenNaiveDynamicOmp(const G& y, const vector<tuple<K, K, V>>& deletions, const vector<tuple<K, K, V>>& insertions, const vector<K>& q, const vector<W>& qvtot, const vector<W>& qctot, const vector<W>& qcdwt, const LeidenOptions& o={}) {
  using B = char;
  vector2d<K> qs;
  vector2d<W> qvtots, qctots, qcdwts;
  leidenSetupInitialsW(qs, qvtots, qctots, qcdwts, q, qvtot, qctot, qcdwt, o.repeat);
  int  r  = 0;
  auto fi = [&](auto& vcom, auto& vtot, auto& ctot, auto& cdwt)  {
    vcom = move(qs[r]);
    vtot = move(qvtots[r]);
    ctot = move(qctots[r]);
    cdwt = move(qcdwts[r]); ++r;
    leidenUpdateWeightsFromOmpU(vtot, ctot, cdwt, y, deletions, insertions, vcom);
  };
  auto fm = [&](auto& vaff, auto& cchg, auto& cspt, auto& cdwt, auto& vcs, auto& vcout, const auto& vcom, const auto& vtot, const auto& ctot) {
    fillValueOmpU(vaff, B(1));
    size_t CCHG = leidenChangedCommunitiesOmpU<SELSPLIT>(cchg, cspt, cdwt, y, deletions, insertions, vcom, ctot, o.refinementTolerance);
    return CCHG;
  };
  auto fa = [ ](auto u) { return true; };
  return leidenInvokeOmp<true, SELSPLIT>(y, o, fi, fm, fa);
}
#pragma endregion




#pragma region DYNAMIC DELTA-SCREENING
/**
 * Find the vertices which should be processed upon a batch of edge insertions and deletions.
 * @param vertices vertex affected flags (output)
 * @param neighbors neighbor affected flags (output)
 * @param communities community affected flags (output)
 * @param cchg community changed flags (updated)
 * @param cspt community split flags (updated)
 * @param cdwt change in total edge weight of each community (updated)
 * @param vcs communities vertex u is linked to (temporary buffer, updated)
 * @param vcout total edge weight from vertex u to community C (temporary buffer, updated)
 * @param y updated graph
 * @param deletions edge deletions for this batch update (undirected, sorted by source vertex id)
 * @param insertions edge insertions for this batch update (undirected, sorted by source vertex id)
 * @param vcom community each vertex belongs to
 * @param vtot total edge weight of each vertex
 * @param ctot total edge weight of each community
 * @param M total weight of "undirected" graph (1/2 of directed graph)
 * @param Q refinement tolerance
 * @param R resolution (0, 1]
 * @returns number of communities marked as changed
 */
template <bool SELSPLIT=false, class B, class G, class K, class V, class W>
inline size_t leidenAffectedVerticesDeltaScreeningOmpW(vector<B>& vertices, vector<B>& neighbors, vector<B>& communities, vector<B>& cchg, vector<B>& cspt, vector<W>& cdwt, vector<vector<K>*>& vcs, vector<vector<W>*>& vcout, const G& y, const vector<tuple<K, K, V>>& deletions, const vector<tuple<K, K, V>>& insertions, const vector<K>& vcom, const vector<W>& vtot, const vector<W>& ctot, double M, double Q, double R=1) {
  size_t S = y.span();
  size_t D = deletions.size();
  size_t I = insertions.size();
  size_t CCHG = 0;
  fillValueOmpU(vertices,    B());
  fillValueOmpU(neighbors,   B());
  fillValueOmpU(communities, B());
  #pragma omp parallel for schedule(auto) reduction(+:CCHG)
  for (size_t i=0; i<D; ++i) {
    K u = get<0>(deletions[i]);
    K v = get<1>(deletions[i]);
    if (vcom[u] != vcom[v]) continue;
    vertices[u]  = 1;
    neighbors[u] = 1;
    communities[vcom[v]] = 1;
    K c = vcom[u];
    if (SELSPLIT) cspt[c] = B(1);
    if (cdwt[c] / ctot[c] < Q) continue;
    cchg[c] = B(1);
    cdwt[c] = W();
    ++CCHG;
  }
  #pragma omp parallel reduction(+:CCHG)
  {
    int T = omp_get_num_threads();
    int t = omp_get_thread_num();
    K  u0 = I>0? get<0>(insertions[0]) : 0;
    for (size_t i=0, n=0; i<I;) {
      K u = get<0>(insertions[i]);
      if (u!=u0) { ++n; u0 = u; }
      if (n % T != t) { ++i; continue; }
      leidenClearScanW(*vcs[t], *vcout[t]);
      for (; i<I && get<0>(insertions[i])==u; ++i) {
        K v = get<1>(insertions[i]);
        V w = get<2>(insertions[i]);
        if (vcom[u] == vcom[v]) {
          K c = vcom[u];
          if (cdwt[c] / ctot[c] < Q) continue;
          cchg[c] = B(1);
          cdwt[c] = W();
          ++CCHG;
          continue;
        }
        leidenScanCommunityW(*vcs[t], *vcout[t], u, v, w, vcom);
      }
      auto [c, e] = leidenChooseCommunity(y, u, vcom[u], vtot, ctot, *vcs[t], *vcout[t], M, R);
      if (e<=0) continue;
      vertices[u]  = 1;
      neighbors[u] = 1;
      communities[c] = 1;
    }
  }
  #pragma omp parallel for schedule(auto)
  for (K u=0; u<S; ++u) {
    if (!y.hasVertex(u)) continue;
    if (neighbors[u]) y.forEachEdgeKey(u, [&](auto v) { vertices[v] = 1; });
    if (communities[vcom[u]]) vertices[u] = 1;
  }
  return CCHG;
}


/**
 * Obtain the community membership of each vertex with Dynamic Delta-screening Leiden.
 * @param y updated graph
 * @param deletions edge deletions in batch update
 * @param insertions edge insertions in batch update
 * @param q initial community each vertex belongs to
 * @param qvtot initial total edge weight of each vertex
 * @param qctot initial total edge weight of each community
 * @param qcdwt initial change in total edge weight of each community
 * @param o leiden options
 * @returns leiden result
 */
template <bool SELSPLIT=false, class G, class K, class V, class W>
inline auto leidenDynamicDeltaScreeningOmp(const G& y, const vector<tuple<K, K, V>>& deletions, const vector<tuple<K, K, V>>& insertions, const vector<K>& q, const vector<W>& qvtot, const vector<W>& qctot, const vector<W>& qcdwt, const LeidenOptions& o={}) {
  using  B = char;
  size_t S = y.span();
  double R = o.resolution;
  double Q = o.refinementTolerance;
  double M = edgeWeightOmp(y)/2;
  int    T = omp_get_max_threads();
  vector<B> vertices(S), neighbors(S), communities(S);
  vector2d<K> qs;
  vector2d<W> qvtots, qctots, qcdwts;
  leidenSetupInitialsW(qs, qvtots, qctots, qcdwts, q, qvtot, qctot, qcdwt, o.repeat);
  int  r  = 0;
  auto fi = [&](auto& vcom, auto& vtot, auto& ctot, auto& cdwt) {
    vcom = move(qs[r]);
    vtot = move(qvtots[r]);
    ctot = move(qctots[r]);
    cdwt = move(qcdwts[r]); ++r;
    leidenUpdateWeightsFromOmpU(vtot, ctot, cdwt, y, deletions, insertions, vcom);
  };
  auto fm = [&](auto& vaff, auto& cchg, auto& cspt, auto& cdwt, auto& vcs, auto& vcout, const auto& vcom, const auto& vtot, const auto& ctot) {
    size_t CCHG = leidenAffectedVerticesDeltaScreeningOmpW<SELSPLIT>(vertices, neighbors, communities, cchg, cspt, cdwt, vcs, vcout, y, deletions, insertions, vcom, vtot, ctot, M, Q, R);
    copyValuesOmpW(vaff, vertices);
    return CCHG;
  };
  auto fa = [&](auto u) { return vertices[u] == B(1); };
  return leidenInvokeOmp<true, SELSPLIT>(y, o, fi, fm, fa);
}
#pragma endregion




#pragma region DYNAMIC FRONTIER APPROACH
/**
 * Find the vertices which should be processed upon a batch of edge insertions and deletions.
 * @param vertices vertex affected flags (output)
 * @param cchg community changed flags (updated)
 * @param cspt community split flags (updated)
 * @param cdwt change in total edge weight of each community (updated)
 * @param y updated graph
 * @param deletions edge deletions for this batch update (undirected)
 * @param insertions edge insertions for this batch update (undirected)
 * @param vcom community each vertex belongs to
 * @param ctot total edge weight of each community
 * @param Q refinement tolerance
 * @returns number of communities marked as changed
 */
template <bool SELSPLIT=false, class B, class G, class K, class V, class W>
inline size_t leidenAffectedVerticesFrontierOmpW(vector<B>& vertices, vector<B>& cchg, vector<B>& cspt, vector<W>& cdwt, const G& y, const vector<tuple<K, K, V>>& deletions, const vector<tuple<K, K, V>>& insertions, const vector<K>& vcom, const vector<W>& ctot, double Q) {
  fillValueOmpU(vertices, B());
  size_t D = deletions.size();
  size_t I = insertions.size();
  size_t CCHG = 0;
  #pragma omp parallel for schedule(auto) reduction(+:CCHG)
  for (size_t i=0; i<D; ++i) {
    K u = get<0>(deletions[i]);
    K v = get<1>(deletions[i]);
    if (vcom[u] != vcom[v]) continue;
    vertices[u]  = 1;
    K c = vcom[u];
    if (SELSPLIT) cspt[c] = B(1);
    if (cdwt[c] / ctot[c] < Q) continue;
    cchg[c] = B(1);
    cdwt[c] = W();
    ++CCHG;
  }
  #pragma omp parallel for schedule(auto) reduction(+:CCHG)
  for (size_t i=0; i<I; ++i) {
    K u = get<0>(insertions[i]);
    K v = get<1>(insertions[i]);
    if (vcom[u] == vcom[v]) {
      K c = vcom[u];
      if (cdwt[c] / ctot[c] < Q) continue;
      cchg[c] = B(1);
      cdwt[c] = W();
      ++CCHG;
      continue;
    }
    vertices[u]  = 1;
  }
  return CCHG;
}


/**
 * Obtain the community membership of each vertex with Dynamic Frontier Leiden.
 * @param y updated graph
 * @param deletions edge deletions in batch update
 * @param insertions edge insertions in batch update
 * @param q initial community each vertex belongs to
 * @param qvtot initial total edge weight of each vertex
 * @param qctot initial total edge weight of each community
 * @param qcdwt initial change in total edge weight of each community
 * @param o leiden options
 * @returns leiden result
 */
template <bool SELSPLIT=false, class G, class K, class V, class W>
inline auto leidenDynamicFrontierOmp(const G& y, const vector<tuple<K, K, V>>& deletions, const vector<tuple<K, K, V>>& insertions, const vector<K>& q, const vector<W>& qvtot, const vector<W>& qctot, const vector<W>& qcdwt, const LeidenOptions& o={}) {
  vector2d<K> qs;
  vector2d<W> qvtots, qctots, qcdwts;
  leidenSetupInitialsW(qs, qvtots, qctots, qcdwts, q, qvtot, qctot, qcdwt, o.repeat);
  int  r  = 0;
  auto fi = [&](auto& vcom, auto& vtot, auto& ctot, auto& cdwt) {
    vcom = move(qs[r]);
    vtot = move(qvtots[r]);
    ctot = move(qctots[r]);
    cdwt = move(qcdwts[r]); ++r;
    leidenUpdateWeightsFromOmpU(vtot, ctot, cdwt, y, deletions, insertions, vcom);
  };
  auto fm = [&](auto& vaff, auto& cchg, auto& cspt, auto& cdwt, auto& vcs, auto& vcout, const auto& vcom, const auto& vtot, const auto& ctot) {
    size_t CCHG = leidenAffectedVerticesFrontierOmpW<SELSPLIT>(vaff, cchg, cspt, cdwt, y, deletions, insertions, vcom, ctot, o.refinementTolerance);
    return CCHG;
  };
  constexpr int CHUNK_SIZE = 32;
  auto fa = [ ](auto u) { return true; };
  return leidenInvokeOmp<true, SELSPLIT, CHUNK_SIZE>(y, o, fi, fm, fa);
}
#pragma endregion
#pragma endregion
