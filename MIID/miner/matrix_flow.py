from collections import deque

class MaxFlow:
    def __init__(self, n):
        self.n = n
        self.adj = [[] for _ in range(n)]
        self.to = []
        self.cap = []
        self.rev = []

    def add_edge(self, u, v, c):
        # forward
        self.to.append(v); self.cap.append(c); self.rev.append(len(self.adj[v]))
        self.adj[u].append(len(self.to)-1)
        # backward
        self.to.append(u); self.cap.append(0); self.rev.append(len(self.adj[u])-1)
        self.adj[v].append(len(self.to)-1)

    def maxflow(self, s, t):
        flow = 0
        INF = 10**18
        while True:
            parent_edge = [-1]*self.n
            q = deque([s])
            parent_edge[s] = -2
            bottleneck = [0]*self.n
            bottleneck[s] = INF
            while q and parent_edge[t] == -1:
                u = q.popleft()
                for ei in self.adj[u]:
                    v = self.to[ei]
                    if parent_edge[v] == -1 and self.cap[ei] > 0:
                        parent_edge[v] = ei
                        bottleneck[v] = min(bottleneck[u], self.cap[ei])
                        if v == t: break
                        q.append(v)
            if parent_edge[t] == -1:
                break
            aug = bottleneck[t]
            flow += aug
            v = t
            while v != s:
                ei = parent_edge[v]
                self.cap[ei] -= aug
                # reverse edge index is ei^1 (paired), but we stored explicit rev[]
                # Find reverse quickly:
                u = self.to[ei ^ 1]  # since we always add edges in pairs
                # increase reverse capacity
                self.cap[ei ^ 1] += aug
                v = u
        return flow
def trans_x(x):
    trans_x = [[0]*len(x) for _ in range(len(x[0]))]
    for i in range(len(x)):
        for j in range(len(x[i])):
            trans_x[j][i] = x[i][j]
    return trans_x

# def trans_x(x):
#     trans_x = [[0]*3 for _ in range(4)]
#     for o_level in range(4):
#         for p_level in range(3): 
#             trans_x[p_level][o_level] = x[o_level][p_level]
#     return trans_x

def solve_maxflow(Max, P, O):
    # nodes: s=0, rows 1..3, cols 4..7, t=8
    Max = trans_x(Max)
    
    s, t = 0, 8
    mf = MaxFlow(9)

    # s -> rows
    for i in range(3):
        mf.add_edge(s, 1+i, P[i])

    # rows -> cols, keep indices to read flows later
    rc_edge_idx = [[None]*4 for _ in range(3)]
    for i in range(3):
        for j in range(4):
            u = 1+i
            v = 4+j
            # record the index of the forward edge being added
            forward_index_before = len(mf.to)
            mf.add_edge(u, v, Max[i][j])
            rc_edge_idx[i][j] = forward_index_before  # index of forward edge

    # cols -> t
    for j in range(4):
        mf.add_edge(4+j, t, O[j])

    K_star = mf.maxflow(s, t)

    # Recover x_ij = original_cap - residual_cap on forward edges row->col
    X = [[0]*4 for _ in range(3)]
    for i in range(3):
        for j in range(4):
            ei = rc_edge_idx[i][j]
            # originally we added a forward edge with capacity Max[i][j]
            # After flow, residual cap is mf.cap[ei]
            X[i][j] = Max[i][j] - mf.cap[ei]
    X = trans_x(X)
    return X, K_star


def max_transport33(MaxU, O, P):
    """
    Maximize sum x_ij subject to:
      0 <= x_ij <= MaxU[i][j], sum_j x_ij <= P[i], sum_i x_ij <= O[j]
    Inputs:
      MaxU: 3x3 nonnegative capacities
      O: length-3 column caps
      P: length-3 row caps
    Returns: (x, flow_value)
      x: 3x3 optimal matrix
    """
    n_rows, n_cols = 3, 3
    # Node indexing: s=0, rows=1..3, cols=4..6, t=7
    S, T = 0, 7
    def ridx(i): return 1 + i           # row node
    def cidx(j): return 4 + j           # col node

    N = 8
    cap = [[0]*N for _ in range(N)]
    adj = [[] for _ in range(N)]
    def add(u,v,w):
        if v not in adj[u]: adj[u].append(v)
        if u not in adj[v]: adj[v].append(u)
        cap[u][v] += w  # allow multi-edges merge

    # s -> rows
    for i in range(n_rows):
        add(S, ridx(i), P[i])
    # rows -> cols (store to recover flows)
    for i in range(n_rows):
        for j in range(n_cols):
            add(ridx(i), cidx(j), MaxU[i][j])
    # cols -> t
    for j in range(n_cols):
        add(cidx(j), T, O[j])

    def bfs():
        parent = [-1]*N; parent[S] = S
        q = deque([S])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if parent[v] == -1 and cap[u][v] > 0:
                    parent[v] = u
                    if v == T:
                        # reconstruct bottleneck
                        f = float('inf'); x = T
                        while x != S:
                            f = min(f, cap[parent[x]][x])
                            x = parent[x]
                        # apply
                        x = T
                        while x != S:
                            u2 = parent[x]
                            cap[u2][x] -= f
                            cap[x][u2] += f
                            x = u2
                        return f, parent
                    q.append(v)
        return 0, parent

    flow = 0
    while True:
        pushed, _ = bfs()
        if pushed == 0: break
        flow += pushed

    # Recover x: original forward capacity minus residual on row->col arcs
    x = [[0]*n_cols for _ in range(n_rows)]
    for i in range(n_rows):
        u = ridx(i)
        for j in range(n_cols):
            v = cidx(j)
            used = MaxU[i][j] - cap[u][v]
            x[i][j] = used
    return x, flow

def max_transport44(MaxU, O, P):
    """
    Maximize sum x_ij subject to:
      0 <= x_ij <= MaxU[i][j], sum_j x_ij <= P[i], sum_i x_ij <= O[j]
    Inputs:
      MaxU: 4x4 nonnegative capacities
      O: length-4 column caps
      P: length-4 row caps
    Returns: (x, flow_value)
      x: 4x4 optimal matrix
    """
    n_rows, n_cols = 4, 4
    # Node indexing: s=0, rows=1..4, cols=5..8, t=9
    S, T = 0, 9
    def ridx(i): return 1 + i           # row node
    def cidx(j): return 5 + j           # col node

    N = 10
    cap = [[0]*N for _ in range(N)]
    adj = [[] for _ in range(N)]
    def add(u,v,w):
        if v not in adj[u]: adj[u].append(v)
        if u not in adj[v]: adj[v].append(u)
        cap[u][v] += w  # allow multi-edges merge

    # s -> rows
    for i in range(n_rows):
        add(S, ridx(i), P[i])
    # rows -> cols (store to recover flows)
    for i in range(n_rows):
        for j in range(n_cols):
            add(ridx(i), cidx(j), MaxU[i][j])
    # cols -> t
    for j in range(n_cols):
        add(cidx(j), T, O[j])

    def bfs():
        parent = [-1]*N; parent[S] = S
        q = deque([S])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if parent[v] == -1 and cap[u][v] > 0:
                    parent[v] = u
                    if v == T:
                        # reconstruct bottleneck
                        f = float('inf'); x = T
                        while x != S:
                            f = min(f, cap[parent[x]][x])
                            x = parent[x]
                        # apply
                        x = T
                        while x != S:
                            u2 = parent[x]
                            cap[u2][x] -= f
                            cap[x][u2] += f
                            x = u2
                        return f, parent
                    q.append(v)
        return 0, parent

    flow = 0
    while True:
        pushed, _ = bfs()
        if pushed == 0: break
        flow += pushed

    # Recover x: original forward capacity minus residual on row->col arcs
    x = [[0]*n_cols for _ in range(n_rows)]
    for i in range(n_rows):
        u = ridx(i)
        for j in range(n_cols):
            v = cidx(j)
            used = MaxU[i][j] - cap[u][v]
            x[i][j] = used
    return x, flow



####################
from heapq import heappush, heappop

class MCMF:
    class Edge:
        __slots__ = ("to","rev","cap","cost","orig")
        def __init__(self, to, rev, cap, cost):
            self.to = to
            self.rev = rev
            self.cap = cap
            self.cost = cost
            self.orig = cap

    def __init__(self, n):
        self.n = n
        self.g = [[] for _ in range(n)]

    def add_edge(self, u, v, cap, cost):
        fwd = MCMF.Edge(v, len(self.g[v]), cap, cost)
        rev = MCMF.Edge(u, len(self.g[u]), 0, -cost)
        self.g[u].append(fwd)
        self.g[v].append(rev)
        return (u, len(self.g[u]) - 1)  # handle to forward edge

    def min_cost_flow(self, s, t, need):
        n = self.n
        flow = 0
        cost = 0
        INF = 10**18
        # costs are non-negative; Dijkstra suffices
        while flow < need:
            dist = [INF] * n
            prev = [(-1, -1)] * n
            dist[s] = 0
            pq = [(0, s)]
            while pq:
                d, u = heappop(pq)
                if d != dist[u]: continue
                for ei, e in enumerate(self.g[u]):
                    if e.cap <= 0: continue
                    v = e.to
                    nd = d + e.cost
                    if nd < dist[v]:
                        dist[v] = nd
                        prev[v] = (u, ei)
                        heappush(pq, (nd, v))
            if dist[t] == INF:
                # cannot push the required amount
                break
            # augment 1 unit (all capacities are integer; we can also push more)
            add = need - flow
            # find bottleneck
            v = t
            while v != s:
                u, ei = prev[v]
                add = min(add, self.g[u][ei].cap)
                v = u
            # apply
            v = t
            while v != s:
                u, ei = prev[v]
                e = self.g[u][ei]
                e.cap -= add
                self.g[v][e.rev].cap += add
                v = u
            flow += add
            cost += add * dist[t]
        return flow, cost

def solve_integer_diverse(Max, O):
    """
    Max: 3x4 matrix of nonnegative integers (upper bounds)
    O  : length-4 list of nonnegative integers (column budgets)
    Returns: x (3x4 integer matrix), row_sums (len 3)
    """
    R, C = 3, 4
    Max = trans_x(Max)
    if len(Max) == 1:
        return trans_x([O]), trans_x([O])

    assert len(Max) == R and all(len(row) == C for row in Max)
    assert len(O) == C

    # Effective per-column supply (force full feasible use)
    t = [min(O[j], sum(Max[i][j] for i in range(R))) for j in range(C)]
    total = sum(t)
    # Row upper bounds
    row_cap = [sum(Max[i][j] for j in range(C)) for i in range(R)]
    # Integer target average
    avg = round(total / R)

    # Build graph
    # nodes: S | C0..C3 | R0..R2 | T
    S = 0
    C0 = 1
    R0 = C0 + C
    T  = R0 + R
    N  = T + 1
    g = MCMF(N)

    # S -> columns
    for j in range(C):
        g.add_edge(S, C0 + j, t[j], 0)

    # columns -> rows (store handles to extract x later)
    rc_handle = [[None]*C for _ in range(R)]
    for j in range(C):
        for i in range(R):
            rc_handle[i][j] = g.add_edge(C0 + j, R0 + i, Max[i][j], 0)

    # rows -> T as unit arcs with convex increasing cost around avg
    # Incremental cost for k-th unit on a row: c(k) = 2*k + 1 - 2*avg
    # Shift all costs to be non-negative (doesn't change argmin since total flow is fixed)
    shift = max(0, 2*avg - 1)
    for i in range(R):
        K = row_cap[i]
        for k in range(K):  # add K parallel unit-cap edges
            inc_cost = (2*k + 1 - 2*avg) + shift
            g.add_edge(R0 + i, T, 1, inc_cost)

    # Push exactly 'total' units
    pushed, _ = g.min_cost_flow(S, T, total)
    if pushed != total:
        raise RuntimeError("Infeasible: not enough capacity to realize all t_j")

    # Extract integer x from column->row edges
    x = [[0]*C for _ in range(R)]
    for i in range(R):
        for j in range(C):
            u, idx = rc_handle[i][j]
            e = g.g[u][idx]
            used = e.orig - e.cap
            x[i][j] = used

    row_sums = [sum(x[i][j] for j in range(C)) for i in range(R)]
    return trans_x(x), row_sums


from typing import List, Tuple

def maxflow_then_maxdisp_int(MaxU: List[List[int]], O: List[int]) -> Tuple[List[List[int]], int]:
    """
    Lexicographic objective with integer capacities:
      1) maximize sum_{i,j} x_ij
      2) among max-flow solutions, maximize sum_i (S_i - avg)^2,
         where S_i = sum_j x_ij and avg = (sum_i S_i)/rows

    Subject to:
      0 <= x_ij <= MaxU[i][j]  (integers)
      sum_i x_ij <= O[j]       (integers)

    Inputs:
      MaxU: matrix of size 1x4 or 3x4 with non-negative integers
      O:    list of 4 non-negative integers (column caps)

    Returns:
      x: integer matrix same shape as MaxU (optimal under the lexicographic objective)
      flow_value: integer total flow
    """
    # # ---- validate shapes and types ----
    # if not isinstance(MaxU, list) or len(MaxU) not in (1, 3) or any(len(r) != 4 for r in MaxU):
    #     raise ValueError("MaxU must be 1x4 or 3x4.")
    # if not isinstance(O, list) or len(O) != 4:
    #     raise ValueError("O must have length 4.")
    # if any((not isinstance(v, int)) or v < 0 for r in MaxU for v in r):
    #     raise ValueError("All MaxU entries must be non-negative integers.")
    # if any((not isinstance(v, int)) or v < 0 for v in O):
    #     raise ValueError("All O entries must be non-negative integers.")

    MaxU = trans_x(MaxU)
    rows, cols = len(MaxU), 4

    # ---- Step 1: compute per-column max flow (max-flow) ----
    col_from_rows = [sum(MaxU[i][j] for i in range(rows)) for j in range(cols)]
    F = [min(O[j], col_from_rows[j]) for j in range(cols)]  # integer by construction
    flow_value = sum(F)

    # ---- Precompute row absorbable capacity given F (for dispersion tie-break) ----
    # R[i] = sum_j min(MaxU[i][j], F[j])
    R = [sum(min(MaxU[i][j], F[j]) for j in range(cols)) for i in range(rows)]

    # Row order: most absorbable first (ties by row index for determinism)
    row_order = sorted(range(rows), key=lambda i: (-R[i], i))

    # ---- Step 2: allocate each column's F[j] concentrating mass by row_order ----
    x = [[0 for _ in range(cols)] for _ in range(rows)]

    for j in range(cols):
        f = F[j]
        if f == 0:
            continue
        for i in row_order:
            if f == 0:
                break
            give = min(f, MaxU[i][j] - x[i][j])
            if give > 0:
                x[i][j] += give
                f -= give
        # Since F[j] <= sum_i MaxU[i][j], f must be 0 now.

    # ---- sanity checks ----
    # bounds
    for i in range(rows):
        for j in range(cols):
            if not (0 <= x[i][j] <= MaxU[i][j]):
                raise AssertionError("x out of bounds at (%d,%d)" % (i, j))
    # column caps
    for j in range(cols):
        if sum(x[i][j] for i in range(rows)) > O[j]:
            raise AssertionError("column cap violated at j=%d" % j)
    # total flow
    if sum(sum(row) for row in x) != flow_value:
        raise AssertionError("flow value mismatch.")

    return trans_x(x), flow_value

# ---------- examples ----------
if __name__ == "__main__":
    # 3x4 example
    MaxU_3x4 = [
        [3, 2, 5, 4],
        [2, 8, 1, 3],
        [7, 1, 2, 6],
    ]
    O = [7, 6, 5, 8]
    x, val = maxflow_then_maxdisp_int(trans_x(MaxU_3x4), O)
    print("3x4 solution:")
    for row in x: print(row)
    print("flow_value:", val)

    # 1x4 example
    MaxU_1x4 = [[5, 0, 9, 2]]
    O2 = [3, 7, 10, 1]
    x2, val2 = maxflow_then_maxdisp_int(trans_x(MaxU_1x4), O2)
    print("\n1x4 solution:")
    for row in x2: print(row)
    print("flow_value:", val2)
