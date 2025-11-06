from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import random, math, time, heapq, collections, itertools
from typing import Dict, Any, List, Tuple

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# -------------------- Utilities --------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

SCENARIO: Dict[str, Any] = {}
ITEMS = [
    {"id": "water", "weight_kg": 1.0, "impact": 5},
    {"id": "food", "weight_kg": 2.0, "impact": 8},
    {"id": "med", "weight_kg": 0.5, "impact": 20}
]

# -------------------- Scenario generator --------------------
def generate_scenario(nw=3, nz=12, seed=42):
    """
    Generate warehouses, zones and vehicles around Chennai city,
    keeping all coordinates within a land-safe bounding box.
    """
    random.seed(int(seed))

    # ✅ Land-safe bounding box (no sea overlap)
    lat_min, lat_max = 12.88, 13.12   # moved slightly south
    lon_min, lon_max = 80.05, 80.25   # moved further west (smaller lon)
# inland, stops before sea

    warehouses = []
    for i in range(int(nw)):
        lat = random.uniform(lat_min + 0.02, lat_max - 0.02)
        lon = random.uniform(lon_min + 0.02, lon_max - 0.02)
        stock = {ic['id']: random.randint(50, 150) for ic in ITEMS}
        warehouses.append({
            "id": f"wh{i}",
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "stock": stock
        })

    zones = []
    for j in range(int(nz)):
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)
        demand = {ic['id']: random.randint(5, 40) for ic in ITEMS}
        priority = random.randint(1, 10)
        zones.append({
            "id": f"z{j}",
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "demand": demand,
            "priority": priority
        })

    vehicles = []
    vid = 0
    for wh in warehouses:
        for _ in range(2):
            vehicles.append({
                "id": f"v{vid}",
                "depot": wh['id'],
                "capacity_kg": 200.0
            })
            vid += 1

    SCENARIO.clear()
    SCENARIO.update({
        "warehouses": warehouses,
        "zones": zones,
        "vehicles": vehicles,
        "items": ITEMS
    })
    return SCENARIO

# -------------------- Assignment Algorithms --------------------
def assign_greedy(scn):
    wh_map = {w['id']: {"lat": w['lat'], "lon": w['lon'], "stock": dict(w['stock'])} for w in scn['warehouses']}
    detailed = []
    for z in sorted(scn['zones'], key=lambda x: -x.get('priority', 0)):
        for ic in scn['items']:
            iid = ic['id']
            need = int(z['demand'].get(iid, 0))
            if need <= 0:
                continue
            whs = sorted(scn['warehouses'], key=lambda w: haversine(w['lat'], w['lon'], z['lat'], z['lon']))
            for w in whs:
                if need <= 0:
                    break
                avail = wh_map[w['id']]['stock'].get(iid, 0)
                if avail <= 0:
                    continue
                use = min(need, avail)
                kg = round(use * ic['weight_kg'], 3)
                detailed.append({
                    "warehouse": w['id'],
                    "zone": z['id'],
                    "item": iid,
                    "count": int(use),
                    "kg": kg
                })
                wh_map[w['id']]['stock'][iid] -= int(use)
                need -= int(use)

    combined = {}
    for d in detailed:
        key = (d['warehouse'], d['zone'])
        combined[key] = combined.get(key, 0) + d['kg']
    assigns = [{"warehouse": k[0], "zone": k[1], "kg": int(round(v))} for k, v in combined.items()]
    return assigns, detailed
# Max-flow (Edmonds-Karp) assignment - returns assignments and detailed item allocation approx.
def assign_maxflow(scn):
    wh_ids=[w['id'] for w in scn['warehouses']]
    z_ids=[z['id'] for z in scn['zones']]
    SRC='src'; SNK='snk'
    # mapping to indices
    idx_map={SRC:0}; idx=1
    for w in wh_ids:
        idx_map['w_'+w]=idx; idx+=1
    for z in z_ids:
        idx_map['z_'+z]=idx; idx+=1
    idx_map[SNK]=idx; N=idx+1
    cap=[[0]*N for _ in range(N)]
    # source->warehouse
    for w in scn['warehouses']:
        total_kg = sum(w['stock'][ic['id']]*ic['weight_kg'] for ic in scn['items'])
        cap[idx_map[SRC]][idx_map['w_'+w['id']]] = int(round(total_kg))
    # zone->sink
    for z in scn['zones']:
        total_demand_kg = sum(z['demand'][ic['id']]*ic['weight_kg'] for ic in scn['items'])
        cap[idx_map['z_'+z['id']]][idx_map[SNK]] = int(round(total_demand_kg))
    # wh->zone
    for w in scn['warehouses']:
        for z in scn['zones']:
            cap[idx_map['w_'+w['id']]][idx_map['z_'+z['id']]] = int(1e6)
    # Edmonds-Karp
    def bfs(res):
        parent=[-1]*N
        dq=collections.deque([idx_map[SRC]]); parent[idx_map[SRC]]=-2
        while dq:
            u=dq.popleft()
            for v in range(N):
                if parent[v]==-1 and res[u][v]>0:
                    parent[v]=u
                    if v==idx_map[SNK]: return parent
                    dq.append(v)
        return parent
    res=[row[:] for row in cap]
    maxflow=0
    while True:
        parent=bfs(res)
        if parent[idx_map[SNK]]==-1: break
        bott=float('inf'); v=idx_map[SNK]
        while v!=idx_map[SRC]:
            u=parent[v]; bott=min(bott,res[u][v]); v=u
        v=idx_map[SNK]
        while v!=idx_map[SRC]:
            u=parent[v]; res[u][v]-=bott; res[v][u]+=bott; v=u
        maxflow+=bott
    assigns=[]
    for w in wh_ids:
        u=idx_map['w_'+w]
        for z in z_ids:
            v=idx_map['z_'+z]
            flow=res[v][u]  # reverse residual contains flow
            if flow>0:
                assigns.append({"warehouse":w,"zone":z,"kg":int(round(flow))})
    # approximate item-level distribution: greedy by impact*priority
    detailed=[]
    wh_stock_counts = {w['id']: dict(w['stock']) for w in scn['warehouses']}
    zone_demands_counts = {z['id']: dict(z['demand']) for z in scn['zones']}
    for a in assigns:
        wh=a['warehouse']; z=a['zone']; kg=a['kg']
        zobj = next(z0 for z0 in scn['zones'] if z0['id']==z)
        items_sorted = sorted(scn['items'], key=lambda ic: -ic['impact']*zobj['priority'])
        kg_left = kg
        for ic in items_sorted:
            iid=ic['id']; wkg=ic['weight_kg']
            want = zone_demands_counts[z].get(iid,0)
            if want<=0: continue
            avail = wh_stock_counts[wh].get(iid,0)
            if avail<=0: continue
            max_by_kg = int(kg_left // wkg)
            send = min(want, avail, max_by_kg)
            if send<=0: continue
            detailed.append({"warehouse":wh,"zone":z,"item":iid,"count":int(send),"kg":round(send*wkg,3)})
            wh_stock_counts[wh][iid]-=int(send)
            zone_demands_counts[z][iid]-=int(send)
            kg_left -= send*wkg
            if kg_left<=0: break
    comb={}
    for d in detailed:
        key=(d['warehouse'],d['zone']); comb[key]=comb.get(key,0)+d['kg']
    out = [{"warehouse":k[0],"zone":k[1],"kg":int(round(v))} for k,v in comb.items()]
    return out, detailed, int(maxflow)

# Min-Cost Flow (successive shortest augmenting path with potentials)
def assign_mincost(scn):
    # nodes: source, warehouses, zones, sink
    wh_ids=[w['id'] for w in scn['warehouses']]
    z_ids=[z['id'] for z in scn['zones']]
    SRC='src'; SNK='snk'
    idx_map={SRC:0}; idx=1
    for w in wh_ids:
        idx_map['w_'+w]=idx; idx+=1
    for z in z_ids:
        idx_map['z_'+z]=idx; idx+=1
    idx_map[SNK]=idx; N=idx+1

    # adjacency lists with capacities and costs for residual graph
    adj=[[] for _ in range(N)]
    def add_edge(u,v,c, cost):
        adj[u].append([v,c,cost,len(adj[v])])
        adj[v].append([u,0,-cost,len(adj[u])-1])

    # source -> warehouses
    for w in scn['warehouses']:
        total_kg = int(round(sum(w['stock'][ic['id']]*ic['weight_kg'] for ic in scn['items'])))
        if total_kg>0:
            add_edge(idx_map[SRC], idx_map['w_'+w['id']], total_kg, 0)
    # zones -> sink
    for z in scn['zones']:
        total_demand_kg = int(round(sum(z['demand'][ic['id']]*ic['weight_kg'] for ic in scn['items'])))
        if total_demand_kg>0:
            add_edge(idx_map['z_'+z['id']], idx_map[SNK], total_demand_kg, 0)
    # warehouse -> zone edges with cost proportional to distance and priority
    for w in scn['warehouses']:
        for z in scn['zones']:
            d = haversine(w['lat'],w['lon'],z['lat'],z['lon'])
            cost = int(round(d * (11 - z['priority']) * 100))  # scale to integer
            add_edge(idx_map['w_'+w['id']], idx_map['z_'+z['id']], int(1e6), cost)

    # successive shortest augmenting path with potentials
    flow=0; cost=0
    INF=10**12
    potential=[0]*N
    parent_edge=[None]*N
    while True:
        dist=[INF]*N; dist[idx_map[SRC]]=0
        visited=[False]*N
        pq=[(0,idx_map[SRC])]
        while pq:
            d,u = heapq.heappop(pq)
            if visited[u]: continue
            visited[u]=True
            for ei, e in enumerate(adj[u]):
                v, cap, wcost, rev = e
                if cap<=0: continue
                nd = d + wcost + potential[u] - potential[v]
                if nd < dist[v]:
                    dist[v]=nd; parent_edge[v]=(u,ei)
                    heapq.heappush(pq,(nd,v))
        if dist[idx_map[SNK]]==INF: break
        # update potentials
        for i in range(N):
            if dist[i]<INF:
                potential[i]+=dist[i]
        # augment bottleneck
        bott=INF; v=idx_map[SNK]
        while v!=idx_map[SRC]:
            u,ei = parent_edge[v]
            cap = adj[u][ei][1]
            bott = min(bott, cap); v=u
        if bott==INF or bott==0: break
        # apply
        v=idx_map[SNK]
        while v!=idx_map[SRC]:
            u,ei = parent_edge[v]
            e = adj[u][ei]
            e[1] -= bott
            rev = e[3]; adj[v][rev][1] += bott
            cost += e[2] * bott
            v=u
        flow += bott

    # read flows from wh->zone edges
    assigns=[]
    detailed=[]
    # reconstruct warehouse stock and zone demand counts to assign items approximately
    wh_stock_counts = {w['id']: dict(w['stock']) for w in scn['warehouses']}
    zone_demands_counts = {z['id']: dict(z['demand']) for z in scn['zones']}
    # iterate over wh->z edges
    for w in scn['warehouses']:
        u = idx_map['w_'+w['id']]
        for e in adj[u]:
            v, cap_left, cost_edge, rev = e
            # find corresponding zone id
            if v==idx_map[SRC] or v==idx_map[SNK]: continue
            flow_sent = adj[v][rev][1]
            if flow_sent>0:
                zone_id = None
                for z in scn['zones']:
                    if idx_map['z_'+z['id']]==v:
                        zone_id = z['id']; break
                if zone_id is None: continue
                assigns.append({"warehouse":w['id'],"zone":zone_id,"kg":int(round(flow_sent))})
                # distribute items roughly by priority*impact
                zobj = next(z0 for z0 in scn['zones'] if z0['id']==zone_id)
                items_sorted = sorted(scn['items'], key=lambda ic: -ic['impact']*zobj['priority'])
                kg_left = flow_sent
                for ic in items_sorted:
                    iid=ic['id']; wkg=ic['weight_kg']
                    want = zone_demands_counts[zone_id].get(iid,0)
                    if want<=0: continue
                    avail = wh_stock_counts[w['id']].get(iid,0)
                    if avail<=0: continue
                    max_by_kg = int(kg_left // wkg)
                    send = min(want, avail, max_by_kg)
                    if send<=0: continue
                    detailed.append({"warehouse":w['id'],"zone":zone_id,"item":iid,"count":int(send),"kg":round(send*wkg,3)})
                    wh_stock_counts[w['id']][iid]-=int(send)
                    zone_demands_counts[zone_id][iid]-=int(send)
                    kg_left -= send*wkg
                    if kg_left<=0: break
    return assigns, detailed, int(flow), int(cost)

# -------------------- Route planning (NN + 2-opt + 3-opt) --------------------
def nearest_neighbour(depot: Tuple[float,float], stops: List[Tuple[float,float]]):
    un = stops.copy(); curr = depot; tour=[depot]
    while un:
        nxt = min(un, key=lambda p: haversine(curr[0],curr[1],p[0],p[1]))
        tour.append(nxt); un.remove(nxt); curr=nxt
    tour.append(depot); return tour

def route_length(route: List[Tuple[float,float]]):
    if not route or len(route)<2: return 0.0
    s=0.0
    for i in range(len(route)-1):
        s+=haversine(route[i][0],route[i][1],route[i+1][0],route[i+1][1])
    return s

def two_opt(route: List[Tuple[float,float]], max_iter=500):
    best = route[:]; best_len = route_length(best)
    n = len(best)
    improved=True; it=0
    while improved and it < max_iter:
        improved=False; it+=1
        for i in range(1, n-2):
            for j in range(i+1, n-1):
                if j-i==1: continue
                new = best[:i] + best[i:j+1][::-1] + best[j+1:]
                nl = route_length(new)
                if nl + 1e-9 < best_len:
                    best, best_len = new, nl
                    improved=True
                    break
            if improved: break
    return best

# Basic 3-opt (limited, practical for small routes): try triple reversal patterns
def three_opt(route: List[Tuple[float,float]], max_iter=200):
    best = route[:]; best_len = route_length(best)
    n = len(best)
    it=0
    improved=True
    while improved and it < max_iter:
        improved=False; it+=1
        for i in range(1,n-4):
            for j in range(i+1, n-3):
                for k in range(j+1, n-1):
                    # generate a few 3-opt reconnections (simple ones)
                    A = best[:i] + best[i:j][::-1] + best[j:k][::-1] + best[k:]
                    B = best[:i] + best[j:k] + best[i:j] + best[k:]
                    for cand in (A,B):
                        nl = route_length(cand)
                        if nl + 1e-9 < best_len:
                            best, best_len = cand, nl
                            improved=True
                            break
                    if improved: break
                if improved: break
            if improved: break
    return best

# -------------------- Graph algorithms --------------------
def dijkstra_complete(points: List[Tuple[float,float]], src:int, dst:int):
    n=len(points)
    dist=[float('inf')]*n; prev=[-1]*n
    dist[src]=0.0
    pq=[(0.0,src)]
    while pq:
        d,u = heapq.heappop(pq)
        if d>dist[u]: continue
        if u==dst: break
        for v in range(n):
            if v==u: continue
            w = haversine(points[u][0],points[u][1],points[v][0],points[v][1])
            nd = d + w
            if nd < dist[v]:
                dist[v]=nd; prev[v]=u; heapq.heappush(pq,(nd,v))
    if dist[dst] == float('inf'): return [], float('inf')
    path=[]; cur=dst
    while cur!=-1:
        path.append(cur); cur=prev[cur]
    path.reverse(); return path, dist[dst]

def bellman_ford(points: List[Tuple[float,float]], src:int):
    n=len(points)
    dist=[float('inf')]*n; prev=[-1]*n
    dist[src]=0.0
    for _ in range(n-1):
        updated=False
        for u in range(n):
            if dist[u]==float('inf'): continue
            for v in range(n):
                if u==v: continue
                w = haversine(points[u][0],points[u][1],points[v][0],points[v][1])
                if dist[u]+w < dist[v]:
                    dist[v] = dist[u]+w; prev[v]=u; updated=True
        if not updated: break
    return dist, prev

def floyd_warshall(points: List[Tuple[float,float]]):
    n=len(points)
    dist=[[0.0 if i==j else haversine(points[i][0],points[i][1],points[j][0],points[j][1]) for j in range(n)] for i in range(n)]
    next_hop=[[j if i!=j else -1 for j in range(n)] for i in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k]+dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k]+dist[k][j]
                    next_hop[i][j] = next_hop[i][k]
    return dist, next_hop

def prim_mst(points: List[Tuple[float,float]]):
    n=len(points)
    if n<=1: return []
    in_mst=[False]*n; key=[float('inf')]*n; parent=[-1]*n
    key[0]=0.0
    for _ in range(n):
        u=min((i for i in range(n) if not in_mst[i]), key=lambda i:key[i])
        in_mst[u]=True
        for v in range(n):
            if in_mst[v] or v==u: continue
            w = haversine(points[u][0],points[u][1],points[v][0],points[v][1])
            if w < key[v]:
                key[v]=w; parent[v]=u
    edges=[]
    for v in range(1,n):
        if parent[v]!=-1:
            edges.append((parent[v], v, round(haversine(points[parent[v]][0],points[parent[v]][1],points[v][0],points[v][1]),3)))
    return edges

# -------------------- K-Means clustering for VRP --------------------
def kmeans(points: List[Tuple[float,float]], k:int=3, max_iter=50):
    # Defensive: always return (clusters, centers) pair
    if k<=0 or not points:
        return [], []
    # init: choose k random centers
    centers = random.sample(points, min(k, len(points)))
    for _ in range(max_iter):
        clusters=[[] for _ in centers]
        for p in points:
            best = min(range(len(centers)), key=lambda i: (p[0]-centers[i][0])**2 + (p[1]-centers[i][1])**2)
            clusters[best].append(p)
        new_centers=[]
        for cl in clusters:
            if not cl:
                new_centers.append(random.choice(points))
            else:
                lat = sum(x for x,y in cl)/len(cl)
                lon = sum(y for x,y in cl)/len(cl)
                new_centers.append((lat,lon))
        if new_centers == centers:
            break
        centers = new_centers
    return clusters, centers

# -------------------- Endpoints --------------------
@app.get("/generate")
def gen(nw:int=3, nz:int=12, seed:int=42):
    return generate_scenario(nw,nz,seed)

@app.get("/scenario")
def scen():
    return SCENARIO

@app.post("/schedule")
def schedule(
    mode: str = Query("Optimized", regex="^(Optimized|Naive|Compare)$"),
    assignment: str = Query("greedy", regex="^(greedy|maxflow|mincost)$"),
    routing_opt: str = Query("2opt", regex="^(2opt|3opt|none)$"),
    clustering: str = Query("none", regex="^(none|kmeans)$"),
    include_floyd: bool = False,
    include_bellman: bool = False
):
    scn = SCENARIO
    if not scn: return {"error":"Generate first"}

    # assignment selection
    t0 = time.time()
    assign_algo = assignment
    if assignment == "greedy":
        assigns, detailed = assign_greedy(scn)
        flow_assigned = None; mincost_cost=None
    elif assignment == "maxflow":
        assigns, detailed, flow_assigned = assign_maxflow(scn)
        mincost_cost=None
    else:
        assigns, detailed, flow_assigned, mincost_cost = assign_mincost(scn)
    assign_time = time.time() - t0

    # supply/demand/assigned/unmet metrics
    total_supply_kg = sum(sum(w['stock'][ic['id']] * ic['weight_kg'] for ic in scn['items']) for w in scn['warehouses'])
    total_demand_kg = sum(sum(z['demand'][ic['id']] * ic['weight_kg'] for ic in scn['items']) for z in scn['zones'])
    total_assigned_kg = sum(a['kg'] for a in assigns) if assigns else 0
    unmet_kg = max(0, total_demand_kg - total_assigned_kg)

    # group assigns by warehouse for vehicle planning
    zones_by_wh = {}
    for a in assigns:
        zones_by_wh.setdefault(a['warehouse'], []).append(a)

    AVG_SPEED_KMPH = 30.0
    kn_t = 0.0; rt_t = 0.0; total_km = 0.0

    def knapsack_simple(cands, cap):
        units=[]
        for c in cands:
            for _ in range(int(c.get("count",0))):
                units.append((int(round(c["weight_kg"]*100)), c["value"]))
        W=int(round(cap*100)); n=len(units)
        if n==0: return {"selected_count":0,"total_weight_kg":0.0,"total_value":0}
        dp=[0]*(W+1); take=[[False]*(W+1) for _ in range(n)]
        for i,(wt,val) in enumerate(units):
            for w in range(W,wt-1,-1):
                if dp[w-wt]+val>dp[w]:
                    dp[w]=dp[w-wt]+val; take[i][w]=True
        sel=[]; w=W
        for i in range(n-1,-1,-1):
            if take[i][w]:
                sel.append(units[i]); w-=units[i][0]
        wt=sum(u[0] for u in sel)/100.0; val=sum(u[1] for u in sel)
        return {"selected_count":len(sel),"total_weight_kg":round(wt,3),"total_value":int(val)}

    vehicles_out=[]
    # clustering: optionally cluster zones per warehouse using kmeans to show clusters
    kmeans_clusters = None; kmeans_centers = None
    if clustering == "kmeans":
        # cluster all zones into number of warehouses
        points = [(z['lat'], z['lon']) for z in scn['zones']]
        k = max(1, len(scn['warehouses']))
        clusters, centers = kmeans(points, k=k)
        kmeans_clusters = clusters; kmeans_centers = centers

    # build vehicle plans (optimize routing with NN + 2opt/3opt depending)
    for v in scn['vehicles']:
        t_kn = time.time()
        whid = v['depot']
        zones_assigned = zones_by_wh.get(whid, [])
        stops=[]; items=[]
        for a in zones_assigned:
            z = next((z0 for z0 in scn['zones'] if z0['id']==a['zone']), None)
            if not z: continue
            stops.append((z['lat'], z['lon']))
            for ic in scn['items']:
                cnt = z['demand'].get(ic['id'],0)
                if cnt>0: items.append({"item":ic['id'],"count":int(cnt),"weight_kg":ic['weight_kg'],"value":ic['impact']*z['priority']})
        kn = knapsack_simple(items, v['capacity_kg'])
        kn_t += time.time()-t_kn

        t_rt = time.time()
        wh_obj = next((w for w in scn['warehouses'] if w['id']==whid), None)
        if wh_obj and stops:
            base = nearest_neighbour((wh_obj['lat'], wh_obj['lon']), stops)
            if routing_opt == "2opt":
                final = two_opt(base)
            elif routing_opt == "3opt":
                final = three_opt(base)
            else:
                final = base
            rkm = round(route_length(final),3)
            travel_time_h = round(rkm / AVG_SPEED_KMPH,3)
        else:
            final=[]; rkm=0.0; travel_time_h=0.0
        rt_t += time.time()-t_rt
        total_km += rkm
        util = round((kn['total_weight_kg'] / v['capacity_kg']) * 100, 2) if v['capacity_kg']>0 else 0.0
        vehicles_out.append({
            "vehicle": v['id'],
            "depot": whid,
            "assigned_zones": [a['zone'] for a in zones_assigned],
            "knapsack": kn,
            "route": final,
            "route_len_km": rkm,
            "travel_time_h": travel_time_h,
            "utilization_pct": util
        })

    # if user asked Compare mode: run naive + optimized and compute deltas
    naive_km = None; opt_km = None
    if mode == "Compare":
        naive_total = 0.0
        for v in scn['vehicles']:
            whid=v['depot']; wh_obj = next((w for w in scn['warehouses'] if w['id']==whid), None)
            zones_assigned = zones_by_wh.get(whid, [])
            stops=[]
            if wh_obj:
                for a in zones_assigned:
                    z = next((z0 for z0 in scn['zones'] if z0['id']==a['zone']), None)
                    if z: stops.append((z['lat'], z['lon']))
                if stops:
                    base = nearest_neighbour((wh_obj['lat'],wh_obj['lon']), stops)
                    naive_total += route_length(base)
        naive_km = round(naive_total,3); opt_km = round(total_km,3)

    # cost calculations (distance * kg)
    def transport_cost(assigns_list):
        c=0.0
        for a in assigns_list:
            w = next(w0 for w0 in scn['warehouses'] if w0['id']==a['warehouse'])
            z = next(z0 for z0 in scn['zones'] if z0['id']==a['zone'])
            c += haversine(w['lat'],w['lon'],z['lat'],z['lon']) * a['kg']
        return c

    naive_cost = transport_cost(assigns) if assigns else 0.0
    optimized_cost = naive_cost  # placeholder unless we calculate improved cost using optimized routes & assignments

    improvement_pct = 0.0
    if naive_km and opt_km is not None and naive_km>0:
        improvement_pct = round((naive_km - opt_km)/naive_km*100,2)

    # build geojson
    features=[]
    for w in scn['warehouses']:
        features.append({"type":"Feature","properties":{"type":"warehouse","id":w['id']},"geometry":{"type":"Point","coordinates":[w['lon'],w['lat']]}})
    for z in scn['zones']:
        features.append({"type":"Feature","properties":{"type":"zone","id":z['id'],"priority":z['priority']},"geometry":{"type":"Point","coordinates":[z['lon'],z['lat']]}})

    # add routes
    for vp in vehicles_out:
        coords=[[p[1],p[0]] for p in vp['route']]
        props={"type":"route","vehicle":vp['vehicle'],"depot":vp['depot'],"knapsack":vp['knapsack'],"route_len_km":vp['route_len_km'],"travel_time_h":vp['travel_time_h'],"utilization_pct":vp['utilization_pct']}
        features.append({"type":"Feature","properties":props,"geometry":{"type":"LineString","coordinates":coords}})

    # include MST, sample shortest path, APSP, BF if requested
    mst_edges=[]; sp_coords=[]; sp_len=None; floyd_mat=None; bf_result=None
    try:
        sample_v = vehicles_out[0] if vehicles_out else None
        if sample_v and sample_v['route']:
            pts = [(sample_v['route'][i][0], sample_v['route'][i][1]) for i in range(len(sample_v['route']))]
            if pts:
                if len(pts)>1:
                    edges = prim_mst(pts)
                    mst_edges = [{"u":u,"v":v,"w":w} for (u,v,w) in edges]
                    for (u,v,w) in edges:
                        features.append({"type":"Feature","properties":{"type":"mst_edge","w":w},"geometry":{"type":"LineString","coordinates":[[pts[u][1],pts[u][0]],[pts[v][1],pts[v][0]]]}})
                    if sample_v['route']:
                        far_idx = max(range(1,len(pts)), key=lambda i: haversine(pts[0][0],pts[0][1],pts[i][0],pts[i][1])) if len(pts)>1 else 0
                        path_idx, sp = dijkstra_complete(pts, 0, far_idx)
                        sp_len = round(sp,3) if sp!=float('inf') else None
                        sp_coords = [[pts[i][1],pts[i][0]] for i in path_idx]
                        if sp_coords:
                            features.append({"type":"Feature","properties":{"type":"shortest_path","length_km":sp_len},"geometry":{"type":"LineString","coordinates":sp_coords}})
                    if include_floyd:
                        floyd_mat, next_hop = floyd_warshall(pts)
                    if include_bellman:
                        dist_bf, prev_bf = bellman_ford(pts, 0)
                        bf_result = {"distances": [round(d,3) if d!=float('inf') else None for d in dist_bf], "prev": prev_bf}
    except Exception:
        pass

    metrics = {
        "total_supply_kg": round(total_supply_kg,3),
        "total_demand_kg": round(total_demand_kg,3),
        "total_assigned_kg": int(total_assigned_kg),
        "unmet_kg": round(unmet_kg,3),
        "assign_time_s": round(assign_time,6),
        "knapsack_time_s": round(kn_t,6),
        "route_time_s": round(rt_t,6),
        "naive_cost": round(naive_cost,4),
        "optimized_cost": round(optimized_cost,4),
        "improvement_pct": improvement_pct
    }

    result = {
        "assignments": assigns,
        "detailed_assignments": detailed,
        "vehicles": vehicles_out,
        "geojson": {"type":"FeatureCollection","features":features},
        "metrics": metrics,
        "algorithms": {
            "assignment_algo": assign_algo,
            "flow_assigned_kg": flow_assigned if 'flow_assigned' in locals() else None,
            "mincost_cost": mincost_cost if 'mincost_cost' in locals() else None,
            "routing_opt": routing_opt,
            "clustering": clustering
        },
        "extra": {
            "mst": mst_edges,
            "shortest_path_km": sp_len,
            "floyd": floyd_mat,
            "bellman": bf_result,
            "kmeans_clusters": kmeans_clusters,
            "kmeans_centers": kmeans_centers
        },
        "mode": mode
    }
    if vehicles_out:
        result["sample_knapsack_vehicle"] = vehicles_out[0]
    if mode=="Compare":
        result["totals"] = {"naive_total_km": naive_km, "optimized_total_km": opt_km, "route_improvement_pct": improvement_pct}
    return result

# -------------------- Stress endpoint --------------------
@app.post("/stress")
def stress_test(n: int = Query(10, gt=0), nw:int=3, nz:int=12, seed:int=1000, assignment:str="greedy", routing_opt:str="2opt", clustering:str="none"):
    stats = {"runs":0,"success":0,"avg_improvement_pct":0.0,"avg_assign_time_s":0.0,"avg_total_assigned_kg":0.0}
    imp_list=[]; assign_time_list=[]; assigned_list=[]
    for i in range(n):
        s = seed + i
        generate_scenario(nw, nz, s)
        resp = schedule(mode="Compare", assignment=assignment, routing_opt=routing_opt, clustering=clustering)
        stats['runs'] += 1
        try:
            imp = float(resp.get("totals",{}).get("route_improvement_pct",0.0) or 0.0)
            at = float(resp.get("metrics",{}).get("assign_time_s",0.0) or 0.0)
            assigned = int(resp.get("metrics",{}).get("total_assigned_kg",0) or 0)
            imp_list.append(imp); assign_time_list.append(at); assigned_list.append(assigned)
            stats['success'] += 1
        except Exception:
            pass
    if stats['success']>0:
        stats['avg_improvement_pct'] = round(sum(imp_list)/len(imp_list),4)
        stats['avg_assign_time_s'] = round(sum(assign_time_list)/len(assign_time_list),6)
        stats['avg_total_assigned_kg'] = round(sum(assigned_list)/len(assigned_list),2)
    return stats

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
