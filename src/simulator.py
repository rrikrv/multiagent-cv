# ============================================================
#  simulator.py — A* navigation + conflict resolution
#
#  Pathfinding: A* with Manhattan heuristic
#  Conflict resolution:
#    - Cell reservation by agent priority (lower index = higher priority)
#    - Passive agents yield to active ones
#    - Swap conflict: lower-priority agent waits or reroutes
#    - Period-2 detector: if an agent loops A→B→A→B,
#      force reroute through a neighbouring row
# ============================================================

import json, csv, os, heapq
from collections import deque


def xy_to_rowcol(coord):
    """
    Scenario files define positions as [col, row] (x, y order).
    Graph nodes are indexed as 'row_col'.
    This function converts [col, row] -> (row, col).
    Without this, agents start and finish at mirrored positions.
    """
    return (coord[1], coord[0])


def load_scenario(path):
    """Loads scenario.json — agent starting positions, goals, scenario type."""
    with open(path) as f:
        return json.load(f)

def load_tasks(path):
    """Loads tasks.json — pickup/dropoff task list for MAPD scenarios."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)

def load_graph(edges_path):
    """
    Builds a navigation graph from edges.csv.
    Returns a dict: node_id -> list of (neighbour_id, cost).
    Graph is undirected — each edge is added in both directions.
    """
    graph = {}
    with open(edges_path) as f:
        for row in csv.DictReader(f):
            a, b, cost = row["from"], row["to"], int(row["cost"])
            graph.setdefault(a, []).append((b, cost))
            graph.setdefault(b, []).append((a, cost))
    return graph

def nearest_node(graph, position):
    """Finds the closest graph node to a given (row, col) position."""
    row, col = position
    best_key, best_dist = None, float("inf")
    for key in graph:
        r, c = map(int, key.split("_"))
        d = abs(r - row) + abs(c - col)
        if d < best_dist:
            best_dist, best_key = d, key
    return best_key

def snap_coord(graph, coord):
    """
    Snaps a coordinate to the nearest existing graph node.
    Handles minor mismatches between scenario coordinates and graph nodes.
    """
    key = f"{coord[0]}_{coord[1]}"
    if key in graph:
        return tuple(coord)
    nn = nearest_node(graph, coord)
    return tuple(map(int, nn.split("_")))

def heuristic(a, b):
    """Manhattan distance heuristic for A* on a four-connected grid."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_path(graph, start, goal, blocked=None):
    """
    A* pathfinding from start to goal on the navigation graph.
    blocked: set of (row, col) positions to avoid (e.g. other agents).
    Returns a list of (row, col) positions from start to goal,
    or empty list if no path exists.
    """
    blocked = blocked or set()
    start   = snap_coord(graph, start)
    goal    = snap_coord(graph, goal)
    if start == goal:
        return [start]
    sk = f"{start[0]}_{start[1]}"
    gk = f"{goal[0]}_{goal[1]}"
    heap    = [(heuristic(start, goal), 0, sk, [sk])]
    visited = {}
    while heap:
        f, g, node, path = heapq.heappop(heap)
        if node in visited and visited[node] <= g:
            continue
        visited[node] = g
        if node == gk:
            return [tuple(map(int, n.split("_"))) for n in path]
        for nb, cost in graph.get(node, []):
            nr, nc = map(int, nb.split("_"))
            if (nr, nc) in blocked and nb != gk:
                continue
            ng = g + cost
            if nb not in visited or visited[nb] > ng:
                heapq.heappush(heap,
                    (ng + heuristic((nr,nc), goal), ng, nb, path + [nb]))
    return astar_path(graph, start, goal, None) if blocked else []


DIR_UP    = "up"
DIR_DOWN  = "down"
DIR_LEFT  = "left"
DIR_RIGHT = "right"


class Agent:
    """
    Represents a single robot agent.
    Stores position, goal, current path, movement direction,
    cargo status (for MAPD), and position history for loop detection.
    """
    def __init__(self, agent_id, start):
        self.id          = agent_id
        self.position    = tuple(start)
        self.goal        = None
        self.path        = []
        self.reached     = False
        self.direction   = DIR_DOWN
        self.task        = None
        self.has_cargo   = False
        self.task_phase  = None
        self._history    = deque(maxlen=6)  # last 6 positions for loop detection

    def is_active(self):
        """Returns True if agent has a goal and has not reached it yet."""
        return self.goal is not None and not self.reached

    def is_looping(self):
        """
        Detects period-2 oscillation: agent stuck in A→B→A→B loop.
        Checks if the last 4 positions follow the pattern [A, B, A, B].
        """
        h = list(self._history)
        if len(h) < 4:
            return False
        return h[-4] == h[-2] and h[-3] == h[-1] and h[-2] != h[-1]

    def set_goal(self, goal, graph, blocked=None):
        """Sets a new goal and immediately plans the path using A*."""
        self.goal    = snap_coord(graph, goal)
        self.reached = False
        self.path    = astar_path(graph, self.position, self.goal, blocked)
        if self.path and self.path[0] == self.position:
            self.path.pop(0)
        if not self.path and self.position == self.goal:
            self.reached = True
        self._history.clear()
        return self.reached or len(self.path) > 0

    def next_position(self):
        """Returns the next planned position, or current if no path."""
        return self.path[0] if self.path else self.position

    def commit_step(self, new_pos):
        """
        Moves the agent to new_pos, updates direction and history.
        Marks agent as reached if it arrived at the goal.
        """
        self._history.append(new_pos)
        if new_pos != self.position:
            if self.path and self.path[0] == new_pos:
                self.path.pop(0)
            dr = new_pos[0] - self.position[0]
            dc = new_pos[1] - self.position[1]
            if abs(dr) >= abs(dc):
                self.direction = DIR_DOWN if dr > 0 else DIR_UP
            else:
                self.direction = DIR_RIGHT if dc > 0 else DIR_LEFT
            self.position = new_pos
        if not self.path and self.goal and self.position == self.goal:
            self.reached = True

    def replan(self, graph, blocked=None):
        """Replans the path from current position to goal, avoiding blocked cells."""
        if self.goal and not self.reached:
            self.path = astar_path(graph, self.position, self.goal, blocked)
            if self.path and self.path[0] == self.position:
                self.path.pop(0)

    def neighbors(self, graph):
        """Returns the set of all adjacent cells in the graph."""
        key = f"{self.position[0]}_{self.position[1]}"
        return {tuple(map(int, nb.split("_"))) for nb, _ in graph.get(key, [])}

    def row_neighbors(self, graph):
        """Returns all cells in the same row as the agent's current position."""
        row = self.position[0]
        return {tuple(map(int, k.split("_")))
                for k in graph if int(k.split("_")[0]) == row}


class Simulator:
    """
    Main simulation engine.
    Loads the scenario, initializes agents, runs A* navigation,
    and resolves movement conflicts on each tick.
    """
    def __init__(self, scenario_path, edges_path):
        scenario    = load_scenario(scenario_path)
        self.graph  = load_graph(edges_path)
        self.agents = []
        self.tick   = 0
        self.type   = scenario["type"]

        # Load and snap task coordinates to the graph
        tasks_path = os.path.join(os.path.dirname(scenario_path), "tasks.json")
        all_tasks  = load_tasks(tasks_path)
        for t in all_tasks:
            t["pickup"]  = list(snap_coord(self.graph, xy_to_rowcol(t["pickup"])))
            t["dropoff"] = list(snap_coord(self.graph, xy_to_rowcol(t["dropoff"])))

        self.task_queue = sorted(all_tasks, key=lambda t: t.get("release_time", 0))
        self.done_tasks = []

        # Initialize agents from scenario
        for a in scenario["agents"]:
            agent = Agent(a["id"],
                          snap_coord(self.graph, xy_to_rowcol(a["start"])))
            if "goal" in a:
                agent.set_goal(xy_to_rowcol(a["goal"]), self.graph)
            self.agents.append(agent)

        # Assign initial tasks for MAPD scenarios
        if "mapd" in self.type:
            self._assign_tasks()

        print(f"Scenario type: {self.type}")
        print(f"Agents loaded: {len(self.agents)}")
        print(f"Tasks in queue: {len(self.task_queue)}")

    def _assign_tasks(self):
        """Assigns available tasks to idle agents (greedy: first available)."""
        available = [t for t in self.task_queue
                     if t.get("release_time", 0) <= self.tick]
        for agent in self.agents:
            if agent.task is None and available:
                task = available.pop(0)
                self.task_queue.remove(task)
                agent.task, agent.has_cargo = task, False
                agent.task_phase = "to_pickup"
                agent.set_goal(task["pickup"], self.graph)
                print(f"  {agent.id} -> pickup {task['pickup']}")

    def update(self):
        """
        Advances simulation by one tick.
        Steps:
        1. Detect period-2 oscillations and force reroute
        2. Compute desired next positions for all agents
        3. Resolve swap conflicts
        4. Reserve cells by priority and finalize moves
        5. Handle MAPD pickup/dropoff events
        """
        self.tick += 1

        active     = [a for a in self.agents if a.is_active()]
        passive    = [a for a in self.agents if not a.is_active()]
        active_pos = {a.position for a in active}

        # Step 1: period-2 loop detection
        # The lowest-priority looping agent is forced to reroute
        # through a neighbouring row to break the deadlock
        looping = [a for a in active if a.is_looping()]
        detour_agent = None
        if looping:
            looping_sorted = sorted(looping,
                                    key=lambda a: self.agents.index(a))
            detour_agent = looping_sorted[-1]
            same_row = detour_agent.row_neighbors(self.graph)
            same_row.discard(detour_agent.goal)
            full_blocked = (active_pos - {detour_agent.position}) | same_row
            full_blocked.discard(detour_agent.goal)
            detour_agent.replan(self.graph, full_blocked)
            for a in looping:
                a._history.clear()

        # Other looping agents wait one tick to let the detour agent move
        forced_wait = set()
        if looping:
            for a in looping:
                if a is not detour_agent:
                    forced_wait.add(a)

        # Step 2: compute desired next positions
        wants = {}
        for agent in active:
            if agent in forced_wait:
                wants[agent] = agent.position
                continue
            blocked = active_pos - {agent.position}
            desired = agent.next_position()
            if desired != agent.position and desired in blocked:
                agent.replan(self.graph, blocked)
                desired = agent.next_position()
            wants[agent] = desired
        for agent in passive:
            wants[agent] = agent.position

        # Step 3: resolve swap conflicts
        # When A wants B's position and B wants A's position:
        # higher-priority agent moves, lower-priority finds a safe neighbour
        swap_wait = set(forced_wait)
        processed_swaps = set()
        for ag in list(active):
            for other in list(active):
                if ag is other:
                    continue
                pair = frozenset([id(ag), id(other)])
                if pair in processed_swaps:
                    continue
                if (wants.get(ag) == other.position and
                        wants.get(other) == ag.position):
                    processed_swaps.add(pair)
                    high = ag if self.agents.index(ag) < self.agents.index(other) else other
                    low  = other if high is ag else ag
                    low_key = f"{low.position[0]}_{low.position[1]}"
                    safe_nb = None
                    for nb, _ in self.graph.get(low_key, []):
                        nb_pos = tuple(map(int, nb.split("_")))
                        if nb_pos != high.position and nb_pos not in active_pos:
                            safe_nb = nb_pos
                            break
                    if safe_nb:
                        wants[low] = safe_nb
                        swap_wait.add(low)
                    else:
                        swap_wait.add(ag)
                        swap_wait.add(other)
                        blocked_swap = (active_pos | {high.position}) - {low.position}
                        blocked_swap.discard(low.goal)
                        low.replan(self.graph, blocked_swap)

        # Step 4: cell reservation by priority
        # Agents with lower index get priority — they reserve cells first
        priority_order = (
            sorted(active,  key=lambda a: self.agents.index(a)) +
            sorted(passive, key=lambda a: self.agents.index(a))
        )
        reserved = {}
        moves    = {}

        for agent in priority_order:
            if agent in swap_wait and wants.get(agent) == agent.position:
                desired = agent.position
            else:
                desired = wants.get(agent, agent.position)

            if desired in reserved:
                if not agent.is_active():
                    moved = False
                    for nb, _ in self.graph.get(
                            f"{agent.position[0]}_{agent.position[1]}", []):
                        nb_pos = tuple(map(int, nb.split("_")))
                        if nb_pos not in reserved:
                            reserved[nb_pos] = agent
                            moves[agent] = nb_pos
                            moved = True
                            break
                    if not moved:
                        moves[agent] = agent.position
                        reserved.setdefault(agent.position, agent)
                else:
                    moves[agent] = agent.position
                    reserved.setdefault(agent.position, agent)
            else:
                reserved[desired] = agent
                moves[agent] = desired

        # Final collision check — resolve any remaining swap or vertex conflicts
        final_moves = dict(moves)
        changed = True
        while changed:
            changed = False
            for i, ag in enumerate(self.agents):
                dest_ag = final_moves.get(ag, ag.position)
                for j, other in enumerate(self.agents):
                    if i >= j:
                        continue
                    dest_other = final_moves.get(other, other.position)
                    real_swap = (
                        dest_ag == other.position and
                        dest_other == ag.position and
                        dest_ag != ag.position and
                        dest_other != other.position and
                        ag.is_active() and other.is_active()
                    )
                    if real_swap:
                        low = other
                        if final_moves[low] != low.position:
                            final_moves[low] = low.position
                            changed = True
                    if dest_ag == dest_other and dest_ag != ag.position:
                        low = other
                        if final_moves[low] != low.position:
                            final_moves[low] = low.position
                            changed = True

        # Apply all moves
        for agent in self.agents:
            agent.commit_step(final_moves[agent])

        # Step 5: MAPD pickup/dropoff events
        for agent in self.agents:
            if "mapd" not in self.type or not agent.task:
                continue
            task = agent.task
            if agent.task_phase == "to_pickup" and agent.reached:
                agent.has_cargo = True
                agent.task_phase = "to_dropoff"
                agent.reached = False
                agent.set_goal(task["dropoff"], self.graph)
                print(f"  {agent.id} picked up cargo -> dropoff {task['dropoff']}")
            elif agent.task_phase == "to_dropoff" and agent.reached:
                agent.has_cargo  = False
                agent.task       = None
                agent.task_phase = None
                agent.reached    = False
                self.done_tasks.append(task)
                print(f"  {agent.id} delivered! Done: {len(self.done_tasks)}")

        if "mapd" in self.type:
            self._assign_tasks()

    def get_agent_positions(self):
        """Returns list of (index, row, col, direction, has_cargo) for all agents."""
        return [(i, a.position[0], a.position[1], a.direction, a.has_cargo)
                for i, a in enumerate(self.agents)]

    def get_task_markers(self):
        """Returns list of active task markers (pickup/dropoff positions)."""
        markers = []
        for i, agent in enumerate(self.agents):
            if agent.task:
                if agent.task_phase == "to_pickup":
                    markers.append({"type": "pickup",
                                    "pos": tuple(agent.task["pickup"]),
                                    "agent_idx": i})
                markers.append({"type": "dropoff",
                                "pos": tuple(agent.task["dropoff"]),
                                "agent_idx": i})
        return markers

    def all_done(self):
        """Returns True when all agents have reached their goals (or all tasks delivered)."""
        if "mapd" in self.type:
            return (not self.task_queue
                    and all(a.task is None for a in self.agents)
                    and len(self.done_tasks) > 0)
        return all(a.reached for a in self.agents)