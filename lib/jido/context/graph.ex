defmodule Jido.Context.Graph do
  @moduledoc """
  One agent's context and knowledge graph: a Glider database, a Lamport clock,
  and a subscription to a mesh.

  This is the process that owns everything. Glider serialises calls against a
  handle internally — the graph is mutated in place, so concurrent access would
  be a data race — and a single owning GenServer satisfies that without ever
  contending on the NIF's mutex.

  Prefer the `Jido.Context` facade; this module is what it talks to.

  ## Two kinds of thing in one graph

  **Knowledge** is what the agents believe about the world: entities and the
  relations between them, meant to outlive any one task.

      Jido.Context.assert(graph, "paper:attention", ["Paper"], %{year: 2017})
      Jido.Context.relate(graph, "paper:attention", "CITES", "paper:seq2seq")

  **Context** is what is true of a particular piece of work: observations
  scoped to a session, a task, a conversation.

      Jido.Context.remember(graph, "task:42", "budget_remaining", 3)

  Both are nodes in one graph, which is the point — an agent can ask what it
  knows *and* what it is doing in a single traversal.

  ## Internal properties

  Every managed entity carries the replication stamp, and user properties may
  not begin with an underscore:

  | Property | Meaning |
  |---|---|
  | `_key` | the mesh-wide identity of this node |
  | `_seq`, `_origin` | the stamp that decides last-writer-wins |
  | `_topic` | the topic the writing delta was published on |
  | `_ts` | wall-clock time of the write, in ms |

  Managed nodes also carry the `:Ctx` label, alongside whatever labels the
  caller gave them, and `:Ctx(_key)` is indexed. A deleted node leaves a
  `:CtxTomb` node behind holding the stamp of its deletion, so a stale assert
  arriving later cannot resurrect it. Tombstones are the only `:CtxTomb` nodes
  and are invisible to `:Ctx` queries.

  ## Telemetry

  Every delta applied — the graph's own writes and those arriving from the
  mesh alike — is one event, so what the rule decided is countable:

  | Event | Measurements |
  |---|---|
  | `[:jido, :context, :delta, :applied]` | `duration`, `queued`, `ops`, `applied`, `superseded`, `tombstoned` |
  | `[:jido, :context, :delta, :failed]` | `duration` |
  | `[:jido, :context, :delta, :backlog]` | `deltas`, `ops`, `duration` |

  Metadata carries `graph`, `id`, `origin`, `topic`, `seq`, `ts`, `local`
  (did this graph write it) and, on `:applied`, `outcomes` — one of
  `:applied | :superseded | :tombstoned` per operation, in order. A
  superseded operation lost to a write with a higher stamp; a tombstoned one
  lost to a deletion. Neither is an error: that is last-writer-wins
  converging, now visible. `duration` is the time spent applying, `queued`
  the time from arrival to applied (0 for the graph's own writes).
  `:backlog` (metadata `graph`) follows every slice of the backlog: what is
  still waiting, and how long the slice took. `telemetry_events/0` lists
  the names.

  ## Catching up

  Deltas from the mesh, and a merged peer export, go into a backlog that is
  applied `:slice_ops` operations at a time; between slices the process
  takes its calls, so a commit or a read waits at most one slice however
  far behind the graph is. Stamps are read from an in-process index of
  every stamp in the graph, not a query per operation, and the statements
  that apply a delta are not traced one by one (the events above measure
  them).

  ## Durability

  Two independent things, worth keeping apart:

  * **The graph file.** With `location: {:disk, path: ...}` Glider opens a
    `.gldb` that is a write-ahead log and the persistent form at once. This is
    what survives a restart.
  * **The snapshot.** A JSON Lines dump written to a `Jido.Context.Store`. This
    is what *moves* — it is engine-independent and is what goes to S3, so a new
    agent anywhere can boot with what the mesh already knew.
  """

  use GenServer

  require Logger

  alias Jido.Context.Cypher
  alias Jido.Context.Delta
  alias Jido.Context.Engine
  alias Jido.Context.Mesh
  alias Jido.Context.Store

  @node_label "Ctx"
  @tomb_label "CtxTomb"
  @key_prop "_key"

  defmodule State do
    @moduledoc false
    @type t :: %__MODULE__{}
    defstruct [
      :name,
      :engine,
      :handle,
      :origin,
      :store,
      :mesh,
      :topics,
      :default_topic,
      :snapshot_key,
      :snapshot_every,
      :snapshot_interval,
      # Every stamp in the graph, by key: an ETS table this process owns, read
      # instead of a Cypher query per operation and kept current by every write.
      :stamps,
      # Deltas from the mesh (and callers waiting for them), applied a slice
      # at a time so a commit never waits behind the whole backlog.
      :backlog,
      lamport: 0,
      writes_since_snapshot: 0,
      backlog_ops: 0,
      draining: false,
      slice_ops: 500
    ]
  end

  # ===========================================================================
  # Lifecycle
  # ===========================================================================

  @doc """
  Starts a graph.

  ## Options

  * `:name` — required. Registered name and default origin.
  * `:origin` — the mesh-wide identity of this graph. Defaults to `:name`.
    Must be unique across the mesh: two graphs sharing an origin will tie in
    the last-writer-wins ordering and diverge.
  * `:location` — `:memory` (default) or `{:disk, path: "...", sync: :normal}`.
  * `:store` — a `Jido.Context.Store` spec for snapshots. Optional; without one
    the graph never snapshots.
  * `:mesh` — the name of a running `Jido.Context.Mesh`. Optional; without one
    the graph is private.
  * `:topics` — patterns to subscribe to. Default `["**"]`.
  * `:default_topic` — topic for writes that do not name one. Default `"context"`.
  * `:snapshot_every` — snapshot after this many local writes. Default `:never`.
  * `:snapshot_interval` — snapshot every N ms. Default `:never`.
  * `:restore` — `true` (default) to load a snapshot from the store on boot.
  * `:slice_ops` — most operations from the mesh applied before the process
    takes its next call. Default 500.
  """
  @spec start_link(keyword()) :: GenServer.on_start()
  def start_link(opts) do
    name = Keyword.fetch!(opts, :name)
    GenServer.start_link(__MODULE__, opts, name: process_name(name))
  end

  @doc false
  def child_spec(opts) do
    %{
      id: {__MODULE__, Keyword.fetch!(opts, :name)},
      start: {__MODULE__, :start_link, [opts]},
      type: :worker
    }
  end

  @doc false
  @spec process_name(atom() | String.t() | GenServer.name()) :: GenServer.name()
  def process_name(name) when is_atom(name), do: :"#{name}.Graph"
  def process_name(name) when is_binary(name), do: :"#{name}.Graph"
  # A registered name — `{:via, Registry, {Reg, key}}` or `{:global, key}` —
  # is used as given, so an application with many graphs (one per tenant,
  # say) does not mint an atom for each.
  def process_name({:via, _, _} = name), do: name
  def process_name({:global, _} = name), do: name

  @impl true
  def init(opts) do
    # A graph holds a Glider handle and, when disk-backed, a file lock. Trapping
    # exits is what makes `terminate/2` run when the process that started this
    # one goes away — without it a normal exit signal from a linked parent is
    # ignored, and the handle and lock outlive the agent that owned them.
    Process.flag(:trap_exit, true)

    name = Keyword.fetch!(opts, :name)
    engine = Keyword.get(opts, :engine, Engine.default())
    location = normalize_location(Keyword.get(opts, :location, :memory))

    with {:ok, handle} <- engine.open(location),
         :ok <- ensure_indexes(engine, handle) do
      state = %State{
        name: name,
        engine: engine,
        handle: handle,
        origin: origin_of(opts, name),
        store: opts |> Keyword.get(:store) |> normalize_store(),
        mesh: Keyword.get(opts, :mesh),
        topics: Keyword.get(opts, :topics, ["**"]),
        default_topic: Keyword.get(opts, :default_topic, "context"),
        snapshot_key: Keyword.get(opts, :snapshot_key, "snapshots/#{name_string(name)}.jsonl"),
        snapshot_every: Keyword.get(opts, :snapshot_every, :never),
        snapshot_interval: Keyword.get(opts, :snapshot_interval, :never),
        slice_ops: Keyword.get(opts, :slice_ops, 500),
        stamps: :ets.new(__MODULE__, [:ordered_set, :private]),
        backlog: :queue.new()
      }

      # A file reopened holds stamps from before; the clock must resume above
      # them or this origin would re-issue sequence numbers and lose every
      # comparison against its own earlier writes.
      state = %{state | lamport: load_stamps(state)}

      state =
        if Keyword.get(opts, :restore, true), do: restore_snapshot(state), else: state

      if state.mesh, do: Mesh.subscribe(state.mesh, state.topics)
      schedule_snapshot(state)

      {:ok, state}
    else
      {:error, reason} -> {:stop, {:graph_open_failed, reason}}
    end
  end

  defp normalize_location(:memory), do: :memory

  defp normalize_location({:disk, opts}) do
    {:file, Keyword.fetch!(opts, :path), Keyword.get(opts, :sync, :normal)}
  end

  defp normalize_location({:file, _path, _sync} = location), do: location

  defp normalize_store(nil), do: nil
  defp normalize_store(spec), do: Store.normalize(spec)

  # `:Ctx(_key)` is the only lookup path that matters: every upsert starts with
  # it, and without the index each one is a linear label scan.
  defp ensure_indexes(engine, handle) do
    with {:ok, _} <- engine.run(handle, "INDEX ON :#{@node_label}(#{@key_prop})"),
         {:ok, _} <- engine.run(handle, "INDEX ON :#{@tomb_label}(#{@key_prop})") do
      :ok
    end
  end

  @impl true
  def terminate(_reason, %State{} = state) do
    # A snapshot on the way out is best-effort: a graph being shut down because
    # its store is unreachable must not hang the supervisor.
    if state.store, do: snapshot_now(state)
    state.engine.close(state.handle)
    :ok
  end

  def terminate(_reason, _state), do: :ok

  # ===========================================================================
  # Writes
  # ===========================================================================

  @doc "Upserts a node."
  @spec assert_node(atom(), String.t(), [String.t()], map(), keyword()) ::
          {:ok, Delta.t()} | {:error, term()}
  def assert_node(graph, key, labels, props, opts \\ []) do
    commit(graph, [{:put_node, key, labels, stringify(props)}], opts)
  end

  @doc "Upserts an edge, creating placeholder endpoints if they are not there yet."
  @spec assert_edge(atom(), String.t(), String.t(), String.t(), map(), keyword()) ::
          {:ok, Delta.t()} | {:error, term()}
  def assert_edge(graph, from, type, to, props, opts \\ []) do
    commit(graph, [{:put_edge, from, type, to, stringify(props)}], opts)
  end

  @doc "Deletes a node and its edges, leaving a tombstone."
  @spec retract_node(atom(), String.t(), keyword()) :: {:ok, Delta.t()} | {:error, term()}
  def retract_node(graph, key, opts \\ []), do: commit(graph, [{:drop_node, key}], opts)

  @doc "Deletes an edge."
  @spec retract_edge(atom(), String.t(), String.t(), String.t(), keyword()) ::
          {:ok, Delta.t()} | {:error, term()}
  def retract_edge(graph, from, type, to, opts \\ []) do
    commit(graph, [{:drop_edge, from, type, to}], opts)
  end

  @doc """
  Applies several operations as one delta.

  A batch is atomic with respect to the mesh: peers see all of it or none of
  it, which matters when a node and the edge that gives it meaning must arrive
  together.
  """
  @spec commit(atom(), [Delta.op()], keyword()) :: {:ok, Delta.t()} | {:error, term()}
  def commit(graph, ops, opts \\ []) do
    GenServer.call(process_name(graph), {:commit, ops, opts}, timeout(opts))
  end

  @doc "Applies a delta that arrived from the mesh. Never republished."
  @spec apply_delta(atom(), Delta.t()) :: :ok
  def apply_delta(graph, %Delta{} = delta) do
    GenServer.cast(process_name(graph), {:apply_delta, delta})
  end

  # ===========================================================================
  # Reads
  # ===========================================================================

  @doc "Runs a raw Cypher query."
  @spec query(atom(), String.t(), keyword()) :: {:ok, Engine.result()} | {:error, term()}
  def query(graph, cypher, opts \\ []) do
    GenServer.call(process_name(graph), {:query, cypher}, timeout(opts))
  end

  @doc "Fetches one node by key."
  @spec fetch(atom(), String.t(), keyword()) :: {:ok, map()} | :not_found | {:error, term()}
  def fetch(graph, key, opts \\ []) do
    GenServer.call(process_name(graph), {:fetch, key}, timeout(opts))
  end

  @doc "Node, edge and label counts."
  @spec stats(atom(), keyword()) :: {:ok, map()} | {:error, term()}
  def stats(graph, opts \\ []), do: GenServer.call(process_name(graph), :stats, timeout(opts))

  @doc "Dumps the whole graph as JSON Lines."
  @spec export(atom(), keyword()) :: {:ok, String.t()} | {:error, term()}
  def export(graph, opts \\ []), do: GenServer.call(process_name(graph), :export, timeout(opts))

  @doc """
  Removes tombstones older than `older_than_ms` (a `_ts` in milliseconds).
  A tombstone only has to outlive the deltas that could still arrive for its
  key; once the mesh has settled — seconds, normally — it is dead weight in
  exports and scans. Local maintenance: every node purges its own copy.
  Returns the number removed.
  """
  @spec purge_tombstones(term(), integer(), keyword()) ::
          {:ok, non_neg_integer()} | {:error, term()}
  def purge_tombstones(graph, older_than_ms, opts \\ []),
    do: GenServer.call(process_name(graph), {:purge_tombstones, older_than_ms}, timeout(opts))

  @doc "Writes a snapshot to the configured store."
  @spec snapshot(atom(), keyword()) :: :ok | {:error, term()}
  def snapshot(graph, opts \\ []),
    do: GenServer.call(process_name(graph), :snapshot, timeout(opts))

  @doc "The telemetry events this graph emits; see the module documentation."
  @spec telemetry_events() :: [[atom()]]
  def telemetry_events,
    do: [
      [:jido, :context, :delta, :applied],
      [:jido, :context, :delta, :failed],
      [:jido, :context, :delta, :backlog]
    ]

  @doc "This graph's origin, as it appears in delta stamps."
  @spec origin(atom()) :: String.t()
  def origin(graph), do: GenServer.call(process_name(graph), :origin)

  @doc """
  The engine handle this graph owns, for reads that bypass the process.

  A read model (an `Ash.DataLayer`, say) can query the handle directly; the
  NIF serialises access, so that is safe. Writes must still go through
  `commit/3`, or they are neither stamped nor replicated.
  """
  @spec handle(atom() | GenServer.name()) :: term()
  def handle(graph), do: GenServer.call(process_name(graph), :handle)

  @doc """
  Loads a JSON Lines snapshot — the output of `export/1` on another graph —
  into this one. Managed nodes keep the stamps they carry, so a delta that
  arrives later still wins or loses by its own `{seq, origin}`, and the
  clock resumes above everything imported. This is how a graph that starts
  late catches up from a peer that was there all along.
  """
  @spec import_snapshot(atom() | GenServer.name(), String.t(), keyword()) ::
          :ok | {:error, term()}
  def import_snapshot(graph, jsonl, opts \\ []),
    do:
      GenServer.call(
        process_name(graph),
        {:import_snapshot, jsonl},
        Keyword.get(opts, :timeout, 60_000)
      )

  @doc """
  Merges another graph's export into this one, keeping last-writer-wins.

  Unlike `import_snapshot/3`, which loads a snapshot into an empty graph,
  this reads every managed node and edge of the export with the stamp it
  carries and applies it as if it had arrived from the mesh: a newer stamp
  wins, an older one is dropped, a tombstone deletes, and doing it twice
  changes nothing. It is how a graph that was down catches up on what the
  mesh carried meanwhile.

  The deltas join the backlog, so the graph keeps serving while they apply.
  With `wait: false` this returns as soon as they are queued; by default it
  returns when they are applied.
  """
  @spec merge_snapshot(atom() | GenServer.name(), String.t(), keyword()) ::
          {:ok, non_neg_integer()} | {:error, term()}
  def merge_snapshot(graph, jsonl, opts \\ []),
    do:
      GenServer.call(
        process_name(graph),
        {:merge_snapshot, jsonl, Keyword.get(opts, :wait, true)},
        Keyword.get(opts, :timeout, 120_000)
      )

  @doc """
  Blocks until every delta already delivered to this graph has been applied.

  Mesh delivery is asynchronous, so a test that publishes on one graph and
  reads from another needs a barrier rather than a sleep.
  """
  @spec sync(atom(), timeout()) :: :ok
  def sync(graph, timeout \\ 5_000), do: GenServer.call(process_name(graph), :sync, timeout)

  defp timeout(opts), do: Keyword.get(opts, :timeout, 15_000)

  # ===========================================================================
  # Server
  # ===========================================================================

  @impl true
  def handle_call({:commit, ops, opts}, _from, state) do
    topic = Keyword.get(opts, :topic, state.default_topic)

    try do
      lamport = state.lamport + 1
      delta = Delta.new(topic, state.origin, lamport, ops)
      state = %{state | lamport: lamport}

      case apply_ops(state, delta) do
        {:ok, state} ->
          if state.mesh, do: Mesh.publish(state.mesh, delta)
          {:reply, {:ok, delta}, maybe_snapshot(state)}

        {:error, reason} ->
          {:reply, {:error, reason}, state}
      end
    rescue
      e in ArgumentError -> {:reply, {:error, Exception.message(e)}, state}
    end
  end

  def handle_call({:query, cypher}, _from, state) do
    {:reply, state.engine.query(state.handle, cypher), state}
  end

  def handle_call({:purge_tombstones, cutoff}, _from, state) when is_integer(cutoff) do
    count =
      case state.engine.query(
             state.handle,
             "MATCH (t#{label(@tomb_label)}) WHERE t._ts < #{cutoff} RETURN count(t)"
           ) do
        {:ok, %{rows: [[n]]}} when is_integer(n) -> n
        _ -> 0
      end

    if count > 0 do
      result = run(state, "MATCH (t#{label(@tomb_label)}) WHERE t._ts < #{cutoff} DELETE t")
      load_tombs(state)

      case result do
        {:ok, _} -> {:reply, {:ok, count}, state}
        {:error, e} -> {:reply, {:error, e}, state}
      end
    else
      {:reply, {:ok, 0}, state}
    end
  end

  def handle_call({:fetch, key}, _from, state) do
    {:reply, do_fetch(state, key), state}
  end

  def handle_call(:stats, _from, state) do
    {:reply, state.engine.stats(state.handle), state}
  end

  def handle_call(:export, _from, state) do
    {:reply, state.engine.export(state.handle), state}
  end

  def handle_call({:merge_snapshot, jsonl, wait?}, from, state) do
    deltas = snapshot_deltas(jsonl)
    # Our own writes come back in a peer's export and are skipped, but the
    # clock must still resume above them.
    top = deltas |> Enum.map(& &1.seq) |> Enum.max(fn -> 0 end)

    state =
      Enum.reduce(deltas, %{state | lamport: max(state.lamport, top)}, &receive_delta(&2, &1))

    reply = {:ok, length(deltas)}

    if wait?,
      do: {:noreply, after_backlog(state, from, reply)},
      else: {:reply, reply, state}
  end

  def handle_call({:import_snapshot, jsonl}, _from, state) do
    case state.engine.import(state.handle, jsonl) do
      {:ok, _} -> {:reply, :ok, %{state | lamport: max(state.lamport, load_stamps(state))}}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  def handle_call(:snapshot, _from, state) do
    {:reply, snapshot_now(state), %{state | writes_since_snapshot: 0}}
  end

  def handle_call(:origin, _from, state), do: {:reply, state.origin, state}

  def handle_call(:handle, _from, state), do: {:reply, state.handle, state}

  def handle_call(:sync, from, state), do: {:noreply, after_backlog(state, from, :ok)}

  @impl true
  def handle_cast({:apply_delta, delta}, state), do: {:noreply, receive_delta(state, delta)}

  @impl true
  def handle_info({:jido_context_delta, _mesh, %Delta{} = delta}, state) do
    {:noreply, receive_delta(state, delta)}
  end

  def handle_info(:drain, state), do: {:noreply, drain(%{state | draining: false})}

  def handle_info(:snapshot_tick, state) do
    state =
      if state.store do
        snapshot_now(state)
        %{state | writes_since_snapshot: 0}
      else
        state
      end

    schedule_snapshot(state)
    {:noreply, state}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  # A delta from our own origin is one we already applied before publishing it.
  defp receive_delta(%State{origin: origin} = state, %Delta{origin: origin}), do: state

  # A delta from the mesh joins the backlog; the clock moves now, so a commit
  # taken before the delta is applied still sorts after what we have seen.
  defp receive_delta(state, %Delta{} = delta) do
    job = %{delta: delta, ops: delta.ops, outcomes: [], duration: 0, at: System.monotonic_time()}

    %{
      state
      | lamport: max(state.lamport, delta.seq) + 1,
        backlog: :queue.in({:delta, job}, state.backlog),
        backlog_ops: state.backlog_ops + length(delta.ops)
    }
    |> schedule_drain()
  end

  # Replies once everything queued before it is applied.
  defp after_backlog(state, from, reply) do
    if :queue.is_empty(state.backlog) do
      GenServer.reply(from, reply)
      state
    else
      %{state | backlog: :queue.in({:reply, from, reply}, state.backlog)}
    end
  end

  defp schedule_drain(%State{draining: true} = state), do: state

  defp schedule_drain(state) do
    send(self(), :drain)
    %{state | draining: true}
  end

  # One slice of the backlog, then back to the mailbox: calls that arrived
  # meanwhile run before the next slice, which is queued behind them.
  defp drain(state) do
    started = System.monotonic_time()
    state = drain_slice(state, state.slice_ops)

    :telemetry.execute(
      [:jido, :context, :delta, :backlog],
      %{
        deltas: :queue.len(state.backlog),
        ops: state.backlog_ops,
        duration: System.monotonic_time() - started
      },
      %{graph: state.name}
    )

    if :queue.is_empty(state.backlog), do: state, else: schedule_drain(state)
  end

  defp drain_slice(state, budget) do
    case :queue.out(state.backlog) do
      {{:value, {:reply, from, reply}}, rest} ->
        GenServer.reply(from, reply)
        drain_slice(%{state | backlog: rest}, budget)

      {{:value, {:delta, job}}, rest} when budget > 0 ->
        {chunk, remaining} = Enum.split(job.ops, budget)
        state = %{state | backlog: rest, backlog_ops: state.backlog_ops - length(chunk)}
        budget = budget - length(chunk)

        case apply_chunk(state, job.delta, chunk) do
          {:ok, state, outcomes, duration} ->
            job = %{
              job
              | ops: remaining,
                outcomes: outcomes ++ job.outcomes,
                duration: job.duration + duration
            }

            if remaining == [] do
              applied(
                state,
                job.delta,
                Enum.reverse(job.outcomes),
                job.duration,
                System.monotonic_time() - job.at
              )

              drain_slice(state, budget)
            else
              # Out of budget mid-delta: the rest goes first next slice.
              %{state | backlog: :queue.in_r({:delta, job}, state.backlog)}
            end

          {:error, reason, duration} ->
            delta = job.delta
            failed(state, delta, reason, job.duration + duration)

            Logger.warning(
              "jido context graph #{inspect(state.name)}: could not apply #{delta.id} " <>
                "from #{delta.origin}: #{inspect(reason)}"
            )

            drain_slice(%{state | backlog_ops: state.backlog_ops - length(remaining)}, budget)
        end

      _empty_or_out_of_budget ->
        state
    end
  end

  # ===========================================================================
  # Applying operations
  # ===========================================================================

  # Every operation reports what became of it — `:applied`, `:superseded` by
  # a write with a higher stamp, or `:tombstoned` by a deletion that wins —
  # and the delta as a whole is one telemetry event carrying the tally. A
  # losing write is not an error; it is the convergence rule working, and
  # until it was counted nobody could see it happen.
  defp apply_ops(state, %Delta{} = delta) do
    case apply_chunk(state, delta, delta.ops) do
      {:ok, state, outcomes, duration} ->
        applied(state, delta, Enum.reverse(outcomes), duration, 0)
        {:ok, state}

      {:error, reason, duration} ->
        failed(state, delta, reason, duration)
        {:error, reason}
    end
  end

  # Outcomes come back newest first. The statements are not traced one by
  # one: a delta of thousands of operations would be thousands of spans.
  defp apply_chunk(state, delta, ops) do
    started = System.monotonic_time()

    result =
      quietly(state, fn ->
        Enum.reduce_while(ops, {:ok, state, []}, fn op, {:ok, acc, outcomes} ->
          case apply_op(acc, delta, op) do
            {:ok, acc, outcome} -> {:cont, {:ok, acc, [outcome | outcomes]}}
            {:error, reason} -> {:halt, {:error, reason}}
          end
        end)
      end)

    duration = System.monotonic_time() - started

    case result do
      {:ok, state, outcomes} -> {:ok, state, outcomes, duration}
      {:error, reason} -> {:error, reason, duration}
    end
  end

  defp quietly(state, fun) do
    if function_exported?(state.engine, :quietly, 1), do: state.engine.quietly(fun), else: fun.()
  end

  defp applied(state, delta, outcomes, duration, queued) do
    tally = Enum.frequencies(outcomes)

    :telemetry.execute(
      [:jido, :context, :delta, :applied],
      %{
        duration: duration,
        queued: queued,
        ops: length(outcomes),
        applied: Map.get(tally, :applied, 0),
        superseded: Map.get(tally, :superseded, 0),
        tombstoned: Map.get(tally, :tombstoned, 0)
      },
      Map.put(delta_metadata(state, delta), :outcomes, outcomes)
    )
  end

  defp failed(state, delta, reason, duration) do
    :telemetry.execute(
      [:jido, :context, :delta, :failed],
      %{duration: duration},
      Map.put(delta_metadata(state, delta), :reason, reason)
    )
  end

  defp delta_metadata(state, delta) do
    %{
      graph: state.name,
      id: delta.id,
      origin: delta.origin,
      topic: delta.topic,
      seq: delta.seq,
      ts: delta.ts,
      local: delta.origin == state.origin
    }
  end

  defp apply_op(state, delta, {:put_node, key, labels, props}) do
    stamp = {delta.seq, delta.origin}

    with {:ok, tomb} <- read_stamp(state, @tomb_label, key) do
      if tomb && Delta.compare_stamp(tomb, stamp) != :lt do
        # The node was deleted by a write that wins over this one.
        {:ok, state, :tombstoned}
      else
        with {:ok, state, outcome} <- write_node(state, delta, key, labels, props),
             :ok <- clear_tombstone(state, key, tomb) do
          {:ok, state, outcome}
        end
      end
    end
  end

  defp apply_op(state, delta, {:drop_node, key}) do
    stamp = {delta.seq, delta.origin}

    with {:ok, existing} <- read_stamp(state, @node_label, key) do
      if existing && Delta.compare_stamp(existing, stamp) == :gt do
        {:ok, state, :superseded}
      else
        with {:ok, _} <-
               run(state, "MATCH (n#{label(@node_label)} #{key_match(key)}) DETACH DELETE n"),
             :ok <- forget_node(state, key),
             {:ok, state} <- write_tombstone(state, delta, key) do
          {:ok, state, :applied}
        end
      end
    end
  end

  defp apply_op(state, delta, {:put_edge, from, type, to, props}) do
    stamp = {delta.seq, delta.origin}

    with {:ok, state} <- ensure_endpoint(state, delta, from),
         {:ok, state} <- ensure_endpoint(state, delta, to),
         {:ok, existing} <- read_edge_stamp(state, from, type, to) do
      cond do
        is_nil(existing) ->
          with {:ok, state} <- create_edge(state, delta, from, type, to, props),
               do: {:ok, state, :applied}

        Delta.compare_stamp(existing, stamp) == :lt ->
          with {:ok, state} <- update_edge(state, delta, from, type, to, props),
               do: {:ok, state, :applied}

        true ->
          {:ok, state, :superseded}
      end
    end
  end

  defp apply_op(state, delta, {:drop_edge, from, type, to}) do
    stamp = {delta.seq, delta.origin}

    with {:ok, existing} <- read_edge_stamp(state, from, type, to) do
      if existing && Delta.compare_stamp(existing, stamp) == :gt do
        {:ok, state, :superseded}
      else
        with {:ok, _} <- run(state, edge_match(from, type, to) <> " DELETE r") do
          forget_edge(state, from, type, to)
          {:ok, state, :applied}
        end
      end
    end
  end

  # An edge can arrive before the nodes it connects — a peer may have published
  # them on a different topic, or in a delta still in flight. A placeholder node
  # carries no labels and the stamp of the edge that needed it, so a later, real
  # assert of that node wins the comparison and fills it in.
  defp ensure_endpoint(state, _delta, key) do
    with {:ok, existing} <- read_stamp(state, @node_label, key) do
      if existing do
        {:ok, state}
      else
        # A placeholder is bookkeeping, not a write that wins or loses: it
        # carries the lowest stamp there is, so the node's real write —
        # whenever it arrives — replaces it.
        create_placeholder(state, key)
      end
    end
  end

  defp write_node(state, delta, key, labels, props) do
    with {:ok, existing} <- read_stamp(state, @node_label, key) do
      stamp = {delta.seq, delta.origin}

      cond do
        is_nil(existing) ->
          with {:ok, state} <- create_node(state, delta, key, labels, props),
               do: {:ok, state, :applied}

        Delta.compare_stamp(existing, stamp) == :lt ->
          with {:ok, state} <- update_node(state, delta, key, labels, props),
               do: {:ok, state, :applied}

        true ->
          {:ok, state, :superseded}
      end
    end
  end

  defp create_placeholder(state, key) do
    props = %{@key_prop => key, "_seq" => 0, "_origin" => "", "_placeholder" => true}
    statement = "CREATE (n#{Cypher.labels([@node_label])} #{Cypher.props(props)})"
    with {:ok, _} <- run(state, statement), do: remember(state, {:n, key}, {0, ""})
  end

  defp create_node(state, delta, key, labels, props) do
    all_props = Map.merge(props, stamp_props(delta, key))
    all_labels = [@node_label | Enum.map(labels, &Cypher.identifier!/1)] |> Enum.uniq()

    statement = "CREATE (n#{Cypher.labels(all_labels)} #{Cypher.props(all_props)})"

    with {:ok, _} <- run(state, statement), do: remember(state, {:n, key}, stamp_of(delta))
  end

  defp update_node(state, delta, key, labels, props) do
    all_props = props |> Map.merge(stamp_props(delta, key)) |> Map.put("_placeholder", nil)

    set_clause =
      [Cypher.set_props("n", all_props), label_set("n", labels)]
      |> Enum.reject(&is_nil/1)
      |> Enum.join(", ")

    statement = "MATCH (n#{label(@node_label)} #{key_match(key)}) SET #{set_clause}"

    with {:ok, _} <- run(state, statement), do: remember(state, {:n, key}, stamp_of(delta))
  end

  defp label_set(_var, []), do: nil

  defp label_set(var, labels) do
    Enum.map_join(labels, ", ", fn label -> "#{var}:#{Cypher.identifier!(label)}" end)
  end

  defp create_edge(state, delta, from, type, to, props) do
    all_props = Map.merge(props, edge_stamp_props(delta))

    statement =
      "MATCH (a#{label(@node_label)} #{key_match(from)}), (b#{label(@node_label)} #{key_match(to)}) " <>
        "CREATE (a)-[:#{Cypher.identifier!(type)} #{Cypher.props(all_props)}]->(b)"

    with {:ok, _} <- run(state, statement), do: remember_edge(state, from, type, to, delta)
  end

  defp update_edge(state, delta, from, type, to, props) do
    all_props = Map.merge(props, edge_stamp_props(delta))
    statement = edge_match(from, type, to) <> " SET " <> Cypher.set_props("r", all_props)

    with {:ok, _} <- run(state, statement), do: remember_edge(state, from, type, to, delta)
  end

  defp write_tombstone(state, delta, key) do
    props = stamp_props(delta, key)

    with {:ok, existing} <- read_stamp(state, @tomb_label, key) do
      statement =
        if existing do
          "MATCH (t#{label(@tomb_label)} #{key_match(key)}) SET " <> Cypher.set_props("t", props)
        else
          "CREATE (t#{label(@tomb_label)} #{Cypher.props(props)})"
        end

      with {:ok, _} <- run(state, statement), do: remember(state, {:t, key}, stamp_of(delta))
    end
  end

  defp clear_tombstone(_state, _key, nil), do: :ok

  defp clear_tombstone(state, key, _tomb) do
    with {:ok, _} <-
           run(state, "MATCH (t#{label(@tomb_label)} #{key_match(key)}) DETACH DELETE t") do
      :ets.delete(state.stamps, {:t, key})
      :ok
    end
  end

  # An edge is identified by its endpoints and type, so it carries the stamp
  # but no `_key` of its own.
  defp edge_stamp_props(delta) do
    %{
      "_seq" => delta.seq,
      "_origin" => delta.origin,
      "_topic" => delta.topic,
      "_ts" => delta.ts
    }
  end

  defp stamp_props(delta, key) do
    %{
      @key_prop => key,
      "_seq" => delta.seq,
      "_origin" => delta.origin,
      "_topic" => delta.topic,
      "_ts" => delta.ts
    }
  end

  # ===========================================================================
  # Reading stamps
  # ===========================================================================

  # Every stamp lives in `state.stamps` as well as on its node or edge, so
  # deciding last-writer-wins is a lookup rather than a query:
  #
  #   {:n, key}             a node's stamp (`{0, ""}` for a placeholder)
  #   {:t, key}             a tombstone's
  #   {:e, from, type, to}  an edge's
  #   {:i, to, from, type}  the same edge seen from its target, so dropping a
  #                         node finds the edges into it without a scan
  #
  # Loaded once when the graph opens (or its contents are replaced) and
  # kept current by every write below.
  defp read_stamp(state, @node_label, key), do: {:ok, lookup(state, {:n, key})}
  defp read_stamp(state, @tomb_label, key), do: {:ok, lookup(state, {:t, key})}

  defp read_edge_stamp(state, from, type, to), do: {:ok, lookup(state, {:e, from, type, to})}

  defp lookup(state, key) do
    case :ets.lookup(state.stamps, key) do
      [{_, stamp}] -> stamp
      [] -> nil
    end
  end

  defp stamp_of(delta), do: {delta.seq, delta.origin}

  defp remember(state, key, stamp) do
    :ets.insert(state.stamps, {key, stamp})
    {:ok, state}
  end

  defp remember_edge(state, from, type, to, delta) do
    :ets.insert(state.stamps, [
      {{:e, from, type, to}, stamp_of(delta)},
      {{:i, to, from, type}, true}
    ])

    {:ok, state}
  end

  defp forget_edge(state, from, type, to) do
    :ets.delete(state.stamps, {:e, from, type, to})
    :ets.delete(state.stamps, {:i, to, from, type})
  end

  # The node and every edge in or out of it: what DETACH DELETE removed.
  defp forget_node(state, key) do
    :ets.delete(state.stamps, {:n, key})

    for [type, to] <- :ets.match(state.stamps, {{:e, key, :"$1", :"$2"}, :_}),
        do: forget_edge(state, key, type, to)

    for [from, type] <- :ets.match(state.stamps, {{:i, key, :"$1", :"$2"}, :_}),
        do: forget_edge(state, from, type, key)

    :ok
  end

  # Reads every stamp in the graph into the index; returns the highest seq,
  # where the clock must resume.
  defp load_stamps(state) do
    :ets.delete_all_objects(state.stamps)

    nodes = stamp_rows!(state, "MATCH (n#{label(@node_label)}) RETURN n._key, n._seq, n._origin")
    tombs = load_tombs(state)

    edges =
      stamp_rows!(
        state,
        "MATCH (a#{label(@node_label)})-[r]->(b#{label(@node_label)}) " <>
          "RETURN a._key, type(r), b._key, r._seq, r._origin"
      )

    :ets.insert(state.stamps, for([key, seq, origin] <- nodes, do: {{:n, key}, {seq, origin}}))

    :ets.insert(
      state.stamps,
      Enum.flat_map(edges, fn [from, type, to, seq, origin] ->
        [{{:e, from, type, to}, {seq, origin}}, {{:i, to, from, type}, true}]
      end)
    )

    [nodes, tombs, edges]
    |> Enum.flat_map(fn rows -> Enum.map(rows, &Enum.at(&1, -2)) end)
    |> Enum.max(fn -> 0 end)
  end

  defp load_tombs(state) do
    :ets.match_delete(state.stamps, {{:t, :_}, :_})
    tombs = stamp_rows!(state, "MATCH (t#{label(@tomb_label)}) RETURN t._key, t._seq, t._origin")
    :ets.insert(state.stamps, for([key, seq, origin] <- tombs, do: {{:t, key}, {seq, origin}}))
    tombs
  end

  # Only stamped entities count, as only they did when stamps were queried.
  defp stamp_rows!(state, statement) do
    case state.engine.query(state.handle, statement) do
      {:ok, %{rows: rows}} ->
        Enum.filter(rows, fn row ->
          [seq, origin] = Enum.take(row, -2)
          is_integer(seq) and is_binary(origin) and Enum.all?(Enum.drop(row, -2), &is_binary/1)
        end)

      {:error, reason} ->
        raise "jido context graph #{inspect(state.name)}: could not read stamps: #{inspect(reason)}"
    end
  end

  defp do_fetch(state, key) do
    statement = "MATCH (n#{label(@node_label)} #{key_match(key)}) RETURN n LIMIT 1"

    case state.engine.query(state.handle, statement) do
      {:ok, %{rows: [[node] | _]}} -> {:ok, node}
      {:ok, _} -> :not_found
      {:error, reason} -> {:error, reason}
    end
  end

  defp label(name), do: ":" <> name

  defp key_match(key), do: "{#{@key_prop}: #{Cypher.encode_value(key)}}"

  defp edge_match(from, type, to) do
    "MATCH (a#{label(@node_label)} #{key_match(from)})-[r:#{Cypher.identifier!(type)}]->" <>
      "(b#{label(@node_label)} #{key_match(to)})"
  end

  defp run(state, statement), do: state.engine.run(state.handle, statement)

  # ===========================================================================
  # Snapshots
  # ===========================================================================

  # An export, read back as the deltas that would have produced it: one per
  # stamp, in stamp order, so the receiving graph judges each by
  # last-writer-wins exactly as it would have on the mesh.
  defp snapshot_deltas(jsonl) do
    lines =
      jsonl
      |> String.split("\n", trim: true)
      |> Enum.flat_map(fn line ->
        case Jason.decode(line) do
          {:ok, %{} = entry} -> [entry]
          _ -> []
        end
      end)

    nodes = for %{"type" => "node"} = n <- lines, into: %{}, do: {n["id"], n}

    node_ops =
      for {_,
           %{
             "labels" => labels,
             "props" => %{"_key" => key, "_seq" => seq, "_origin" => origin} = props
           }} <- nodes,
          origin != "" do
        op =
          if @tomb_label in labels,
            do: {:drop_node, key},
            else: {:put_node, key, labels -- [@node_label], user_props(props)}

        {{origin, seq}, props["_ts"], op}
      end

    edge_ops =
      for %{"type" => "edge", "label" => type, "from" => from, "to" => to, "props" => props} <-
            lines,
          %{"_seq" => seq, "_origin" => origin} = props,
          from_key = get_in(nodes, [from, "props", "_key"]),
          to_key = get_in(nodes, [to, "props", "_key"]),
          not is_nil(from_key) and not is_nil(to_key) do
        {{origin, seq}, props["_ts"], {:put_edge, from_key, type, to_key, user_props(props)}}
      end

    (node_ops ++ edge_ops)
    |> Enum.group_by(fn {stamp, _, _} -> stamp end)
    |> Enum.sort_by(fn {{origin, seq}, _} -> {seq, origin} end)
    |> Enum.map(fn {{origin, seq}, entries} ->
      ts = entries |> Enum.map(&elem(&1, 1)) |> Enum.reject(&is_nil/1) |> List.first()

      Delta.new("merge", origin, seq, Enum.map(entries, &elem(&1, 2)),
        ts: ts || System.system_time(:millisecond)
      )
    end)
  end

  defp user_props(props),
    do: props |> Enum.reject(fn {k, _} -> String.starts_with?(k, "_") end) |> Map.new()

  defp restore_snapshot(%State{store: nil} = state), do: state

  defp restore_snapshot(state) do
    case Store.get(state.store, state.snapshot_key) do
      {:ok, jsonl} when byte_size(jsonl) > 0 ->
        case state.engine.import(state.handle, jsonl) do
          {:ok, _} ->
            Logger.debug("jido context graph #{inspect(state.name)}: restored a snapshot")
            %{state | lamport: load_stamps(state)}

          {:error, reason} ->
            Logger.warning(
              "jido context graph #{inspect(state.name)}: snapshot restore failed: " <>
                inspect(reason)
            )

            state
        end

      _ ->
        state
    end
  end

  defp maybe_snapshot(%State{store: nil} = state), do: state

  defp maybe_snapshot(%State{snapshot_every: :never} = state), do: state

  defp maybe_snapshot(%State{snapshot_every: every} = state) when is_integer(every) do
    writes = state.writes_since_snapshot + 1

    if writes >= every do
      snapshot_now(state)
      %{state | writes_since_snapshot: 0}
    else
      %{state | writes_since_snapshot: writes}
    end
  end

  defp snapshot_now(%State{store: nil}), do: {:error, :no_store}

  defp snapshot_now(state) do
    with {:ok, jsonl} <- state.engine.export(state.handle),
         :ok <- Store.put(state.store, state.snapshot_key, jsonl) do
      state.engine.checkpoint(state.handle)
      :ok
    else
      {:error, reason} ->
        Logger.warning(
          "jido context graph #{inspect(state.name)}: snapshot failed: #{inspect(reason)}"
        )

        {:error, reason}
    end
  end

  defp schedule_snapshot(%State{snapshot_interval: :never}), do: :ok

  defp schedule_snapshot(%State{snapshot_interval: ms}) when is_integer(ms) do
    Process.send_after(self(), :snapshot_tick, ms)
    :ok
  end

  defp stringify(props) when is_map(props) do
    Map.new(props, fn {k, v} -> {to_string(k), v} end)
  end

  # An origin must be a plain identifier; a registered-name tuple has none of
  # its own, so it must be given explicitly.
  defp origin_of(opts, name) do
    case Keyword.get(opts, :origin) do
      nil when is_atom(name) or is_binary(name) -> to_string(name)
      nil -> raise ArgumentError, "a graph named #{inspect(name)} needs an explicit :origin"
      origin -> to_string(origin)
    end
  end

  defp name_string(name) when is_atom(name) or is_binary(name), do: to_string(name)
  defp name_string({:via, _, {_, key}}), do: key |> inspect() |> String.replace(~r/[^\w.-]+/, "_")
  defp name_string({:global, key}), do: key |> inspect() |> String.replace(~r/[^\w.-]+/, "_")
  defp name_string(other), do: other |> inspect() |> String.replace(~r/[^\w.-]+/, "_")
end
