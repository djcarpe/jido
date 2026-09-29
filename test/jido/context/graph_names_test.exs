defmodule Jido.Context.GraphNamesTest do
  use ExUnit.Case, async: false

  alias Jido.Context
  alias Jido.Context.Graph

  setup do
    unless Context.available?(), do: raise("the Glider engine is unavailable")
    reg = :"names_#{System.unique_integer([:positive])}"
    start_supervised!({Registry, keys: :unique, name: reg})
    {:ok, reg: reg}
  end

  test "a graph may be registered through a Registry instead of an atom", %{reg: reg} do
    name = {:via, Registry, {reg, {:graph, "org-1"}}}

    start_supervised!(
      {Graph, name: name, origin: "pod-a_org-1", location: :memory},
      id: :via_graph
    )

    {:ok, _} = Context.assert(name, "n:1", ["Thing"], %{"v" => 1})
    assert {:ok, %{props: %{"v" => 1}}} = Context.fetch(name, "n:1")
    assert Graph.origin(name) == "pod-a_org-1"
    assert [{_pid, _}] = Registry.lookup(reg, {:graph, "org-1"})
  end

  test "a registered name needs an explicit origin", %{reg: reg} do
    name = {:via, Registry, {reg, {:graph, "org-2"}}}
    Process.flag(:trap_exit, true)
    assert {:error, {%ArgumentError{}, _}} = Graph.start_link(name: name, location: :memory)
  end

  test "the handle can be read directly, and writes still replicate through commit", %{reg: reg} do
    name = {:via, Registry, {reg, {:graph, "org-3"}}}
    start_supervised!({Graph, name: name, origin: "o", location: :memory}, id: :handle_graph)
    {:ok, _} = Context.assert(name, "n:1", ["Thing"], %{"v" => 7})

    handle = Graph.handle(name)
    assert {:ok, %{rows: [[7]]}} = Glider.query(handle, ~s|MATCH (n:Thing) RETURN n.v|)
  end

  test "a late graph catches up from a peer's export, keeping stamps" do
    early = :"early_#{System.unique_integer([:positive])}"
    late = :"late_#{System.unique_integer([:positive])}"
    start_supervised!({Jido.Context.Graph, name: early, location: :memory}, id: :early)
    start_supervised!({Jido.Context.Graph, name: late, location: :memory}, id: :late)

    {:ok, _} = Jido.Context.assert(early, "paper:1", ["Paper"], %{title: "one"})
    {:ok, _} = Jido.Context.assert(early, "paper:2", ["Paper"], %{title: "two"})
    {:ok, jsonl} = Jido.Context.export(early)

    assert :ok = Jido.Context.Graph.import_snapshot(late, jsonl)

    {:ok, %{rows: rows}} =
      Jido.Context.query(late, "MATCH (n:Paper) RETURN n.title ORDER BY n.title")

    assert rows == [["one"], ["two"]]

    # The stamps came across: a fresh write on the late graph gets a higher seq.
    {:ok, delta} = Jido.Context.assert(late, "paper:3", ["Paper"], %{title: "three"})
    assert delta.seq > 2
  end

  test "a reopened file resumes its clock above what it holds" do
    dir = Path.join(System.tmp_dir!(), "jido-reopen-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)
    path = Path.join(dir, "g.gldb")
    name = :"reopen_#{System.unique_integer([:positive])}"

    {:ok, pid} = Jido.Context.Graph.start_link(name: name, location: {:disk, path: path})
    {:ok, first} = Jido.Context.assert(name, "k:1", ["K"], %{v: 1})
    {:ok, second} = Jido.Context.assert(name, "k:1", ["K"], %{v: 2})
    assert second.seq > first.seq
    GenServer.stop(pid)

    {:ok, pid} = Jido.Context.Graph.start_link(name: name, location: {:disk, path: path})
    {:ok, third} = Jido.Context.assert(name, "k:1", ["K"], %{v: 3})
    assert third.seq > second.seq
    {:ok, %{rows: [[3]]}} = Jido.Context.query(name, "MATCH (n:K) RETURN n.v")
    GenServer.stop(pid)
    File.rm_rf!(dir)
  end

  test "floats are written in plain decimal notation, whatever their size" do
    alias Jido.Context.Cypher
    assert Cypher.encode_value(1.5) == "1.5"
    assert Cypher.encode_value(-8.831631857901812e-5) == "-0.00008831631857901812"
    assert Cypher.encode_value(1.0e20) == "100000000000000000000.0"
    assert Cypher.encode_value(2.0e-5) == "0.00002"

    name = :"floats_#{System.unique_integer([:positive])}"
    start_supervised!({Jido.Context.Graph, name: name, location: :memory})
    {:ok, _} = Jido.Context.assert(name, "v:1", ["V"], %{vec: [1.0e-7, -3.25e-5, 12.0, 1.0e21]})
    {:ok, %{rows: [[vec]]}} = Jido.Context.query(name, "MATCH (n:V) RETURN n.vec")
    assert_in_delta Enum.at(vec, 0), 1.0e-7, 1.0e-20
    assert_in_delta Enum.at(vec, 1), -3.25e-5, 1.0e-20
    assert Enum.at(vec, 2) == 12.0
    assert_in_delta Enum.at(vec, 3), 1.0e21, 1.0e6
  end
end
