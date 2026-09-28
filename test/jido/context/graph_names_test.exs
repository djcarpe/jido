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
    {:ok, %{rows: rows}} = Jido.Context.query(late, "MATCH (n:Paper) RETURN n.title ORDER BY n.title")
    assert rows == [["one"], ["two"]]

    # The stamps came across: a fresh write on the late graph gets a higher seq.
    {:ok, delta} = Jido.Context.assert(late, "paper:3", ["Paper"], %{title: "three"})
    assert delta.seq > 2
  end
end
