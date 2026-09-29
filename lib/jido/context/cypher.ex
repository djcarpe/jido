defmodule Jido.Context.Cypher do
  @moduledoc """
  Builds Cypher fragments for the Glider engine.

  `Jido.Context` sends Glider complete statements with every value written as a
  literal (Glider also accepts `$name` parameters through `glider_ex`, which the
  engine does not use yet). That makes correct encoding a safety requirement
  rather than a convenience, so this module is the only place in
  `Jido.Context` that turns Elixir terms into query text.

  Two rules do the work:

  * **Values are escaped.** `encode_value/1` renders a term as a Glider literal,
    escaping quotes, backslashes and control characters in strings.
  * **Identifiers are validated, never escaped.** Labels, relationship types and
    property keys appear outside quotes, where no escape exists that would make
    an arbitrary binary safe. `identifier/1` accepts `[A-Za-z_][A-Za-z0-9_]*`
    and rejects everything else, so a hostile label fails loudly at the
    boundary instead of altering the query.

  Entity keys — the user-supplied strings that identify nodes across the mesh —
  are *values*, not identifiers, so they are unrestricted.
  """

  @identifier ~r/^[A-Za-z_][A-Za-z0-9_]*$/

  @typedoc "A term encodable as a Glider literal."
  @type value :: String.t() | number() | boolean() | nil | [value()]

  @doc """
  Encodes a term as a Glider literal.

  Glider values are null, bool, int, float, text, or a list of those. Anything
  else — a tuple, a map, a pid — has no literal form, so it is rendered as its
  `inspect/1` text rather than silently losing structure.

      iex> Jido.Context.Cypher.encode_value("Ada")
      ~S|"Ada"|

      iex> Jido.Context.Cypher.encode_value(~S|say "hi"|)
      ~S|"say \\"hi\\""|

      iex> Jido.Context.Cypher.encode_value([1, true, nil])
      "[1, true, null]"
  """
  @spec encode_value(term()) :: String.t()
  def encode_value(nil), do: "null"
  def encode_value(true), do: "true"
  def encode_value(false), do: "false"
  def encode_value(v) when is_integer(v), do: Integer.to_string(v)

  # Always plain decimal notation: glider's parser reads `1.5` and `-0.00002`
  # but not `2.0e-5`, which Float.to_string/1 produces for small values.
  def encode_value(v) when is_float(v) do
    cond do
      v != v -> "null"
      v in [:infinity, :neg_infinity] -> "null"
      true -> :erlang.float_to_binary(v, [:short]) |> plain_decimal()
    end
  end

  def encode_value(v) when is_binary(v), do: [?", escape(v), ?"] |> IO.iodata_to_binary()

  def encode_value(v) when is_atom(v), do: encode_value(Atom.to_string(v))

  def encode_value(v) when is_list(v) do
    "[" <> Enum.map_join(v, ", ", &encode_value/1) <> "]"
  end

  def encode_value(v), do: encode_value(inspect(v))

  defp plain_decimal(str) do
    case String.split(str, "e") do
      [_] -> str
      [mantissa, exponent] -> shift_decimal(mantissa, String.to_integer(exponent))
    end
  end

  defp shift_decimal(mantissa, exponent) do
    {sign, digits} =
      if String.starts_with?(mantissa, "-"),
        do: {"-", String.slice(mantissa, 1..-1//1)},
        else: {"", mantissa}

    [int, frac] =
      case String.split(digits, ".") do
        [i] -> [i, ""]
        [i, f] -> [i, f]
      end

    all = int <> frac
    point = String.length(int) + exponent

    cond do
      point <= 0 ->
        sign <> "0." <> String.duplicate("0", -point) <> all

      point >= String.length(all) ->
        sign <> all <> String.duplicate("0", point - String.length(all)) <> ".0"

      true ->
        sign <> String.slice(all, 0, point) <> "." <> String.slice(all, point..-1//1)
    end
    |> trim_fraction()
  end

  # "0.000020" -> "0.00002"; "12.0" stays "12.0".
  defp trim_fraction(str) do
    case String.split(str, ".") do
      [int, frac] ->
        trimmed = String.trim_trailing(frac, "0")
        int <> "." <> if(trimmed == "", do: "0", else: trimmed)

      _ ->
        str
    end
  end

  @doc """
  Validates a Cypher identifier — a label, relationship type or property key.

  Returns the identifier unchanged, or raises `ArgumentError`. Identifiers are
  interpolated unquoted, so there is no safe way to accept arbitrary text here.

      iex> Jido.Context.Cypher.identifier!("Person")
      "Person"

      iex> Jido.Context.Cypher.identifier!(:_seq)
      "_seq"
  """
  @spec identifier!(String.t() | atom()) :: String.t()
  def identifier!(name) when is_atom(name), do: identifier!(Atom.to_string(name))

  def identifier!(name) when is_binary(name) do
    if Regex.match?(@identifier, name) do
      name
    else
      raise ArgumentError, """
      invalid Cypher identifier: #{inspect(name)}

      Labels, relationship types and property keys are interpolated into the
      query unquoted, so they must match #{inspect(@identifier.source)}.
      Entity keys and property *values* have no such restriction.
      """
    end
  end

  @doc """
  Same as `identifier!/1` but returns `{:ok, name}` / `{:error, reason}`.
  """
  @spec identifier(String.t() | atom()) :: {:ok, String.t()} | {:error, term()}
  def identifier(name) do
    {:ok, identifier!(name)}
  rescue
    e in ArgumentError -> {:error, Exception.message(e)}
  end

  @doc """
  Renders a property map as a Cypher map literal, including the braces.

  Returns `""` for an empty map so it can be concatenated into a pattern
  unconditionally. Keys are sorted, which keeps generated queries stable and
  makes them comparable in tests.

      iex> Jido.Context.Cypher.props(%{"name" => "Ada", "age" => 36})
      ~S|{age: 36, name: "Ada"}|

      iex> Jido.Context.Cypher.props(%{})
      ""
  """
  @spec props(map()) :: String.t()
  def props(props) when map_size(props) == 0, do: ""

  def props(props) when is_map(props) do
    body =
      props
      |> Enum.map(fn {k, v} -> {identifier!(k), v} end)
      |> Enum.sort_by(&elem(&1, 0))
      |> Enum.map_join(", ", fn {k, v} -> "#{k}: #{encode_value(v)}" end)

    "{" <> body <> "}"
  end

  @doc """
  Renders `SET` assignments for a property map, bound to `var`.

  Returns `nil` for an empty map, so the caller can skip the clause.

      iex> Jido.Context.Cypher.set_props("n", %{"v" => 2})
      "n.v = 2"
  """
  @spec set_props(String.t(), map()) :: String.t() | nil
  def set_props(_var, props) when map_size(props) == 0, do: nil

  def set_props(var, props) when is_map(props) do
    props
    |> Enum.map(fn {k, v} -> {identifier!(k), v} end)
    |> Enum.sort_by(&elem(&1, 0))
    |> Enum.map_join(", ", fn {k, v} -> "#{var}.#{k} = #{encode_value(v)}" end)
  end

  @doc """
  Renders a label suffix — `:A:B` — from a list of labels.

  Returns `""` for an empty list.

      iex> Jido.Context.Cypher.labels(["Person", "Author"])
      ":Person:Author"
  """
  @spec labels([String.t() | atom()]) :: String.t()
  def labels([]), do: ""
  def labels(labels) when is_list(labels), do: Enum.map_join(labels, "", &":#{identifier!(&1)}")

  # Glider's lexer reads a double-quoted string with the usual backslash
  # escapes. Control characters are escaped rather than emitted raw so a value
  # containing a newline cannot break the query across lines.
  defp escape(binary) do
    for <<c <- binary>>, into: "", do: escape_char(c)
  end

  defp escape_char(?"), do: "\\\""
  defp escape_char(?\\), do: "\\\\"
  defp escape_char(?\n), do: "\\n"
  defp escape_char(?\r), do: "\\r"
  defp escape_char(?\t), do: "\\t"

  defp escape_char(c) when c < 0x20,
    do: "\\u" <> String.pad_leading(Integer.to_string(c, 16), 4, "0")

  defp escape_char(c), do: <<c>>
end
