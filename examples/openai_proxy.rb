# An OpenAI-compatible /v1/embeddings endpoint in front of kohagi.
#
#     ruby examples/openai_proxy.rb --kohagi ./target/release/kohagi
#
# Then point any OpenAI client at it and nothing else in your code changes:
#
#     client = OpenAI::Client.new(uri_base: "http://127.0.0.1:8080", access_token: "unused")
#     client.embeddings(parameters: { model: "ruri-v3-130m", input: ["…", "…"] })
#
# That base URL swap is the point. kohagi has no HTTP mode — it speaks JSONL
# over a pipe, which is a smaller contract and works from any language that can
# spawn a process. This file is the bridge.
#
# ## How it works, and why this way
#
# One kohagi process per request, with `--format openai`, and its stdout
# returned verbatim. The response is already the right shape, `usage` included,
# so there is no envelope to assemble here.
#
# The obvious alternative — one long-lived kohagi, fed request by request — does
# not work, and it is worth knowing why before you try it. kohagi's protocol is
# a batch protocol: it embeds in chunks of 1024 records and flushes when a chunk
# fills or when stdin closes. A request of two texts produces nothing until one
# of those happens, so a server holding the pipe open waits forever. Closing
# stdin is the only end-of-request signal there is, and closing it ends the
# process.
#
# The cost is a model load per request: about 0.3 s warm on CPU for
# ruri-v3-130m. If that matters more than simplicity, batch texts into fewer,
# larger requests — which is what the API's array `input` is for anyway.
#
# ## Before swapping a production base URL
#
# - Dimensions differ. ruri-v3-130m returns 512 where text-embedding-3-small
#   returns 1536, so an existing index has to be rebuilt. The compatibility is
#   in the protocol, not in the vectors.
# - `model` in the request is ignored. Which checkpoint runs is decided by the
#   flags this proxy passes to kohagi. The response says which one actually ran.
#
# Standard library only (webrick is a bundled gem on Ruby 3; `gem install
# webrick` if require fails).

require "json"
require "open3"
require "optparse"
require "webrick"

options = {
  kohagi: "kohagi",
  host: "127.0.0.1",
  port: 8080,
  model_id: "cl-nagoya/ruri-v3-130m",
  prefix: "",
  device: "cpu"
}
parser = OptionParser.new do |o|
  o.on("--kohagi PATH") { |v| options[:kohagi] = v }
  o.on("--host HOST") { |v| options[:host] = v }
  o.on("--port PORT", Integer) { |v| options[:port] = v }
  o.on("--model-id ID") { |v| options[:model_id] = v }
  o.on("--prefix PREFIX", 'e.g. "検索文書: " for Ruri v3') { |v| options[:prefix] = v }
  o.on("--device DEVICE") { |v| options[:device] = v }
end
# Anything the parser does not recognize is passed through to kohagi, so
# `--max-seq-length 256` or `--coreml-buckets …` work without being redeclared
# here. OptionParser raises on unknown switches, so they are collected instead.
extra = []
argv = ARGV.dup
until argv.empty?
  begin
    parser.parse!(argv)
    break
  rescue OptionParser::InvalidOption => e
    extra.concat(e.args)
    # The value after an unknown switch belongs to it, not to this parser.
    extra << argv.shift unless argv.empty? || argv.first.start_with?("-")
  end
end

ARGV_KOHAGI = [
  options[:kohagi],
  "--model-id", options[:model_id],
  "--device", options[:device],
  "--prefix", options[:prefix],
  "--format", "openai",
  *extra
].freeze

# kohagi's own OpenAI response for +texts+, as a String.
def embed(texts)
  payload = texts.each_with_index.map { |t, i| JSON.generate({ id: i, text: t }) }.join("\n")
  out, err, status = Open3.capture3(*ARGV_KOHAGI, stdin_data: payload)
  # Exit 2 means some lines were skipped; the records that did come back are
  # still valid, but a proxy cannot return a short array as if nothing happened.
  # See PROTOCOL.md for the exit codes.
  raise(err.strip.empty? ? "kohagi failed" : err.strip) unless status.success?

  out
end

def send_json(res, status, payload)
  res.status = status
  res["Content-Type"] = "application/json"
  res.body = payload.is_a?(String) ? payload : JSON.generate(payload)
end

# The error shape OpenAI clients expect, so their exceptions carry the message
# rather than "unknown error".
def send_error(res, status, message)
  send_json(res, status, { error: { message: message, type: "invalid_request_error" } })
end

server = WEBrick::HTTPServer.new(
  BindAddress: options[:host],
  Port: options[:port],
  AccessLog: [],
  # kohagi's own stderr is the interesting log here.
  Logger: WEBrick::Log.new(File::NULL)
)

server.mount_proc "/v1/embeddings" do |req, res|
  next send_error(res, 405, "POST only") unless req.request_method == "POST"

  begin
    body = JSON.parse(req.body.to_s.empty? ? "{}" : req.body)
  rescue JSON::ParserError => e
    next send_error(res, 400, "invalid JSON: #{e.message}")
  end

  # The API takes a string or an array of them. Arrays of tokens are also legal
  # there and not supported here; say so rather than embedding the digits.
  given = body["input"]
  texts = given.is_a?(String) ? [given] : given
  unless texts.is_a?(Array) && !texts.empty? && texts.all?(String)
    next send_error(res, 400, "`input` must be a string or an array of strings")
  end

  begin
    send_json(res, 200, embed(texts))
  rescue RuntimeError => e
    send_error(res, 500, e.message)
  end
end

# Some clients list models before their first call.
server.mount_proc "/v1/models" do |_req, res|
  send_json(res, 200, {
              object: "list",
              data: [{ id: options[:model_id], object: "model", owned_by: "kohagi" }]
            })
end

trap("INT") { server.shutdown }
puts "kohagi-openai-proxy: http://#{options[:host]}:#{options[:port]}/v1  (#{options[:model_id]})"
server.start
