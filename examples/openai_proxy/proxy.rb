# An OpenAI-compatible /v1/embeddings endpoint in front of kohagi.
#
#     ruby examples/openai_proxy/proxy.rb --kohagi ./target/release/kohagi
#
#     client = OpenAI::Client.new(uri_base: "http://127.0.0.1:8080", access_token: "unused")
#     client.embeddings(parameters: { model: "ruri-v3-130m", input: ["…", "…"] })
#
# See README.md in this directory for what this is for, how it works, and what
# to know before pointing production at it. Needs `gem install puma`; everything
# else is the standard library.

require "json"
require "open3"
require "optparse"
require "puma"
require "puma/configuration"
require "puma/launcher"

# One long-lived kohagi process, one request at a time.
#
# The mutex is not optional. kohagi's stdout carries batches in the order the
# batches were asked for, with nothing tying a reply to a requester, so two
# overlapping requests would each read the other's response.
class Kohagi
  def initialize(argv)
    @stdin, @stdout, @wait = Open3.popen2(*argv)
    @mutex = Mutex.new
  end

  # kohagi's own OpenAI response for +texts+, as a String.
  def embed(texts)
    payload = texts.each_with_index.map { |t, i| "#{JSON.generate({ id: i, text: t })}\n" }.join
    line = @mutex.synchronize do
      # The blank line ends the batch; without it kohagi waits for 1024 records
      # before embedding anything.
      @stdin.write("#{payload}\n")
      @stdin.flush
      @stdout.gets
    end
    raise "kohagi exited; see its stderr" if line.nil?

    line
  end
end

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

KOHAGI = Kohagi.new([
  options[:kohagi],
  "--model-id", options[:model_id],
  "--device", options[:device],
  "--prefix", options[:prefix],
  "--format", "openai",
  *extra
])

JSON_HEADERS = { "content-type" => "application/json" }.freeze

def reply(status, payload)
  [status, JSON_HEADERS, [payload.is_a?(String) ? payload : JSON.generate(payload)]]
end

# The error shape OpenAI clients expect, so their exceptions carry the message
# rather than "unknown error".
def fail_with(status, message)
  reply(status, { error: { message: message, type: "invalid_request_error" } })
end

def embeddings(env)
  return fail_with(405, "POST only") unless env["REQUEST_METHOD"] == "POST"

  begin
    raw = env["rack.input"].read
    body = JSON.parse(raw.to_s.empty? ? "{}" : raw)
  rescue JSON::ParserError => e
    return fail_with(400, "invalid JSON: #{e.message}")
  end

  # The API takes a string or an array of them. Arrays of tokens are also legal
  # there and not supported here; say so rather than embedding the digits.
  given = body["input"]
  texts = given.is_a?(String) ? [given] : given
  unless texts.is_a?(Array) && !texts.empty? && texts.all?(String)
    return fail_with(400, "`input` must be a string or an array of strings")
  end

  begin
    reply(200, KOHAGI.embed(texts))
  rescue RuntimeError => e
    fail_with(500, e.message)
  end
end

APP = lambda do |env|
  case env["PATH_INFO"].sub(%r{/\z}, "")
  when "/v1/embeddings" then embeddings(env)
  # Some clients list models before their first call.
  when "/v1/models"
    reply(200, { object: "list",
                 data: [{ id: options[:model_id], object: "model", owned_by: "kohagi" }] })
  else fail_with(404, "only /v1/embeddings is served")
  end
end

# Puma serves requests on a thread pool, which is why Kohagi is behind a mutex:
# one process, one batch at a time, with the threads queueing on it.
config = Puma::Configuration.new do |c|
  c.bind "tcp://#{options[:host]}:#{options[:port]}"
  c.app APP
  c.log_requests false
  c.environment "production"
end
puts "kohagi-openai-proxy: http://#{options[:host]}:#{options[:port]}/v1  (#{options[:model_id]})"
Puma::Launcher.new(config, events: Puma::Events.new).run
