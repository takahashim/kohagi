# Embedding records from Ruby/Rails via kohagi's stdio protocol.
#
# The essential pattern: write stdin from one thread, read stdout from
# another, and map results back by id. Writing everything before reading
# anything can deadlock both processes once the pipe buffer fills.
#
# Exit codes (see PROTOCOL.md): 0 = clean, 2 = finished with skipped lines
# (received output is still valid — consume it, then investigate stderr),
# 1 = fatal.

require "open3"
require "json"

books = { 1 => "夏目漱石『吾輩は猫である』…", 2 => "青空文庫の長い紹介文…" }

cmd = ["kohagi", "--prefix", "検索文書: "]
embeddings = {}

Open3.popen3(*cmd) do |stdin, stdout, stderr, wait|
  writer = Thread.new do
    books.each { |id, text| stdin.puts JSON.generate(id: id, text: text) }
    stdin.close
  end
  logger = Thread.new { stderr.each_line { |l| warn l } }

  stdout.each_line do |line|
    rec = JSON.parse(line)
    embeddings[rec["id"]] = rec["embedding"]
  end

  writer.join
  logger.join
  status = wait.value
  raise "kohagi failed (exit #{status.exitstatus})" if status.exitstatus == 1
end

# e.g. verify dimensions before writing to a pgvector column
embeddings.each do |id, vec|
  raise "unexpected dim #{vec.size} for id #{id}" unless vec.size == 512
end
puts "embedded #{embeddings.size} records"
