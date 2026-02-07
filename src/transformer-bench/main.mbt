///|
struct BenchArgs {
  steps : Int
  warmup : Int
  batch_size : Int
  seq_len : Int
  d_model : Int
  num_heads : Int
  num_layers : Int
  d_ff : Int
  lr : Float
  seed : Int
  print_every : Int
  repeat : Int
  sweep : Bool
  adamw : Bool
  shard_meta : String?
  shard_max_shards : Int?
  shard_max_tokens : Int?
} derive(Show, Eq)

///|
fn bench_args_default() -> BenchArgs {
  {
    steps: 40,
    warmup: 5,
    batch_size: 8,
    seq_len: 32,
    d_model: 64,
    num_heads: 4,
    num_layers: 2,
    d_ff: 256,
    lr: Float::from_double(0.005),
    seed: 42,
    print_every: 10,
    repeat: 256,
    sweep: false,
    adamw: false,
    shard_meta: None,
    shard_max_shards: None,
    shard_max_tokens: None,
  }
}

///|
fn parse_positive_int(name : String, value : String) -> Result[Int, String] {
  let parsed = try? @strconv.parse_int(value)
  match parsed {
    Ok(v) => if v > 0 { Ok(v) } else { Err(name + " must be > 0") }
    Err(err) => Err("invalid " + name + ": " + err.to_string())
  }
}

///|
fn parse_nonnegative_int(name : String, value : String) -> Result[Int, String] {
  let parsed = try? @strconv.parse_int(value)
  match parsed {
    Ok(v) => if v >= 0 { Ok(v) } else { Err(name + " must be >= 0") }
    Err(err) => Err("invalid " + name + ": " + err.to_string())
  }
}

///|
fn parse_positive_float(name : String, value : String) -> Result[Float, String] {
  let parsed = try? @strconv.parse_double(value)
  match parsed {
    Ok(v) =>
      if v > 0.0 {
        Ok(Float::from_double(v))
      } else {
        Err(name + " must be > 0")
      }
    Err(err) => Err("invalid " + name + ": " + err.to_string())
  }
}

///|
fn print_usage(program : String) -> Unit {
  println("Usage: " + program + " [options]")
  println("  --steps N         measured steps (default 40)")
  println("  --warmup N        warmup steps (default 5)")
  println("  --batch-size N    mini-batch size (default 8)")
  println("  --seq-len N       sequence length (default 32)")
  println("  --d-model N       model width (default 64)")
  println("  --heads N         number of heads (default 4)")
  println("  --layers N        number of layers (default 2)")
  println("  --d-ff N          FFN hidden size (default 256)")
  println("  --lr X            learning rate (default 0.005)")
  println("  --seed N          RNG seed (default 42)")
  println("  --print-every N   progress interval (default 10)")
  println("  --repeat N        corpus repeat count (default 256)")
  println("  --sweep           sweep seq/head/layer combinations")
  println("  --adamw           use AdamW optimizer (default SGD)")
  println("  --shard-meta P    meta.json path for token shards")
  println("  --shard-max-shards N limit number of shard files to load")
  println("  --shard-max-tokens N limit number of token ids to load")
}

///|
fn parse_args(args : Array[String]) -> Result[BenchArgs, String] {
  let default = bench_args_default()
  let mut steps = default.steps
  let mut warmup = default.warmup
  let mut batch_size = default.batch_size
  let mut seq_len = default.seq_len
  let mut d_model = default.d_model
  let mut num_heads = default.num_heads
  let mut num_layers = default.num_layers
  let mut d_ff = default.d_ff
  let mut lr = default.lr
  let mut seed = default.seed
  let mut print_every = default.print_every
  let mut repeat = default.repeat
  let mut sweep = default.sweep
  let mut adamw = default.adamw
  let mut shard_meta : String? = default.shard_meta
  let mut shard_max_shards : Int? = default.shard_max_shards
  let mut shard_max_tokens : Int? = default.shard_max_tokens
  let mut i = 1
  while i < args.length() {
    let arg = args[i]
    if arg == "--help" || arg == "-h" {
      return Err("help")
    }
    if arg == "--sweep" {
      sweep = true
      i = i + 1
      continue
    }
    if arg == "--adamw" {
      adamw = true
      i = i + 1
      continue
    }
    if arg == "--steps" {
      if i + 1 >= args.length() {
        return Err("missing value for --steps")
      }
      match parse_positive_int("steps", args[i + 1]) {
        Ok(v) => steps = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--steps=") {
      let view = arg.sub(start=8) catch {
        _ => return Err("invalid --steps value")
      }
      match parse_positive_int("steps", view.to_string()) {
        Ok(v) => steps = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--warmup" {
      if i + 1 >= args.length() {
        return Err("missing value for --warmup")
      }
      match parse_nonnegative_int("warmup", args[i + 1]) {
        Ok(v) => warmup = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--warmup=") {
      let view = arg.sub(start=9) catch {
        _ => return Err("invalid --warmup value")
      }
      match parse_nonnegative_int("warmup", view.to_string()) {
        Ok(v) => warmup = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--batch-size" {
      if i + 1 >= args.length() {
        return Err("missing value for --batch-size")
      }
      match parse_positive_int("batch-size", args[i + 1]) {
        Ok(v) => batch_size = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--batch-size=") {
      let view = arg.sub(start=13) catch {
        _ => return Err("invalid --batch-size value")
      }
      match parse_positive_int("batch-size", view.to_string()) {
        Ok(v) => batch_size = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--seq-len" {
      if i + 1 >= args.length() {
        return Err("missing value for --seq-len")
      }
      match parse_positive_int("seq-len", args[i + 1]) {
        Ok(v) => seq_len = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--seq-len=") {
      let view = arg.sub(start=10) catch {
        _ => return Err("invalid --seq-len value")
      }
      match parse_positive_int("seq-len", view.to_string()) {
        Ok(v) => seq_len = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--d-model" {
      if i + 1 >= args.length() {
        return Err("missing value for --d-model")
      }
      match parse_positive_int("d-model", args[i + 1]) {
        Ok(v) => d_model = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--d-model=") {
      let view = arg.sub(start=10) catch {
        _ => return Err("invalid --d-model value")
      }
      match parse_positive_int("d-model", view.to_string()) {
        Ok(v) => d_model = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--heads" {
      if i + 1 >= args.length() {
        return Err("missing value for --heads")
      }
      match parse_positive_int("heads", args[i + 1]) {
        Ok(v) => num_heads = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--heads=") {
      let view = arg.sub(start=8) catch {
        _ => return Err("invalid --heads value")
      }
      match parse_positive_int("heads", view.to_string()) {
        Ok(v) => num_heads = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--layers" {
      if i + 1 >= args.length() {
        return Err("missing value for --layers")
      }
      match parse_positive_int("layers", args[i + 1]) {
        Ok(v) => num_layers = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--layers=") {
      let view = arg.sub(start=9) catch {
        _ => return Err("invalid --layers value")
      }
      match parse_positive_int("layers", view.to_string()) {
        Ok(v) => num_layers = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--d-ff" {
      if i + 1 >= args.length() {
        return Err("missing value for --d-ff")
      }
      match parse_positive_int("d-ff", args[i + 1]) {
        Ok(v) => d_ff = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--d-ff=") {
      let view = arg.sub(start=7) catch {
        _ => return Err("invalid --d-ff value")
      }
      match parse_positive_int("d-ff", view.to_string()) {
        Ok(v) => d_ff = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--lr" {
      if i + 1 >= args.length() {
        return Err("missing value for --lr")
      }
      match parse_positive_float("lr", args[i + 1]) {
        Ok(v) => lr = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--lr=") {
      let view = arg.sub(start=5) catch {
        _ => return Err("invalid --lr value")
      }
      match parse_positive_float("lr", view.to_string()) {
        Ok(v) => lr = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--seed" {
      if i + 1 >= args.length() {
        return Err("missing value for --seed")
      }
      match parse_nonnegative_int("seed", args[i + 1]) {
        Ok(v) => seed = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--seed=") {
      let view = arg.sub(start=7) catch {
        _ => return Err("invalid --seed value")
      }
      match parse_nonnegative_int("seed", view.to_string()) {
        Ok(v) => seed = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--print-every" {
      if i + 1 >= args.length() {
        return Err("missing value for --print-every")
      }
      match parse_positive_int("print-every", args[i + 1]) {
        Ok(v) => print_every = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--print-every=") {
      let view = arg.sub(start=14) catch {
        _ => return Err("invalid --print-every value")
      }
      match parse_positive_int("print-every", view.to_string()) {
        Ok(v) => print_every = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--repeat" {
      if i + 1 >= args.length() {
        return Err("missing value for --repeat")
      }
      match parse_positive_int("repeat", args[i + 1]) {
        Ok(v) => repeat = v
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--repeat=") {
      let view = arg.sub(start=9) catch {
        _ => return Err("invalid --repeat value")
      }
      match parse_positive_int("repeat", view.to_string()) {
        Ok(v) => repeat = v
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--shard-meta" {
      if i + 1 >= args.length() {
        return Err("missing value for --shard-meta")
      }
      shard_meta = Some(args[i + 1])
      i = i + 2
      continue
    }
    if arg.has_prefix("--shard-meta=") {
      let view = arg.sub(start=13) catch {
        _ => return Err("invalid --shard-meta value")
      }
      shard_meta = Some(view.to_string())
      i = i + 1
      continue
    }
    if arg == "--shard-max-shards" {
      if i + 1 >= args.length() {
        return Err("missing value for --shard-max-shards")
      }
      match parse_positive_int("shard-max-shards", args[i + 1]) {
        Ok(v) => shard_max_shards = Some(v)
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--shard-max-shards=") {
      let view = arg.sub(start=19) catch {
        _ => return Err("invalid --shard-max-shards value")
      }
      match parse_positive_int("shard-max-shards", view.to_string()) {
        Ok(v) => shard_max_shards = Some(v)
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    if arg == "--shard-max-tokens" {
      if i + 1 >= args.length() {
        return Err("missing value for --shard-max-tokens")
      }
      match parse_positive_int("shard-max-tokens", args[i + 1]) {
        Ok(v) => shard_max_tokens = Some(v)
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--shard-max-tokens=") {
      let view = arg.sub(start=19) catch {
        _ => return Err("invalid --shard-max-tokens value")
      }
      match parse_positive_int("shard-max-tokens", view.to_string()) {
        Ok(v) => shard_max_tokens = Some(v)
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    return Err("unknown option: " + arg)
  }
  if shard_meta is None {
    match shard_max_shards {
      Some(_) => return Err("--shard-max-shards requires --shard-meta")
      None => ()
    }
    match shard_max_tokens {
      Some(_) => return Err("--shard-max-tokens requires --shard-meta")
      None => ()
    }
  }
  Ok({
    steps,
    warmup,
    batch_size,
    seq_len,
    d_model,
    num_heads,
    num_layers,
    d_ff,
    lr,
    seed,
    print_every,
    repeat,
    sweep,
    adamw,
    shard_meta,
    shard_max_shards,
    shard_max_tokens,
  })
}

///|
fn make_default_corpus(repeat : Int) -> String {
  let base = "To be, or not to be, that is the question.\n"
  let mut out = ""
  for _i = 0; _i < repeat; _i = _i + 1 {
    out = out + base
  }
  out
}

///|
struct BenchCorpus {
  all_ids : Array[Int]
  vocab : Int
  source : String
} derive(Show, Eq)

///|
async fn load_bench_corpus(cfg : BenchArgs) -> Result[BenchCorpus, String] {
  match cfg.shard_meta {
    Some(meta_path) => {
      let meta = match load_token_shards_meta(meta_path) {
        Ok(v) => v
        Err(msg) => return Err(msg)
      }
      if meta.n_vocab <= 0 {
        return Err("invalid n_vocab in shard meta: " + meta.n_vocab.to_string())
      }
      let ids = match
        load_token_ids_from_shards(
          meta,
          cfg.shard_max_shards,
          cfg.shard_max_tokens,
        ) {
        Ok(v) => v
        Err(msg) => return Err(msg)
      }
      if ids.length() == 0 {
        return Err("no token ids loaded from shards")
      }
      Ok({
        all_ids: ids,
        vocab: meta.n_vocab,
        source: "shards(" + meta_path + ")",
      })
    }
    None => {
      let text = make_default_corpus(cfg.repeat)
      let tokenizer = @tensor.char_tokenizer_from_text(text)
      let ids = tokenizer.encode(text)
      if ids.length() == 0 {
        return Err("default corpus produced zero tokens")
      }
      Ok({
        all_ids: ids,
        vocab: tokenizer.vocab_size(),
        source: "default-corpus",
      })
    }
  }
}

///|
fn ns_to_ms(ns : UInt64) -> Float {
  Float::from_uint64(ns) / Float::from_int(1000000)
}

///|
async fn run_bench(cfg : BenchArgs) -> Unit {
  let corpus = match load_bench_corpus(cfg) {
    Ok(v) => v
    Err(msg) => {
      println("error: " + msg)
      return
    }
  }
  let windows = corpus.all_ids.length() - cfg.seq_len
  if windows <= 0 {
    println(
      "dataset is too short for seq_len=" +
      cfg.seq_len.to_string() +
      " tokens=" +
      corpus.all_ids.length().to_string(),
    )
    return
  }
  let effective_batch = if cfg.batch_size > windows {
    windows
  } else {
    cfg.batch_size
  }
  let config = @tensor.transformer_config(
    corpus.vocab,
    cfg.d_model,
    cfg.num_heads,
    cfg.num_layers,
    cfg.d_ff,
    cfg.seq_len,
  ).unwrap()
  let metrics = if cfg.adamw {
    @tensor.transformer_train_lm_profile_steps_ids_adamw(
      corpus.all_ids,
      config,
      cfg.warmup,
      cfg.steps,
      effective_batch,
      cfg.lr,
      cfg.seed,
    )
  } else {
    @tensor.transformer_train_lm_profile_steps_ids(
      corpus.all_ids,
      config,
      cfg.warmup,
      cfg.steps,
      effective_batch,
      cfg.lr,
      cfg.seed,
    )
  }
  if metrics.length() == 0 {
    println("no measured steps")
    return
  }
  println("=== Transformer LM Benchmark ===")
  println(
    "data=" + corpus.source + " tokens=" + corpus.all_ids.length().to_string(),
  )
  println(
    "steps=" +
    cfg.steps.to_string() +
    " warmup=" +
    cfg.warmup.to_string() +
    " batch_size=" +
    effective_batch.to_string() +
    " seq_len=" +
    cfg.seq_len.to_string(),
  )
  println(
    "d_model=" +
    cfg.d_model.to_string() +
    " heads=" +
    cfg.num_heads.to_string() +
    " layers=" +
    cfg.num_layers.to_string() +
    " d_ff=" +
    cfg.d_ff.to_string() +
    " vocab=" +
    corpus.vocab.to_string(),
  )
  let optim = if cfg.adamw { "adamw" } else { "sgd" }
  println(
    "lr=" +
    cfg.lr.to_string() +
    " seed=" +
    cfg.seed.to_string() +
    " optim=" +
    optim,
  )
  println("")
  let mut total_ns = 0UL
  let mut total_loss = Float::from_int(0)
  let mut total_ppl = Float::from_int(0)
  let tokens_per_step = effective_batch * cfg.seq_len
  for i = 0; i < metrics.length(); i = i + 1 {
    let m = metrics[i]
    total_ns = total_ns + m.step_ns
    total_loss = total_loss + m.loss
    total_ppl = total_ppl + m.perplexity
    if (i + 1) % cfg.print_every == 0 || i + 1 == metrics.length() {
      let step_tps = if m.step_ns > 0UL {
        Float::from_int(tokens_per_step) *
        Float::from_int(1000000000) /
        Float::from_uint64(m.step_ns)
      } else {
        Float::from_int(0)
      }
      println(
        "step " +
        (i + 1).to_string() +
        "/" +
        metrics.length().to_string() +
        ": loss=" +
        m.loss.to_string() +
        " ppl=" +
        m.perplexity.to_string() +
        " step_ms=" +
        ns_to_ms(m.step_ns).to_string() +
        " tok/s=" +
        step_tps.to_string(),
      )
    }
  }
  let n = Float::from_int(metrics.length())
  let avg_ms = ns_to_ms(total_ns) / n
  let avg_loss = total_loss / n
  let avg_ppl = total_ppl / n
  let overall_tps = if total_ns > 0UL {
    Float::from_int(tokens_per_step * metrics.length()) *
    Float::from_int(1000000000) /
    Float::from_uint64(total_ns)
  } else {
    Float::from_int(0)
  }
  println("")
  println("summary:")
  println("  avg_step_ms=" + avg_ms.to_string())
  println("  avg_loss=" + avg_loss.to_string())
  println("  avg_ppl=" + avg_ppl.to_string())
  println("  avg_tok/s=" + overall_tps.to_string())
}

///|
async fn run_sweep(cfg : BenchArgs) -> Unit {
  let corpus = match load_bench_corpus(cfg) {
    Ok(v) => v
    Err(msg) => {
      println("error: " + msg)
      return
    }
  }
  let seq_opts = [16, 32, 64]
  let head_opts = [2, 4, 8]
  let layer_opts = [1, 2, 4]
  println("=== Transformer LM Sweep ===")
  println(
    "data=" + corpus.source + " tokens=" + corpus.all_ids.length().to_string(),
  )
  println(
    "steps=" +
    cfg.steps.to_string() +
    " warmup=" +
    cfg.warmup.to_string() +
    " batch_size=" +
    cfg.batch_size.to_string() +
    " d_model=" +
    cfg.d_model.to_string() +
    " d_ff=" +
    cfg.d_ff.to_string(),
  )
  println("seq\theads\tlayers\tavg_step_ms\tavg_loss\tavg_ppl\tavg_tok/s")
  for si = 0; si < seq_opts.length(); si = si + 1 {
    let seq = seq_opts[si]
    let windows = corpus.all_ids.length() - seq
    if windows <= 0 {
      continue
    }
    let effective_batch = if cfg.batch_size > windows {
      windows
    } else {
      cfg.batch_size
    }
    for hi = 0; hi < head_opts.length(); hi = hi + 1 {
      let heads = head_opts[hi]
      if cfg.d_model % heads != 0 {
        continue
      }
      for li = 0; li < layer_opts.length(); li = li + 1 {
        let layers = layer_opts[li]
        let config = @tensor.transformer_config(
          corpus.vocab,
          cfg.d_model,
          heads,
          layers,
          cfg.d_ff,
          seq,
        ).unwrap()
        let metrics = if cfg.adamw {
          @tensor.transformer_train_lm_profile_steps_ids_adamw(
            corpus.all_ids,
            config,
            cfg.warmup,
            cfg.steps,
            effective_batch,
            cfg.lr,
            cfg.seed,
          )
        } else {
          @tensor.transformer_train_lm_profile_steps_ids(
            corpus.all_ids,
            config,
            cfg.warmup,
            cfg.steps,
            effective_batch,
            cfg.lr,
            cfg.seed,
          )
        }
        if metrics.length() == 0 {
          continue
        }
        let mut total_ns = 0UL
        let mut total_loss = Float::from_int(0)
        let mut total_ppl = Float::from_int(0)
        for i = 0; i < metrics.length(); i = i + 1 {
          total_ns = total_ns + metrics[i].step_ns
          total_loss = total_loss + metrics[i].loss
          total_ppl = total_ppl + metrics[i].perplexity
        }
        let n = Float::from_int(metrics.length())
        let avg_ms = ns_to_ms(total_ns) / n
        let avg_loss = total_loss / n
        let avg_ppl = total_ppl / n
        let avg_tps = if total_ns > 0UL {
          Float::from_int(effective_batch * seq * metrics.length()) *
          Float::from_int(1000000000) /
          Float::from_uint64(total_ns)
        } else {
          Float::from_int(0)
        }
        println(
          seq.to_string() +
          "\t" +
          heads.to_string() +
          "\t" +
          layers.to_string() +
          "\t" +
          avg_ms.to_string() +
          "\t" +
          avg_loss.to_string() +
          "\t" +
          avg_ppl.to_string() +
          "\t" +
          avg_tps.to_string(),
        )
      }
    }
  }
}

///|
fn main {
  @async.run_async_main(bench_main)
}

///|
async fn bench_main() -> Unit {
  let args = @env.args()
  let program = if args.length() > 0 { args[0] } else { "transformer-bench" }
  let cfg = match parse_args(args) {
    Ok(v) => v
    Err(msg) => {
      if msg != "help" {
        println("error: " + msg)
      }
      print_usage(program)
      return
    }
  }
  if cfg.sweep {
    run_sweep(cfg)
  } else {
    run_bench(cfg)
  }
}
