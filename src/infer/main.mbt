///|
fn mnist_path(base : String, name : String) -> String {
  base + "/" + name
}

///|
struct InferArgs {
  weights_path : String
  limit : Int?
} derive(Show, Eq)

fn infer_args_default() -> InferArgs {
  let base = "data/mnist"
  { weights_path: mnist_path(base, "mlp_784_128_10.bin"), limit: None }
}

fn print_usage(program : String) -> Unit {
  println("Usage: " + program + " [--weights PATH] [--limit N]")
  println("  --weights PATH   weights file (default data/mnist/mlp_784_128_10.bin)")
  println("  --limit N        evaluate first N samples")
}

fn parse_limit(value : String) -> Result[Int, String] {
  let parsed = try? @strconv.parse_int(value)
  match parsed {
    Ok(v) =>
      if v > 0 {
        Ok(v)
      } else {
        Err("limit must be > 0")
      }
    Err(err) => Err("invalid limit: " + err.to_string())
  }
}

fn parse_args(args : Array[String]) -> Result[InferArgs, String] {
  let default = infer_args_default()
  let mut weights_path = default.weights_path
  let mut limit = default.limit
  let mut i = 1
  while i < args.length() {
    let arg = args[i]
    if arg == "--help" || arg == "-h" {
      return Err("help")
    }
    if arg == "--weights" {
      if i + 1 >= args.length() {
        return Err("missing value for --weights")
      }
      weights_path = args[i + 1]
      i = i + 2
      continue
    }
    if arg.has_prefix("--weights=") {
      let view = arg.sub(start=10) catch {
        _ => return Err("invalid --weights value")
      }
      weights_path = view.to_string()
      i = i + 1
      continue
    }
    if arg == "--limit" {
      if i + 1 >= args.length() {
        return Err("missing value for --limit")
      }
      match parse_limit(args[i + 1]) {
        Ok(v) => limit = Some(v)
        Err(msg) => return Err(msg)
      }
      i = i + 2
      continue
    }
    if arg.has_prefix("--limit=") {
      let view = arg.sub(start=8) catch {
        _ => return Err("invalid --limit value")
      }
      let value = view.to_string()
      match parse_limit(value) {
        Ok(v) => limit = Some(v)
        Err(msg) => return Err(msg)
      }
      i = i + 1
      continue
    }
    return Err("unknown option: " + arg)
  }
  Ok({ weights_path: weights_path, limit: limit })
}

///|
async fn infer_main() -> Unit {
  let args = @env.args()
  let program =
    if args.length() > 0 { args[0] } else { "mnist-infer" }
  let cfg_result = parse_args(args)
  let cfg = match cfg_result {
    Ok(c) => c
    Err(msg) => {
      if msg != "help" {
        println("error: " + msg)
      }
      print_usage(program)
      return
    }
  }

  let base = "data/mnist"
  let weights_path = cfg.weights_path

  let exists_result : Result[Bool, Error] = try? @afs.exists(weights_path)
  match exists_result {
    Ok(true) => ()
    Ok(false) => {
      println("weights not found: " + weights_path)
      return
    }
    Err(err) => {
      println("exists error: " + err.to_string())
      return
    }
  }

  let test_images = mnist_path(base, "t10k-images-idx3-ubyte")
  let test_labels = mnist_path(base, "t10k-labels-idx1-ubyte")

  let test_dataset = @mnist.mnist_load_mlp_dataset(
    test_images,
    test_labels,
    cfg.limit,
  )
  let test_set = match test_dataset {
    Ok(ds) => ds
    Err(err) => {
      println("mnist test load error: " + err.to_string())
      return
    }
  }

  let spec = match @nn.mlp_spec_new(784, 128, 10, 1) {
    Ok(s) => s
    Err(err) => {
      println("spec error: " + err.to_string())
      return
    }
  }

  let data_result = try? @afs.read_file(weights_path)
  let data = match data_result {
    Ok(d) => d
    Err(err) => {
      println("read error: " + err.to_string())
      return
    }
  }

  let bytes = data.binary()
  let eval = @nn.mlp_eval_from_bytes(spec, bytes, test_set)
  match eval {
    Ok(m) =>
      println(
        "test loss=" + m.loss.to_string() + " acc=" + m.accuracy.to_string()
      )
    Err(err) => println("eval error: " + err.to_string())
  }
}

///|
fn main {
  @async.run_async_main(infer_main)
}
