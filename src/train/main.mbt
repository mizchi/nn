///|
fn mnist_path(base : String, name : String) -> String {
  base + "/" + name
}

///|
async fn train_main() -> Unit {
  let base = "data/mnist"
  let train_images = mnist_path(base, "train-images-idx3-ubyte")
  let train_labels = mnist_path(base, "train-labels-idx1-ubyte")
  let test_images = mnist_path(base, "t10k-images-idx3-ubyte")
  let test_labels = mnist_path(base, "t10k-labels-idx1-ubyte")

  let train_dataset = @mnist.mnist_load_mlp_dataset(train_images, train_labels, None)
  let test_dataset = @mnist.mnist_load_mlp_dataset(test_images, test_labels, None)

  let train = match train_dataset {
    Ok(ds) => ds
    Err(err) => {
      println("mnist train load error: " + err.to_string())
      return
    }
  }
  let test_set = match test_dataset {
    Ok(ds) => ds
    Err(err) => {
      println("mnist test load error: " + err.to_string())
      return
    }
  }

  let cfg = @nn.mlp_train_config(
    20,
    128,
    Float::from_int(1) / Float::from_int(10),
    true,
    0,
  )

  let spec = match @nn.mlp_spec_new(784, 128, 10, cfg.batch_size) {
    Ok(s) => s
    Err(err) => {
      println("spec error: " + err.to_string())
      return
    }
  }

  let params = match @nn.mlp_init_params_with_policy(
    spec,
    cfg.seed,
    @nn.mlp_init_policy_he_uniform,
  ) {
    Ok(p) => p
    Err(err) => {
      println("init error: " + err.to_string())
      return
    }
  }

  println("train start: samples=" + train.count.to_string())
  let train_result = @nn.mlp_train(spec, params, train, cfg)
  let trained = match train_result {
    Ok(r) => r
    Err(err) => {
      println("train error: " + err.to_string())
      return
    }
  }

  for i = 0; i < trained.metrics.length(); i = i + 1 {
    let m = trained.metrics[i]
    println(
      "epoch " +
      (i + 1).to_string() +
      " loss=" +
      m.loss.to_string() +
      " acc=" +
      m.accuracy.to_string()
    )
  }

  let eval = @nn.mlp_eval(spec, trained.params, test_set)
  match eval {
    Ok(m) =>
      println(
        "test loss=" + m.loss.to_string() + " acc=" + m.accuracy.to_string(),
      )
    Err(err) => println("eval error: " + err.to_string())
  }

  let exists_result : Result[Bool, Error] = try? @afs.exists(base)
  match exists_result {
    Ok(true) => ()
    Ok(false) => {
      let mkdir_result : Result[Unit, Error] =
        try? @afs.mkdir(base, permission=0o755, recursive=true)
      match mkdir_result {
        Ok(_) => ()
        Err(err) => println("mkdir error: " + err.to_string())
      }
    }
    Err(err) => println("exists error: " + err.to_string())
  }

  let bytes = @nn.mlp_params_to_bytes(spec, trained.params)
  match bytes {
    Ok(data) => {
      let write_result : Result[Unit, Error] = try? @afs.write_file(
        mnist_path(base, "mlp_784_128_10.bin"),
        data,
        create=0o644,
        truncate=true,
      )
      match write_result {
        Ok(_) => println("saved weights: data/mnist/mlp_784_128_10.bin")
        Err(err) => println("save error: " + err.to_string())
      }
    }
    Err(err) => println("serialize error: " + err.to_string())
  }
}

///|
fn main {
  @async.run_async_main(train_main)
}
