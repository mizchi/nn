///|
fn read_u32_be(buf : Bytes, offset : Int) -> Int {
  (buf[offset].to_int() << 24) | (buf[offset + 1].to_int() << 16) | (buf[offset + 2].to_int() << 8) | buf[offset + 3].to_int()
}

///|
fn load_images(buf : Bytes, limit : Int?) -> (Array[Float], Int) {
  let n_raw = read_u32_be(buf, 4)
  let rows = read_u32_be(buf, 8)
  let cols = read_u32_be(buf, 12)
  let img_size = rows * cols
  let n = match limit {
    Some(l) => if l < n_raw { l } else { n_raw }
    None => n_raw
  }
  let data = Array::make(n * img_size, Float::from_int(0))
  for i = 0; i < n * img_size; i = i + 1 {
    data[i] = Float::from_int(buf[16 + i].to_int()) / Float::from_int(255)
  }
  (data, n)
}

///|
fn load_labels(buf : Bytes, limit : Int?) -> Array[Int] {
  let n_raw = read_u32_be(buf, 4)
  let n = match limit {
    Some(l) => if l < n_raw { l } else { n_raw }
    None => n_raw
  }
  let labels = Array::make(n, 0)
  for i = 0; i < n; i = i + 1 {
    labels[i] = buf[8 + i].to_int()
  }
  labels
}

///|
fn format_ms(ms : UInt64) -> String {
  let sec = ms / 1000UL
  let frac = ms % 1000UL
  if frac < 10UL {
    sec.to_string() + ".00" + frac.to_string()
  } else if frac < 100UL {
    sec.to_string() + ".0" + frac.to_string()
  } else {
    sec.to_string() + "." + frac.to_string()
  }
}

///|
async fn main_() -> Unit {
  let args = @env.args()
  let mut limit : Int? = Some(1000)
  let mut epochs = 5
  let mut batch_size = 64
  let mut profile = false
  for i = 1; i < args.length(); i = i + 1 {
    let arg = args[i]
    if arg.has_prefix("--limit=") {
      limit = Some(@strconv.parse_int(arg[8:].to_string()))
    } else if arg.has_prefix("--epochs=") {
      epochs = @strconv.parse_int(arg[9:].to_string())
    } else if arg.has_prefix("--batch-size=") {
      batch_size = @strconv.parse_int(arg[13:].to_string())
    } else if arg == "--profile" {
      profile = true
    }
  }
  let base = "data/mnist"
  let train_img_buf = (try! @afs.read_file(base + "/train-images-idx3-ubyte")).binary()
  let train_lbl_buf = (try! @afs.read_file(base + "/train-labels-idx1-ubyte")).binary()
  let test_img_buf = (try! @afs.read_file(base + "/t10k-images-idx3-ubyte")).binary()
  let test_lbl_buf = (try! @afs.read_file(base + "/t10k-labels-idx1-ubyte")).binary()
  let (train_data, train_count) = load_images(train_img_buf, limit)
  let train_labels = load_labels(train_lbl_buf, limit)
  let (test_data, test_count) = load_images(test_img_buf, None)
  let test_labels = load_labels(test_lbl_buf, None)
  let lr = Float::from_double(0.001)
  let config = @tensor.vit_config(28, 7, 64, 4, 2, 128, 10).unwrap()
  let img_dim = 784
  let nc = 10
  println(
    "Train: " + train_count.to_string() + " samples, Test: " + test_count.to_string() + " samples",
  )
  println(
    "Config: epochs=" + epochs.to_string() + ", batch_size=" + batch_size.to_string() + ", lr=0.001",
  )
  println("ViT: image=28, patch=7, d_model=64, heads=4, layers=2, d_ff=128")
  println("")
  if profile {
    run_profile(train_data, train_count, train_labels, config, batch_size, lr)
    return
  }
  let train_images = @tensor.tensor_new(
    @tensor.shape_new_unchecked([train_count, img_dim]),
    train_data,
  ).unwrap()
  let ic = train_images.contiguous()
  // Initialize params and grads
  let params = @tensor.vit_init_params(config, 42)
  let grads = @tensor.vit_zero_grads(config)
  let mut rng = 42
  let mut total_train_ms = 0UL
  for epoch = 0; epoch < epochs; epoch = epoch + 1 {
    // Shuffle indices
    let indices = Array::makei(train_count, fn(i) { i })
    for i = train_count - 1; i > 0; i = i - 1 {
      rng = rng * 1103515245 + 12345
      let j = (rng / 65536 % 32768).abs() % (i + 1)
      let tmp = indices[i]
      indices[i] = indices[j]
      indices[j] = tmp
    }
    let mut epoch_loss = Float::from_int(0)
    let mut epoch_correct = 0
    let mut epoch_samples = 0
    let t0 = @env.now()
    let mut offset = 0
    while offset < train_count {
      let bs = if offset + batch_size <= train_count {
        batch_size
      } else {
        train_count - offset
      }
      let batch_data = Array::make(bs * img_dim, Float::from_int(0))
      let batch_labels : Array[Int] = Array::make(bs, 0)
      for b = 0; b < bs; b = b + 1 {
        let idx = indices[offset + b]
        let src = idx * img_dim
        let dst = b * img_dim
        for d = 0; d < img_dim; d = d + 1 {
          batch_data[dst + d] = ic.data[src + d]
        }
        batch_labels[b] = train_labels[idx]
      }
      let batch_images = @tensor.tensor_new(
        @tensor.shape_new_unchecked([bs, img_dim]),
        batch_data,
      ).unwrap()
      @tensor.vit_zero_grads_inplace(grads)
      let (logits, cache) = @tensor.vit_forward_with_cache(
        batch_images,
        params,
        config,
      )
      let loss = @tensor.vit_backward(cache, params, config, grads, batch_labels)
      @tensor.vit_sgd_update(params, grads, lr)
      epoch_loss = epoch_loss + loss * Float::from_int(bs)
      let lc = logits.contiguous()
      for b = 0; b < bs; b = b + 1 {
        let lbase = b * nc
        let mut max_idx = 0
        let mut max_val = lc.data[lbase]
        for c = 1; c < nc; c = c + 1 {
          if lc.data[lbase + c] > max_val {
            max_val = lc.data[lbase + c]
            max_idx = c
          }
        }
        if max_idx == batch_labels[b] {
          epoch_correct = epoch_correct + 1
        }
      }
      epoch_samples = epoch_samples + bs
      offset = offset + bs
    }
    let t1 = @env.now()
    let epoch_ms = t1 - t0
    total_train_ms = total_train_ms + epoch_ms
    let avg_loss = epoch_loss / Float::from_int(epoch_samples)
    let avg_acc = Float::from_int(epoch_correct) / Float::from_int(epoch_samples)
    println(
      "Epoch " + (epoch + 1).to_string() + "/" + epochs.to_string() +
      ": loss=" + avg_loss.to_string() +
      " train_acc=" + avg_acc.to_string() +
      " time=" + format_ms(epoch_ms) + "s",
    )
  }
  println("")
  println("Total training time: " + format_ms(total_train_ms) + "s")
  // Test evaluation
  let test_images = @tensor.tensor_new(
    @tensor.shape_new_unchecked([test_count, img_dim]),
    test_data,
  ).unwrap()
  let eval_batch = 256
  let mut total_correct = 0
  let mut eval_offset = 0
  let tc = test_images.contiguous()
  while eval_offset < test_count {
    let bs = if eval_offset + eval_batch <= test_count {
      eval_batch
    } else {
      test_count - eval_offset
    }
    let batch_data = Array::make(bs * img_dim, Float::from_int(0))
    let batch_labels : Array[Int] = Array::make(bs, 0)
    for b = 0; b < bs; b = b + 1 {
      let src = (eval_offset + b) * img_dim
      let dst = b * img_dim
      for d = 0; d < img_dim; d = d + 1 {
        batch_data[dst + d] = tc.data[src + d]
      }
      batch_labels[b] = test_labels[eval_offset + b]
    }
    let batch_tensor = @tensor.tensor_new(
      @tensor.shape_new_unchecked([bs, img_dim]),
      batch_data,
    ).unwrap()
    match @tensor.vit_eval(batch_tensor, params, config, batch_labels, bs) {
      Ok((_loss, acc)) => {
        let correct_in_batch = ((acc * Float::from_int(bs)).to_double() + 0.5)
          |> Double::to_int
        total_correct = total_correct + correct_in_batch
      }
      Err(err) => {
        println("eval error: " + err)
        return
      }
    }
    eval_offset = eval_offset + bs
  }
  let test_acc = Float::from_int(total_correct) / Float::from_int(test_count)
  println("Test accuracy: " + test_acc.to_string())
}

///|
fn run_profile(
  train_data : Array[Float],
  train_count : Int,
  train_labels : Array[Int],
  config : @tensor.VitConfig,
  batch_size : Int,
  lr : Float,
) -> Unit {
  let img_dim = 784
  let params = @tensor.vit_init_params(config, 42)
  let grads = @tensor.vit_zero_grads(config)
  let train_images = @tensor.tensor_new(
    @tensor.shape_new_unchecked([train_count, img_dim]),
    train_data,
  ).unwrap()
  let ic = train_images.contiguous()
  // Run profiling over batches
  let num_steps = if train_count / batch_size > 0 {
    train_count / batch_size
  } else {
    1
  }
  let results : Array[@tensor.VitProfileResult] = []
  let mut offset = 0
  for step = 0; step < num_steps; step = step + 1 {
    let bs = if offset + batch_size <= train_count {
      batch_size
    } else {
      train_count - offset
    }
    if bs <= 0 {
      break
    }
    let batch_data = Array::make(bs * img_dim, Float::from_int(0))
    let batch_labels : Array[Int] = Array::make(bs, 0)
    for b = 0; b < bs; b = b + 1 {
      let src = (offset + b) * img_dim
      let dst = b * img_dim
      for d = 0; d < img_dim; d = d + 1 {
        batch_data[dst + d] = ic.data[src + d]
      }
      batch_labels[b] = train_labels[offset + b]
    }
    let batch_images = @tensor.tensor_new(
      @tensor.shape_new_unchecked([bs, img_dim]),
      batch_data,
    ).unwrap()
    @tensor.vit_zero_grads_inplace(grads)
    let r = @tensor.vit_profile_step(
      batch_images, batch_labels, params, grads, config, lr,
    )
    results.push(r)
    offset = offset + bs
  }
  println("Profiled " + results.length().to_string() + " steps (batch_size=" + batch_size.to_string() + ")")
  println("")
  let avg = @tensor.vit_profile_aggregate(results)
  @tensor.vit_profile_result_print(avg)
}

///|
async fn main {
  main_()
}
