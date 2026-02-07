///|
test "checkpoint_path_with_step_inserts_before_extension" {
  let actual = checkpoint_path_with_step("/tmp/model.ckpt", 42)
  inspect(actual, content="/tmp/model.step-00000042.ckpt")
}

///|
test "checkpoint_path_with_step_without_extension" {
  let actual = checkpoint_path_with_step("checkpoints/model", 7)
  inspect(actual, content="checkpoints/model.step-00000007")
}

///|
test "jsonl_step_event_line_is_valid_json" {
  let line = jsonl_step_event_line(
    12,
    Float::from_double(1.5),
    Float::from_double(4.2),
    Float::from_double(3.0),
    Float::from_double(1024.0),
  )
  let parsed = @json.parse(line.to_string_view()) catch {
    err => fail("parse error: " + err.to_string())
  }
  let obj : Map[String, Json] = @json.from_json(parsed) catch {
    err => fail("decode map error: " + err.to_string())
  }
  let kind : String = @json.from_json(obj["kind"]) catch {
    err => fail("decode kind error: " + err.to_string())
  }
  let step : Int = @json.from_json(obj["step"]) catch {
    err => fail("decode step error: " + err.to_string())
  }
  inspect(kind, content="step")
  inspect(step, content="12")
}

///|
test "parse_prompt_ids_csv_accepts_comma_separated_ints" {
  let ids = match parse_prompt_ids_csv("11, 22,33") {
    Ok(v) => v
    Err(err) => fail("parse prompt ids: " + err)
  }
  inspect(ids.length(), content="3")
  inspect(ids[0], content="11")
  inspect(ids[1], content="22")
  inspect(ids[2], content="33")
}

///|
test "parse_args_preview_requires_resume_checkpoint" {
  let args = ["transformer-bench", "--preview-prompt-ids", "1,2,3"]
  match parse_args(args) {
    Ok(_) => fail("expected parse_args to fail without --resume-checkpoint")
    Err(msg) =>
      inspect(msg, content="--preview-prompt-ids requires --resume-checkpoint")
  }
}

///|
test "parse_args_preview_parses_prompt_ids_and_new_tokens" {
  let args = [
    "transformer-bench", "--preview-prompt-ids", "7,8,9", "--preview-new-tokens",
    "16", "--preview-temperature", "0.7", "--resume-checkpoint", "/tmp/model.ckpt",
  ]
  let cfg = match parse_args(args) {
    Ok(v) => v
    Err(msg) => fail("parse_args failed: " + msg)
  }
  let prompt_ids = match cfg.preview_prompt_ids {
    Some(v) => v
    None => fail("preview_prompt_ids was None")
  }
  inspect(prompt_ids.length(), content="3")
  inspect(prompt_ids[0], content="7")
  inspect(prompt_ids[1], content="8")
  inspect(prompt_ids[2], content="9")
  inspect(cfg.preview_new_tokens, content="16")
}

///|
test "parse_args_preview_parses_prompt_text" {
  let args = [
    "transformer-bench", "--preview-prompt-text", "hello world", "--resume-checkpoint",
    "/tmp/model.ckpt",
  ]
  let cfg = match parse_args(args) {
    Ok(v) => v
    Err(msg) => fail("parse_args failed: " + msg)
  }
  let prompt_text = match cfg.preview_prompt_text {
    Some(v) => v
    None => fail("preview_prompt_text was None")
  }
  inspect(prompt_text, content="hello world")
}

///|
test "parse_args_preview_rejects_prompt_ids_and_text_together" {
  let args = [
    "transformer-bench", "--preview-prompt-ids", "1,2,3", "--preview-prompt-text",
    "hello", "--resume-checkpoint", "/tmp/model.ckpt",
  ]
  match parse_args(args) {
    Ok(_) =>
      fail("expected parse_args to fail when both prompt options are set")
    Err(msg) =>
      inspect(
        msg,
        content="--preview-prompt-ids and --preview-prompt-text are mutually exclusive",
      )
  }
}

///|
test "parse_args_preview_text_rejects_shard_meta" {
  let args = [
    "transformer-bench", "--preview-prompt-text", "hello", "--shard-meta", "/tmp/meta.json",
    "--resume-checkpoint", "/tmp/model.ckpt",
  ]
  match parse_args(args) {
    Ok(_) => fail("expected parse_args to fail for prompt-text with shard-meta")
    Err(msg) =>
      inspect(
        msg,
        content="--preview-prompt-text currently requires default corpus (no --shard-meta)",
      )
  }
}
