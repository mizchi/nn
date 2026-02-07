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
