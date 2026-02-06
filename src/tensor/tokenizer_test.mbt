///|
test "char_tokenizer_encode_decode_roundtrip" {
  let text = "hello world"
  let tok = char_tokenizer_from_text(text)
  let ids = tok.encode(text)
  let decoded = tok.decode(ids)
  inspect(decoded, content="hello world")
}

///|
test "char_tokenizer_vocab_size" {
  let tok = char_tokenizer_from_text("abcabc")
  inspect(tok.vocab_size(), content="3")
}

///|
test "char_tokenizer_sorted_order" {
  let tok = char_tokenizer_from_text("cab")
  // chars should be sorted: ['a', 'b', 'c']
  let ids = tok.encode("abc")
  inspect(ids[0], content="0")
  inspect(ids[1], content="1")
  inspect(ids[2], content="2")
}

///|
test "char_tokenizer_unknown_char_skipped" {
  let tok = char_tokenizer_from_text("ab")
  let ids = tok.encode("axb")
  // 'x' is not in vocab, should be skipped
  inspect(ids.length(), content="2")
}
