///|
/// Character-level tokenizer
pub struct CharTokenizer {
  chars : Array[Char]
}

///|
/// Build a character tokenizer from text
pub fn char_tokenizer_from_text(text : String) -> CharTokenizer {
  let text_chars = text.to_array()
  // Collect unique characters
  let seen : Array[Char] = []
  for i = 0; i < text_chars.length(); i = i + 1 {
    let c = text_chars[i]
    let mut found = false
    for j = 0; j < seen.length(); j = j + 1 {
      if seen[j] == c {
        found = true
        break
      }
    }
    if !found {
      seen.push(c)
    }
  }
  // Sort characters by code point for deterministic ordering
  seen.sort_by(fn(a, b) { a.to_int().compare(b.to_int()) })
  CharTokenizer::{ chars: seen }
}

///|
pub fn CharTokenizer::vocab_size(self : CharTokenizer) -> Int {
  self.chars.length()
}

///|
pub fn CharTokenizer::encode(self : CharTokenizer, text : String) -> Array[Int] {
  let text_chars = text.to_array()
  let result : Array[Int] = []
  for i = 0; i < text_chars.length(); i = i + 1 {
    let c = text_chars[i]
    let mut idx = -1
    for j = 0; j < self.chars.length(); j = j + 1 {
      if self.chars[j] == c {
        idx = j
        break
      }
    }
    if idx >= 0 {
      result.push(idx)
    }
  }
  result
}

///|
pub fn CharTokenizer::decode(self : CharTokenizer, ids : Array[Int]) -> String {
  let buf = StringBuilder::new()
  for i = 0; i < ids.length(); i = i + 1 {
    let id = ids[i]
    if id >= 0 && id < self.chars.length() {
      buf.write_char(self.chars[id])
    }
  }
  buf.to_string()
}
