import HCBacktrace
import Honeycrisp

public class Tokenizer {
  public let maxBytes: Int

  public var inputCount: Int {
    maxBytes + 1
  }

  public var vocabSize: Int {
    maxBytes + 256 + 1
  }

  public init(maxBytes: Int) {
    self.maxBytes = maxBytes
  }

  @recordCaller
  private func _tokenize(sequence: [UInt8]) -> (inputs: Tensor, targets: Tensor, mask: Tensor) {
    #alwaysAssert(
      sequence.count <= maxBytes,
      "sequence length \(sequence) is greater than maximum \(maxBytes)")
    var fullSeq = [0, sequence.count + 256] + sequence.map(Int.init)
    while fullSeq.count < maxBytes + 2 {
      fullSeq.append(0)
    }
    let inputs = Tensor(data: fullSeq[..<(fullSeq.count - 1)])
    let targets = Tensor(data: fullSeq[1...])
    let mask = Tensor(
      data: Array(repeating: true, count: sequence.count + 1)
        + Array(repeating: false, count: inputCount - (sequence.count + 1)))
    precondition(inputs.shape == targets.shape)
    precondition(inputs.shape == mask.shape)
    return (inputs: inputs, targets: targets, mask: mask)
  }
}
