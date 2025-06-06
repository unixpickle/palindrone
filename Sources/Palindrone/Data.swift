import Foundation
import HCBacktrace
import Honeycrisp

public class Dataset {

  public struct State: Codable {
    public let buffer: [[UInt8]]
    public let rngState: TensorState
    public let nextFile: Int
  }

  public enum DatasetError: Error {
    case couldNotEnumerate
    case reachedEndOfData
  }

  public let directory: String
  public let minChunkLength: Int
  public let maxChunkLength: Int
  public let textFilenames: [String]
  private let bufferSize: Int
  private let rng: RandomGenerator

  private var nextFile: Int = 0
  private var buffer: [[UInt8]] = []
  private var currentBufferSize: Int = 0

  @recordCaller
  private func _state() async throws -> State {
    return State(
      buffer: buffer,
      rngState: try await rng.state.state(),
      nextFile: nextFile
    )
  }

  public init(
    directory: String,
    minChunkLength: Int,
    maxChunkLength: Int,
    bufferSize: Int,
    state: State? = nil
  ) throws {
    self.directory = directory
    self.minChunkLength = minChunkLength
    self.maxChunkLength = maxChunkLength
    self.bufferSize = bufferSize
    self.rng = Backend.current.createRandom()
    if let state = state {
      self.nextFile = state.nextFile
      self.buffer = state.buffer
      self.rng.state = Tensor(state: state.rngState)
    } else {
      self.rng.seed(0)
    }

    let fileManager = FileManager.default
    let baseURL = URL(filePath: directory).absoluteURL
    guard
      let enumerator = fileManager.enumerator(
        at: baseURL,
        includingPropertiesForKeys: nil,
        options: [.skipsHiddenFiles, .skipsPackageDescendants]
      )
    else {
      throw DatasetError.couldNotEnumerate
    }

    print("listing files in \(directory) ...")
    textFilenames =
      enumerator
      .compactMap { $0 as? URL }
      .filter { $0.pathExtension.lowercased() == "txt" }
      .map { $0.path.replacingOccurrences(of: baseURL.path + "/", with: "") }
      .sorted()
    print("listed \(textFilenames.count) dataset files")
  }

  public func next() async throws -> [UInt8] {
    if currentBufferSize == 0 {
      print("filling dataset buffer from scratch...")
    }
    while currentBufferSize < bufferSize {
      try await readNextFile()
    }
    let idx = try await Tensor(randInt: [1], in: 0..<Int64(buffer.count), generator: rng)
      .item(Int.self)
    let result = buffer.remove(at: idx)
    currentBufferSize -= result.count
    return result
  }

  @recordCaller
  internal func _readNextFile() async throws {
    if nextFile >= textFilenames.count {
      throw DatasetError.reachedEndOfData
    }
    let fileURL = URL(filePath: directory).appending(path: textFilenames[nextFile])
    nextFile += 1
    let paragraphs = try loadParagraphWords(fileURL)
    let seqs = try await paragraphsToSequences(
      paragraphs, rng: rng, minChunkLength: minChunkLength, maxChunkLength: maxChunkLength)
    buffer.append(contentsOf: seqs)
    currentBufferSize += seqs.reduce(0, { $0 + $1.count })
  }

}

private func paragraphsToSequences(
  _ paragraphs: [[[UInt8]]], rng: RandomGenerator, minChunkLength: Int, maxChunkLength: Int
) async throws -> [[UInt8]] {
  var chunks = [[UInt8]]()
  for words in paragraphs {
    let lengths = try await Tensor(
      randInt: [words.count],
      in: Int64(minChunkLength)..<(Int64(maxChunkLength) + 1),
      generator: rng
    ).ints()

    var wordOffset = 0
    func popWord() -> [UInt8]? {
      if wordOffset >= words.count {
        return nil
      } else {
        wordOffset += 1
        return words[wordOffset - 1]
      }
    }
    func unpopWord() {
      wordOffset -= 1
    }

    for length in lengths {
      var chunk = [[UInt8]]()
      var chunkSize = 0
      while chunkSize < length {
        guard let nextWord = popWord() else { break }
        if chunkSize + nextWord.count > length {
          if nextWord.count < minChunkLength {
            // Only return the word if we know it can be used by the
            // next document, otherwise we may skip an arbitrary number
            // of lengths until we find one that can fit the word.
            unpopWord()
          }
          break
        }
        chunkSize += nextWord.count
        chunk.append(nextWord)
      }
      chunks.append(chunk.flatMap { $0 })
      if wordOffset >= words.count {
        break
      }
    }
  }
  return chunks
}

private func loadParagraphWords(_ url: URL) throws -> [[[UInt8]]] {
  let allText = try String(contentsOf: url, encoding: .utf8)
  return allText.lowercased().components(separatedBy: .newlines).compactMap {
    paragraph -> [[UInt8]]? in
    let result = paragraph.components(separatedBy: .whitespaces).compactMap { word -> [UInt8]? in
      let result = word.compactMap { char -> UInt8? in
        guard let ascii = char.asciiValue, ascii >= 97 && ascii <= 122 else {
          return nil
        }
        return ascii
      }
      if result.isEmpty {
        return nil
      } else {
        return result
      }
    }
    if result.isEmpty {
      return nil
    } else {
      return result
    }
  }
}
