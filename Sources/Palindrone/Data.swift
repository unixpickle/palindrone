import Foundation
import Honeycrisp

public class Dataset {

  public struct State: Codable {
    public let currentEpoch: Int
    public let currentDocOffset: Int
  }

  public enum DatasetError: Error {
    case datasetIsEmpty
  }

  public let directory: String
  public let minChunkLength: Int
  public let maxChunkLength: Int
  public let textFilenames: [String]

  private var currentEpoch: Int = 0
  private var currentDocOffset: Int = 0
  private var allWords: [[UInt8]]? = nil
  private var docsInEpoch: [[UInt8]]? = nil

  public var state: State {
    get {
      State(
        currentEpoch: currentEpoch, currentDocOffset: currentDocOffset
      )
    }
    set {
      currentEpoch = newValue.currentEpoch
      currentDocOffset = newValue.currentDocOffset
      docsInEpoch = nil
    }
  }

  public init(directory: String, minChunkLength: Int, maxChunkLength: Int) throws {
    self.directory = directory
    self.minChunkLength = minChunkLength
    self.maxChunkLength = maxChunkLength

    let fileManager = FileManager.default
    let files = try fileManager.contentsOfDirectory(
      at: URL(filePath: directory), includingPropertiesForKeys: nil)
    textFilenames =
      files
      .filter { $0.pathExtension == "txt" }
      .map { $0.deletingPathExtension().lastPathComponent }
      .sorted()
  }

  public func next() async throws -> [UInt8] {
    let allDocs = try await epochDocs()
    if currentDocOffset >= allDocs.count {
      if currentDocOffset == 0 {
        throw DatasetError.datasetIsEmpty
      }
      currentEpoch += 1
      currentDocOffset = 0
      docsInEpoch = nil
      return try await next()
    }

    currentDocOffset += 1
    return allDocs[currentDocOffset - 1]
  }

  private func epochDocs() async throws -> [[UInt8]] {
    if let docs = docsInEpoch {
      return docs
    }

    let words = try loadWords()

    let rng = Backend.current.createRandom()
    rng.seed(currentEpoch)

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

    var chunks = [[UInt8]]()
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

    // Global shuffle of documents
    let indices = try await Tensor(
      randPerm: [chunks.count],
      generator: rng
    ).ints()
    docsInEpoch = indices.map { chunks[$0] }

    return docsInEpoch!
  }

  private func loadWords() throws -> [[UInt8]] {
    if let result = allWords {
      return result
    }
    let fileURL = URL(filePath: directory)
    let allURLs = textFilenames.map { fileURL.appending(component: $0) }
    allWords = try allURLs.map { url in
      try String(contentsOf: url, encoding: .utf8)
    }.joined(separator: "\n").lowercased().components(separatedBy: .whitespacesAndNewlines)
      .compactMap { word in
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
    return allWords!
  }

}
