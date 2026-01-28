using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using LLama;
using LLama.Common;
using Newtonsoft.Json;

namespace JihadAttorney.Llama
{
    public class LlamaService
    {
        private const int EmbeddingDimensions = 384;

        private readonly List<Surah> _surahs = new();
        private List<VerseEmbedding>? _embeddings;
        private string? _selectedModelPath;

        public LlamaService()
        {
            LoadQuran();
            EnsureEmbeddings();
        }

        public IReadOnlyList<string> GetAvailableModels(string modelsRoot = "D:\\Models")
        {
            if (!Directory.Exists(modelsRoot))
            {
                return Array.Empty<string>();
            }

            return Directory
                .EnumerateFiles(modelsRoot, "*.gguf", SearchOption.AllDirectories)
                .OrderBy(p => p)
                .ToList();
        }

        public bool SelectModelByIndex(int index, string modelsRoot = "D:\\Models")
        {
            var models = GetAvailableModels(modelsRoot);
            if (index < 0 || index >= models.Count)
            {
                return false;
            }

            _selectedModelPath = models[index];
            return true;
        }

        // Fix for CS1503, CS1674, IDE0008 in PrepareAsync
        public async Task PrepareAsync(CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(_selectedModelPath))
            {
                throw new InvalidOperationException("Kein Modell ausgewählt. Bitte SelectModelByIndex aufrufen.");
            }

            EnsureEmbeddings();
            EnsureOpenClBackend();

            ModelParams @params = new ModelParams(_selectedModelPath)
            {
                ContextSize = 2048,
                GpuLayerCount = 99
            };

            using LLamaWeights weights = LLamaWeights.LoadFromFile(@params);
            using LLamaContext context = weights.CreateContext(@params);
            StatelessExecutor executor = new StatelessExecutor(weights, @params, null);

            InferenceParams warmupParams = new InferenceParams
            {
                MaxTokens = 1,
                AntiPrompts = Array.Empty<string>()
            };

            await foreach (var _ in executor
                .InferAsync("", warmupParams)
                .WithCancellation(cancellationToken))
            {
                break;
            }
        }

        public async Task<string> AnswerQuestionAsync(string question, string responseLanguage = "auto", int topK = 3, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(question))
            {
                throw new ArgumentException("Question must not be empty", nameof(question));
            }

            if (TryHandleSlashQuery(question, out var slashResponse))
            {
                return slashResponse;
            }

            if (string.IsNullOrWhiteSpace(_selectedModelPath))
            {
                throw new InvalidOperationException("No model selected. Call SelectModelByIndex first.");
            }

            EnsureEmbeddings();
            var queryEmbedding = Embed(question);
            var nearest = _embeddings!
                .Select(e => new { Embedding = e, Score = CosineSimilarity(queryEmbedding, e.Vector) })
                .OrderByDescending(x => x.Score)
                .Take(Math.Max(1, topK))
                .ToList();

            var contextBuilder = new StringBuilder();
            foreach (var item in nearest)
            {
                contextBuilder
                    .Append(item.Embedding.Reference)
                    .Append(" | score=")
                    .Append(item.Score.ToString("F3"))
                    .AppendLine()
                    .AppendLine(item.Embedding.Text)
                    .AppendLine();
            }

            var promptBuilder = new StringBuilder();
            promptBuilder.AppendLine("You are a concise assistant. Answer in your own words based only on the provided Qur'an context. When you refer to verses, cite them as surah:ayah numbers inside the answer. Keep quotes balanced and closed.");
            promptBuilder.AppendLine(BuildLanguageInstruction(responseLanguage));
            promptBuilder.AppendLine("Context:");
            promptBuilder.AppendLine(contextBuilder.ToString());
            promptBuilder.AppendLine("Question: " + question);
            promptBuilder.Append("Answer:");

            var answer = await RunLlamaAsync(promptBuilder.ToString(), cancellationToken);

            var references = BuildReferenceBlock(nearest.Select(n => n.Embedding.Reference));
            if (!string.IsNullOrWhiteSpace(references)) 
            {
                return string.Concat(answer, "\n\nReferenzen:\n", references);
            }

            return answer;
        }

        private string BuildReferenceBlock(IEnumerable<string> references)
        {
            var sb = new StringBuilder();
            foreach (var reference in references)
            {
                if (!TryFindVerse(reference, out var surah, out var verse))
                {
                    continue;
                }

                sb.Append("- ")
                  .Append(reference)
                  .Append(" | ")
                  .Append(surah!.Transliteration)
                  .AppendLine()
                  .AppendLine(verse!.Text)
                  .AppendLine(verse.Translation)
                  .AppendLine();
            }

            return sb.ToString().TrimEnd();
        }

        private bool TryFindVerse(string reference, out Surah? surah, out Verse? verse)
        {
            surah = null;
            verse = null;

            var parts = reference.Split(':');
            if (parts.Length != 2)
            {
                return false;
            }

            if (!int.TryParse(parts[0], out var surahId) || !int.TryParse(parts[1], out var verseId))
            {
                return false;
            }

            surah = _surahs.FirstOrDefault(s => s.Id == surahId);
            verse = surah?.Verses.FirstOrDefault(v => v.Id == verseId);
            return surah != null && verse != null;
        }

        private async Task<string> RunLlamaAsync(string prompt, CancellationToken cancellationToken)
        {
            EnsureOpenClBackend();

            var @params = new ModelParams(_selectedModelPath!)
            {
                ContextSize = 2048,
                GpuLayerCount = 99
            };

            var inferenceParams = new InferenceParams
            {
                MaxTokens = 512,
                AntiPrompts = new[] { "User:", "Question:" }
            };

            using var weights = LLamaWeights.LoadFromFile(@params);
            using var context = weights.CreateContext(@params);
            var executor = new StatelessExecutor(weights, @params, null);

            var sb = new StringBuilder();
            await foreach (var token in executor
                .InferAsync(prompt, inferenceParams)
                .WithCancellation(cancellationToken))
            {
                sb.Append(token);
            }

            return sb.ToString().Trim();
        }

        private void EnsureOpenClBackend()
        {
            // Default backend (CPU/native) is used; OpenCL package removed because it lacks gemma3 support.
        }

        private void LoadQuran()
        {
            if (_surahs.Count > 0)
            {
                return;
            }

            string json;

            var resourceName = Assembly
                .GetExecutingAssembly()
                .GetManifestResourceNames()
                .FirstOrDefault(n => n.EndsWith("quran_en.json", StringComparison.OrdinalIgnoreCase));

            if (resourceName != null)
            {
                using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resourceName);
                using var reader = new StreamReader(stream!);
                json = reader.ReadToEnd();
            }
            else
            {
                var fallbackPath = Path.Combine(AppContext.BaseDirectory, "quran_en.json");
                json = File.ReadAllText(fallbackPath);
            }

            var data = JsonConvert.DeserializeObject<List<Surah>>(json);
            if (data != null)
            {
                _surahs.AddRange(data);
            }
        }

        private void EnsureEmbeddings()
        {
            if (_embeddings != null)
            {
                return;
            }

            var embeddings = new List<VerseEmbedding>();
            foreach (var surah in _surahs)
            {
                foreach (var verse in surah.Verses)
                {
                    var text = $"{surah.Transliteration} ({surah.Id}:{verse.Id}) - {verse.Text} / {verse.Translation}";
                    embeddings.Add(new VerseEmbedding
                    {
                        Reference = $"{surah.Id}:{verse.Id}",
                        Text = text,
                        Vector = Embed(text)
                    });
                }
            }

            _embeddings = embeddings;
        }

        private static float[] Embed(string text)
        {
            var vector = new float[EmbeddingDimensions];
            if (string.IsNullOrWhiteSpace(text))
            {
                return vector;
            }

            var tokens = Tokenize(text);
            foreach (var token in tokens)
            {
                var idx = Math.Abs(token.GetHashCode()) % EmbeddingDimensions;
                vector[idx] += 1f;
            }

            Normalize(vector);
            return vector;
        }

        private static IEnumerable<string> Tokenize(string text)
        {
            var sb = new StringBuilder();
            foreach (var ch in text.ToLowerInvariant())
            {
                if (char.IsLetterOrDigit(ch))
                {
                    sb.Append(ch);
                }
                else if (sb.Length > 0)
                {
                    yield return sb.ToString();
                    sb.Clear();
                }
            }

            if (sb.Length > 0)
            {
                yield return sb.ToString();
            }
        }

        private static void Normalize(IList<float> vector)
        {
            var length = MathF.Sqrt(vector.Sum(v => v * v));
            if (length <= 0)
            {
                return;
            }

            for (var i = 0; i < vector.Count; i++)
            {
                vector[i] /= length;
            }
        }

        private static float CosineSimilarity(IReadOnlyList<float> a, IReadOnlyList<float> b)
        {
            var sum = 0f;
            for (var i = 0; i < Math.Min(a.Count, b.Count); i++)
            {
                sum += a[i] * b[i];
            }

            return sum;
        }

        private bool TryHandleSlashQuery(string question, out string response)
        {
            response = string.Empty;

            var trimmed = question.Trim();
            if (!trimmed.StartsWith("/"))
            {
                return false;
            }

            var payload = trimmed[1..].Trim();
            if (string.IsNullOrWhiteSpace(payload))
            {
                return false;
            }

            var surahPart = payload;
            var versePart = string.Empty;

            var colonIndex = payload.IndexOf(':');
            if (colonIndex >= 0)
            {
                surahPart = payload[..colonIndex];
                versePart = payload[(colonIndex + 1)..];
            }

            if (!int.TryParse(surahPart, out var surahId))
            {
                return false;
            }

            var surah = _surahs.FirstOrDefault(s => s.Id == surahId);
            if (surah == null)
            {
                response = $"Sure {surahId} nicht gefunden.";
                return true;
            }

            IReadOnlyList<int> versesToReturn;

            if (string.IsNullOrWhiteSpace(versePart))
            {
                versesToReturn = surah.Verses.Select(v => v.Id).ToList();
            }
            else if (versePart.Contains('-'))
            {
                var rangeParts = versePart.Split('-', 2);
                if (rangeParts.Length != 2 ||
                    !int.TryParse(rangeParts[0], out var start) ||
                    !int.TryParse(rangeParts[1], out var end) ||
                    start <= 0 || end < start)
                {
                    response = $"Ungültiger Vers-Bereich: {versePart}";
                    return true;
                }

                if (start > surah.Verses.Count || end > surah.Verses.Count)
                {
                    response = surah.Transliteration;
                    return true;
                }

                versesToReturn = surah.Verses
                    .Where(v => v.Id >= start && v.Id <= end)
                    .Select(v => v.Id)
                    .ToList();
            }
            else if (int.TryParse(versePart, out var singleVerse))
            {
                if (singleVerse <= 0 || singleVerse > surah.Verses.Count)
                {
                    response = surah.Transliteration;
                    return true;
                }

                versesToReturn = new[] { singleVerse };
            }
            else
            {
                response = $"Ungültige Eingabe nach Sure: {versePart}";
                return true;
            }

            var sb = new StringBuilder();
            sb.AppendLine($"{surah.Transliteration} ({surah.Id})");

            foreach (var verseId in versesToReturn)
            {
                var verse = surah.Verses.First(v => v.Id == verseId);
                sb.AppendLine($"{surah.Id}:{verse.Id} | {verse.Text} / {verse.Translation}");
            }

            response = sb.ToString().TrimEnd();
            return true;
        }

        private static string BuildLanguageInstruction(string responseLanguage)
        {
            if (string.IsNullOrWhiteSpace(responseLanguage) || responseLanguage.Equals("auto", StringComparison.OrdinalIgnoreCase))
            {
                return "Detect the language of the question and answer in that language.";
            }

            return $"Answer in {responseLanguage}. If the question is in a different language, still respond in {responseLanguage}.";
        }

        private class Surah
        {
            [JsonProperty("id")]
            public int Id { get; set; }

            [JsonProperty("transliteration")]
            public string Transliteration { get; set; } = string.Empty;

            [JsonProperty("verses")]
            public List<Verse> Verses { get; set; } = new();
        }

        private class Verse
        {
            [JsonProperty("id")]
            public int Id { get; set; }

            [JsonProperty("text")]
            public string Text { get; set; } = string.Empty;

            [JsonProperty("translation")]
            public string Translation { get; set; } = string.Empty;
        }

        private class VerseEmbedding
        {
            public string Reference { get; set; } = string.Empty;
            public string Text { get; set; } = string.Empty;
            public float[] Vector { get; set; } = Array.Empty<float>();
        }
    }
}
