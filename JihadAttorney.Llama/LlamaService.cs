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
        private readonly List<HadithBook> _hadithBooks = new();
        private List<VerseEmbedding>? _embeddings;
        private string? _selectedModelPath;

        public LlamaService()
        {
            this.LoadQuran();
            this.LoadHadiths();
            this.EnsureEmbeddings();
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
            var models = this.GetAvailableModels(modelsRoot);
            if (index < 0 || index >= models.Count)
            {
                return false;
            }

            this._selectedModelPath = models[index];
            return true;
        }

        // Fix for CS1503, CS1674, IDE0008 in PrepareAsync
        public async Task PrepareAsync(CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(this._selectedModelPath))
            {
                throw new InvalidOperationException("Kein Modell ausgewählt. Bitte SelectModelByIndex aufrufen.");
            }

            this.EnsureEmbeddings();
            ModelParams @params = new ModelParams(this._selectedModelPath)
            {
                ContextSize = 1024,
                GpuLayerCount = 0
            };

            using LLamaWeights weights = LLamaWeights.LoadFromFile(@params);
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

            if (this.TryHandleSlashQuery(question, out var slashResponse))
            {
                return slashResponse;
            }

            if (string.IsNullOrWhiteSpace(this._selectedModelPath))
            {
                throw new InvalidOperationException("No model selected. Call SelectModelByIndex first.");
            }

            this.EnsureEmbeddings();
            var queryEmbedding = Embed(question);
            var nearest = this._embeddings!
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
            promptBuilder.AppendLine("You are a concise assistant. Answer in your own words based only on the provided Qur'an and hadith context. When you refer to verses, cite them as surah:ayah numbers inside the answer. When you refer to hadith, cite them as hadith:bookId:hadithId. Keep quotes balanced and closed.");
            promptBuilder.AppendLine(BuildLanguageInstruction(responseLanguage));
            promptBuilder.AppendLine("Context:");
            promptBuilder.AppendLine(contextBuilder.ToString());
            promptBuilder.AppendLine("Question: " + question);
            promptBuilder.Append("Answer:");

            var answer = await this.RunLlamaAsync(promptBuilder.ToString(), cancellationToken);

            var references = this.BuildReferenceBlock(nearest.Select(n => n.Embedding.Reference));
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
                if (this.TryFindVerse(reference, out var surah, out var verse))
                {
                    sb.Append("- ")
                      .Append(reference)
                      .Append(" | ")
                      .Append(surah!.Transliteration)
                      .AppendLine()
                      .AppendLine(verse!.Text)
                      .AppendLine(verse.Translation)
                      .AppendLine();
                    continue;
                }

                if (this.TryFindHadith(reference, out var book, out var hadith))
                {
                    var title = book.Metadata?.English?.Title ?? $"Book {book.Id}";
                    var narrator = hadith.English?.Narrator ?? string.Empty;
                    var text = hadith.English?.Text ?? string.Empty;
                    sb.Append("- ")
                      .Append(reference)
                      .Append(" | ")
                      .AppendLine(title)
                      .AppendLine(narrator)
                      .AppendLine(text)
                      .AppendLine();
                }
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

            surah = this._surahs.FirstOrDefault(s => s.Id == surahId);
            verse = surah?.Verses.FirstOrDefault(v => v.Id == verseId);
            return surah != null && verse != null;
        }

        private bool TryFindHadith(string reference, out HadithBook? book, out HadithEntry? hadith)
        {
            book = null;
            hadith = null;

            if (!reference.StartsWith("hadith:", StringComparison.OrdinalIgnoreCase))
            {
                return false;
            }

            var parts = reference.Split(':');
            if (parts.Length != 3)
            {
                return false;
            }

            if (!int.TryParse(parts[1], out var bookId) || !int.TryParse(parts[2], out var hadithId))
            {
                return false;
            }

            book = this._hadithBooks.FirstOrDefault(b => b.Id == bookId);
            hadith = book?.Hadiths.FirstOrDefault(h => h.IdInBook == hadithId);
            return book != null && hadith != null;
        }

        public async IAsyncEnumerable<string> AnswerQuestionStreamAsync(string question, string responseLanguage = "auto", int topK = 3, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(question))
            {
                throw new ArgumentException("Question must not be empty", nameof(question));
            }

            if (this.TryHandleSlashQuery(question, out var slashResponse))
            {
                yield return slashResponse;
                yield break;
            }

            if (string.IsNullOrWhiteSpace(this._selectedModelPath))
            {
                throw new InvalidOperationException("No model selected. Call SelectModelByIndex first.");
            }

            this.EnsureEmbeddings();
            var queryEmbedding = Embed(question);
            var nearest = this._embeddings!
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
            promptBuilder.AppendLine("You are a concise assistant. Answer in your own words based only on the provided Qur'an and hadith context. When you refer to verses, cite them as surah:ayah numbers inside the answer. When you refer to hadith, cite them as hadith:bookId:hadithId. Keep quotes balanced and closed. Answer in the language the user put his question.");
            promptBuilder.AppendLine(BuildLanguageInstruction(responseLanguage));
            promptBuilder.AppendLine("Context:");
            promptBuilder.AppendLine(contextBuilder.ToString());
            promptBuilder.AppendLine("Question: " + question);
            promptBuilder.Append("Answer:");

            await foreach (var token in this.RunLlamaStreamAsync(promptBuilder.ToString(), cancellationToken))
            {
                yield return token;
            }

            var references = this.BuildReferenceBlock(nearest.Select(n => n.Embedding.Reference));
            if (!string.IsNullOrWhiteSpace(references))
            {
                yield return "\n\nReferenzen:\n" + references;
            }
        }

        private async Task<string> RunLlamaAsync(string prompt, CancellationToken cancellationToken)
        {
            var sb = new StringBuilder();
            await foreach (var token in this.RunLlamaStreamAsync(prompt, cancellationToken))
            {
                sb.Append(token);
            }

            return sb.ToString().Trim();
        }

        private async IAsyncEnumerable<string> RunLlamaStreamAsync(string prompt, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
        {
            var @params = new ModelParams(this._selectedModelPath!)
            {
                ContextSize = 8192,
                GpuLayerCount = 999
            };

            var inferenceParams = new InferenceParams
            {
                MaxTokens = 4096,
                AntiPrompts = new[] { "User:", "Question:" }
            };

            using var weights = LLamaWeights.LoadFromFile(@params);
            var executor = new StatelessExecutor(weights, @params, null);

            await foreach (var token in executor
                .InferAsync(prompt, inferenceParams)
                .WithCancellation(cancellationToken))
            {
                yield return token;
            }
        }

        private void LoadQuran()
        {
            if (this._surahs.Count > 0)
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
                var fallbackPath = Path.Combine(AppContext.BaseDirectory, "Ressources", "quran_en.json");
                json = File.ReadAllText(fallbackPath);
            }

            var data = JsonConvert.DeserializeObject<List<Surah>>(json);
            if (data != null)
            {
                this._surahs.AddRange(data);
            }
        }

        private void LoadHadiths()
        {
            if (this._hadithBooks.Count > 0)
            {
                return;
            }

            var assembly = Assembly.GetExecutingAssembly();
            var resourceNames = assembly
                .GetManifestResourceNames()
                .Where(n => n.EndsWith(".json", StringComparison.OrdinalIgnoreCase)
                         && !n.EndsWith("quran_en.json", StringComparison.OrdinalIgnoreCase))
                .OrderBy(n => n)
                .ToList();

            foreach (var resourceName in resourceNames)
            {
                using var stream = assembly.GetManifestResourceStream(resourceName);
                if (stream == null)
                {
                    continue;
                }

                using var reader = new StreamReader(stream);
                var json = reader.ReadToEnd();

                var book = JsonConvert.DeserializeObject<HadithBook>(json);
                if (book != null)
                {
                    this._hadithBooks.Add(book);
                }
            }
        }

        private void EnsureEmbeddings()
        {
            if (this._embeddings != null)
            {
                return;
            }

            var embeddings = new List<VerseEmbedding>();
            foreach (var surah in this._surahs)
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

            foreach (var book in this._hadithBooks)
            {
                var bookTitle = book.Metadata?.English?.Title ?? $"Book {book.Id}";
                foreach (var hadith in book.Hadiths)
                {
                    var narrator = hadith.English?.Narrator ?? string.Empty;
                    var hadithText = hadith.English?.Text ?? string.Empty;
                    var combined = $"{bookTitle} (Hadith {hadith.IdInBook}) - {narrator} {hadithText}";
                    embeddings.Add(new VerseEmbedding
                    {
                        Reference = $"hadith:{book.Id}:{hadith.IdInBook}",
                        Text = combined,
                        Vector = Embed(combined)
                    });
                }
            }

            this._embeddings = embeddings;
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

            var surah = this._surahs.FirstOrDefault(s => s.Id == surahId);
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

        private class HadithBook
        {
            [JsonProperty("id")]
            public int Id { get; set; }

            [JsonProperty("metadata")]
            public HadithMetadata? Metadata { get; set; }

            [JsonProperty("chapters")]
            public List<HadithChapter> Chapters { get; set; } = new();

            [JsonProperty("hadiths")]
            public List<HadithEntry> Hadiths { get; set; } = new();
        }

        private class HadithMetadata
        {
            [JsonProperty("id")]
            public int Id { get; set; }

            [JsonProperty("length")]
            public int Length { get; set; }

            [JsonProperty("arabic")]
            public HadithLocalizedMeta? Arabic { get; set; }

            [JsonProperty("english")]
            public HadithLocalizedMeta? English { get; set; }
        }

        private class HadithLocalizedMeta
        {
            [JsonProperty("title")]
            public string Title { get; set; } = string.Empty;

            [JsonProperty("author")]
            public string Author { get; set; } = string.Empty;

            [JsonProperty("introduction")]
            public string Introduction { get; set; } = string.Empty;
        }

        private class HadithChapter
        {
            [JsonProperty("id")]
            public int? Id { get; set; }

            [JsonProperty("bookId")]
            public int BookId { get; set; }

            [JsonProperty("arabic")]
            public string Arabic { get; set; } = string.Empty;

            [JsonProperty("english")]
            public string English { get; set; } = string.Empty;
        }

        private class HadithEntry
        {
            [JsonProperty("id")]
            public int Id { get; set; }

            [JsonProperty("idInBook")]
            public int IdInBook { get; set; }

            [JsonProperty("chapterId")]
            public int? ChapterId { get; set; } // nullable

            [JsonProperty("bookId")]
            public int BookId { get; set; }

            [JsonProperty("arabic")]
            public string Arabic { get; set; } = string.Empty;

            [JsonProperty("english")]
            public HadithEnglish? English { get; set; }
        }

        private class HadithEnglish
        {
            [JsonProperty("narrator")]
            public string Narrator { get; set; } = string.Empty;

            [JsonProperty("text")]
            public string Text { get; set; } = string.Empty;
        }

        public string GetReferenceText(string command)
        {
            if (string.IsNullOrWhiteSpace(command))
            {
                return "Ungültiger Befehl.";
            }

            command = command.Trim();
            if (command.Contains(':'))
            {
                // ayah or ayah-range: surah:aya or surah:aya1-aya2
                var parts = command.Split(':');
                if (parts.Length != 2 || !int.TryParse(parts[0], out var surahId))
                {
                    return "Ungültiges Ayah-Format.";
                }

                var surah = this._surahs.FirstOrDefault(s => s.Id == surahId);
                if (surah == null)
                {
                    return "Sure nicht gefunden.";
                }

                var ayaPart = parts[1];
                if (ayaPart.Contains('-'))
                {
                    var range = ayaPart.Split('-');
                    if (range.Length == 2 && int.TryParse(range[0], out var start) && int.TryParse(range[1], out var end))
                    {
                        return this.BuildSurahRange(surah, start, end);
                    }

                    return "Ungültiger Ayah-Bereich.";
                }

                if (!int.TryParse(ayaPart, out var ayaId))
                {
                    return "Ungültige Ayah.";
                }

                return this.BuildSurahRange(surah, ayaId, ayaId);
            }

            // surah or surah-range: 23 or 23-24
            if (command.Contains('-'))
            {
                var range = command.Split('-');
                if (range.Length == 2 && int.TryParse(range[0], out var start) && int.TryParse(range[1], out var end))
                {
                    var sb = new StringBuilder();
                    for (var id = start; id <= end; id++)
                    {
                        var surah = this._surahs.FirstOrDefault(s => s.Id == id);
                        if (surah != null)
                        {
                            sb.AppendLine(this.BuildSurahRange(surah, 1, surah.Verses.Count)).AppendLine();
                        }
                    }

                    return sb.Length > 0 ? sb.ToString().TrimEnd() : "Suren nicht gefunden.";
                }

                return "Ungültiger Suren-Bereich.";
            }

            if (!int.TryParse(command, out var surahOnly))
            {
                return "Ungültige Sure.";
            }

            var target = this._surahs.FirstOrDefault(s => s.Id == surahOnly);
            if (target == null)
            {
                return "Sure nicht gefunden.";
            }

            return this.BuildSurahRange(target, 1, target.Verses.Count);
        }

        private string BuildSurahRange(Surah surah, int startAya, int endAya)
        {
            var sb = new StringBuilder();
            sb.Append("Sure ")
              .Append(surah.Id)
              .Append(" | ")
              .AppendLine(surah.Transliteration);

            var ordered = surah.Verses.OrderBy(v => v.Id);
            foreach (var verse in ordered)
            {
                if (verse.Id < startAya || verse.Id > endAya)
                {
                    continue;
                }

                sb.Append(surah.Id).Append(":").Append(verse.Id).AppendLine();
                sb.AppendLine(verse.Text);
                sb.AppendLine(verse.Translation);
                sb.AppendLine();
            }

            return sb.ToString().TrimEnd();
        }
    }
}
