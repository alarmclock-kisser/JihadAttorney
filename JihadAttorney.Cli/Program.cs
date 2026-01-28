using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using JihadAttorney.Llama;
using Microsoft.Extensions.Configuration;

namespace JihadAttorney.Cli
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            var configuration = new ConfigurationBuilder()
                .SetBasePath(AppContext.BaseDirectory)
                .AddJsonFile("appsettings.json", optional: true, reloadOnChange: false)
                .Build();

            var responseLanguage = configuration["ResponseLanguage"]?.Trim();
            if (string.IsNullOrWhiteSpace(responseLanguage))
            {
                responseLanguage = "auto";
            }

            var preferredModel = configuration["PreferredModel"]?.Trim() ?? string.Empty;

            var llama = new LlamaService();
            Console.WriteLine("Suche Modelle in D:\\Models ...");
            var models = llama.GetAvailableModels();
            if (!models.Any())
            {
                Console.WriteLine("Keine gguf-Modelle gefunden unter D:\\Models.");
                return;
            }

            int? selectedIndex = null;

            if (!string.IsNullOrWhiteSpace(preferredModel))
            {
                var preferredIndex = FindPreferredModelIndex(preferredModel, models);
                if (preferredIndex >= 0 && llama.SelectModelByIndex(preferredIndex))
                {
                    selectedIndex = preferredIndex;
                    Console.WriteLine($"Bevorzugtes Modell geladen: {models[preferredIndex]}");
                }
                else
                {
                    Console.WriteLine($"Bevorzugtes Modell \"{preferredModel}\" nicht gefunden. Bitte wähle aus der Liste.");
                }
            }

            if (!selectedIndex.HasValue)
            {
                for (var i = 0; i < models.Count; i++)
                {
                    Console.WriteLine($"[{i}] {models[i]}");
                }

                while (true)
                {
                    Console.Write("Modell-Nummer wählen: ");
                    var input = Console.ReadLine();
                    if (int.TryParse(input, out var manualIndex) && llama.SelectModelByIndex(manualIndex))
                    {
                        selectedIndex = manualIndex;
                        Console.WriteLine($"Gewählt: {models[manualIndex]}");
                        break;
                    }

                    Console.WriteLine("Ungültige Auswahl, bitte erneut versuchen.");
                }
            }

            Console.WriteLine("Lade Modell und bereite Embeddings vor (Warmup)...");
            try
            {
                await llama.PrepareAsync();
                Console.WriteLine("Bereit. Du kannst jetzt Fragen stellen.");
                Console.WriteLine("Commands: /<sura>, /<sura>-<sura>, /<sura>:<ayah>, /<sura>:<start-end>");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Fehler beim Laden/Warmup: {ex.Message}");
                Console.WriteLine("Bitte stelle sicher, dass die passende llama DLL zur LLamaSharp-Version vorhanden ist.");
                return;
            }

            Console.WriteLine();
            Console.WriteLine("Frage eingeben (leer oder 'exit' zum Beenden):");

            while (true)
            {
                Console.Write("?> ");
                var question = Console.ReadLine();
                if (string.IsNullOrWhiteSpace(question) || question.Equals("exit", StringComparison.OrdinalIgnoreCase))
                {
                    break;
                }

                if (question.StartsWith("/"))
                {
                    var cmd = question.Substring(1);
                    var result = llama.GetReferenceText(cmd);
                    Console.WriteLine(result);
                    Console.WriteLine();
                    continue;
                }

                try
                {
                    var answer = await llama.AnswerQuestionAsync(question.Trim(), responseLanguage);
                    Console.WriteLine();
                    Console.WriteLine("Antwort:");
                    Console.WriteLine(answer);
                    Console.WriteLine();
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Fehler: {ex.Message}");
                }
            }

        }

        private static int FindPreferredModelIndex(string preferredModel, IReadOnlyList<string> models)
        {
            var preferredFileName = Path.GetFileName(preferredModel);

            for (var i = 0; i < models.Count; i++)
            {
                if (string.Equals(models[i], preferredModel, StringComparison.OrdinalIgnoreCase))
                {
                    return i;
                }

                if (!string.IsNullOrWhiteSpace(preferredFileName) &&
                    string.Equals(Path.GetFileName(models[i]), preferredFileName, StringComparison.OrdinalIgnoreCase))
                {
                    return i;
                }
            }

            return -1;
        }
    }
}
