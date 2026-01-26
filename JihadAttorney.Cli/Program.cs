using System;
using System.Linq;
using System.Threading.Tasks;
using JihadAttorney.Llama;

namespace JihadAttorney.Cli
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;

            var llama = new LlamaService();
            Console.WriteLine("Suche Modelle in D:\\Models ...");
            var models = llama.GetAvailableModels();
            if (!models.Any())
            {
                Console.WriteLine("Keine gguf-Modelle gefunden unter D:\\Models.");
                return;
            }

            for (var i = 0; i < models.Count; i++)
            {
                Console.WriteLine($"[{i}] {models[i]}");
            }

            int selectedIndex;
            while (true)
            {
                Console.Write("Modell-Nummer wählen: ");
                var input = Console.ReadLine();
                if (int.TryParse(input, out selectedIndex) && llama.SelectModelByIndex(selectedIndex))
                {
                    Console.WriteLine($"Gewählt: {models[selectedIndex]}");
                    break;
                }

                Console.WriteLine("Ungültige Auswahl, bitte erneut versuchen.");
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
                    var answer = await llama.AnswerQuestionAsync(question.Trim());
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
    }
}
