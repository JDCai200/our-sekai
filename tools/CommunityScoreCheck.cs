// Deserialize using an installed community player's actual assemblies.
// This does not need Unity, and does not modify the player's files.
using System;
using System.IO;
using System.Reflection;
using System.Runtime.CompilerServices;

public static class CommunityScoreCheck
{
    public static int Main(string[] args)
    {
        string managed = Path.GetFullPath(args[0]);
        AppDomain.CurrentDomain.AssemblyResolve += (sender, eventArgs) => {
            string path = Path.Combine(managed, new AssemblyName(eventArgs.Name).Name + ".dll");
            return File.Exists(path) ? Assembly.LoadFrom(path) : null;
        };
        Validate(args[1]);
        return 0;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Validate(string file)
    {
        var data = Sekai.MusicScoreMaker.Ingame.Utilities.DeepCopyHelper.FromJson<Sekai.MusicScoreMaker.Ingame.Models.MusicScoreMakerData>(File.ReadAllText(file));
        data.InitializeIdCount();
        if (data.VersionCode != 1 || data.NoteList.Count == 0 || data.MusicScoreEventDataList.Count != 4)
            throw new Exception("Invalid community score");
        var manifest = Newtonsoft.Json.JsonConvert.DeserializeObject<Sekai.MusicScoreMaker.Common.CustomMusicScoreManifest>(File.ReadAllText(Path.Combine(Path.GetDirectoryName(file),"manifest.json")));
        var entry = new Sekai.MusicScoreMaker.Common.CustomMusicScoreEntry(Path.GetDirectoryName(file),manifest);
        if (entry.MusicId != data.MusicId) throw new Exception("Player music ID differs from exported score");
        Console.WriteLine("Community deserialization passed: " + data.NoteList.Count + " note structures");
    }
}
