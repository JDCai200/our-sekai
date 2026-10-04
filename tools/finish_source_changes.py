"""One-time source migration; kept as a readable record of removed editor routes."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / 'Assets/CustomMusicScoreManager/Runtime/UI/ScreenLayerCustomMusicScoreManager.cs'

def replace_method(text, signature, body):
    start = text.index(signature)
    opening = text.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[:opening + 1] + '\n' + body + '\n\t\t}' + text[end:]

def main():
    text = UI.read_text(encoding='utf-8')
    text = replace_method(text, 'private void ReplaceSelectedAudio()', '\t\t\tOpenGenerationDialog();')
    text = replace_method(text, 'private void OpenEditor()', '\t\t\tOpenGenerationForSelected();')
    text = replace_method(text, 'private void CloseSettingsAndReturnToEditor()', '\t\t\tCloseSettings();')
    text = replace_method(text, 'public static void OpenSettingsAfterReturnFromEditor()',
        '\t\t\tUnityEngine.Object.FindObjectOfType<ScreenLayerCustomMusicScoreManager>()?.OpenSettings();')
    text = replace_method(text, 'private void OpenEditorAfterSettingsClosed()', '\t\t\tCloseSettings();')
    # Prevent timing edits from silently desynchronizing generated audio and notes.
    text = text.replace('SetFormInteractable(hasSelection);',
        'SetFormInteractable(hasSelection);\n\t\t\t_fillerInput.interactable = false;\n\t\t\t_durationInput.interactable = false;\n\t\t\t_audioInput.interactable = false;\n\t\t\t_scoreInput.interactable = false;')
    UI.write_text(text, encoding='utf-8')
    path = ROOT / 'Assets/Scripts/Assembly-CSharp/Sekai/ScreenManager.cs'
    text = path.read_text(encoding='utf-8')
    signature = 'public void PushUIScreen(MenuScreenType screenType, ScreenLayer.BootArgBase bootArg, bool isWaitExitAnimation = false)'
    point = text.index('{', text.index(signature)) + 1
    text = text[:point] + '\n\t\t\t// All legacy editor navigation returns to the playable song library.\n\t\t\tif (screenType == MenuScreenType.MusicScoreMaker)\n\t\t\t{\n\t\t\t\tscreenType = MenuScreenType.MusicScoreMakerTop;\n\t\t\t\tbootArg = null;\n\t\t\t}\n' + text[point:]
    path.write_text(text, encoding='utf-8')
    path = ROOT / 'Assets/Scripts/Assembly-CSharp/Sekai/MusicScoreMaker/Ingame/Presenters/MusicScoreMakerEntryPoint.cs'
    text = path.read_text(encoding='utf-8').replace('OtaChecker.Instance.CheckForUpdates();', '// Community OTA updates do not apply to Our Sekai; startup remains offline.')
    path.write_text(text, encoding='utf-8')
    path = ROOT / 'ProjectSettings/ProjectSettings.asset'
    text = path.read_text(encoding='utf-8').replace('companyName: JsoftStudio', 'companyName: OurSekai').replace('productName: Ojsk Community', 'productName: Our Sekai').replace('Android: top.jsoftstudio.ojsk', 'Android: org.oursekai.game').replace('Standalone: com.cube.opensekai', 'Standalone: org.oursekai.game')
    path.write_text(text, encoding='utf-8')

if __name__ == '__main__':
    main()
