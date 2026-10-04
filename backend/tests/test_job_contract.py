import json
import tempfile
import unittest
from pathlib import Path
from worker import timeline, checked_chart, atomic_json
from hybrid_pjsk.genelive_settings import preset_settings

class JobContractTests(unittest.TestCase):
    def test_music_duration_excludes_padding_and_keeps_tail_buffer(self):
        values = timeline(120, 9)
        self.assertEqual(values['audioDurationSec'], 129)
        self.assertEqual(values['secForMusicScoreMaker'], 122)
        self.assertEqual(values['previewStartTimeSec'], 9)
    def test_fractional_padding_and_preview(self):
        values = timeline(6.125, 9, 2)
        self.assertEqual(values['audioDurationSec'], 15.125)
        self.assertEqual(values['secForMusicScoreMaker'], 9)
        self.assertEqual(values['previewStartTimeSec'], 11)
    def test_rejects_nonfinite_and_out_of_range_timeline(self):
        for args in ((float('nan'), 9), (120, float('inf')), (0, 9), (120, -1), (120, 121), (120, 9, 121)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                timeline(*args)
    def test_five_difficulties_match_existing_tool(self):
        for i, name in enumerate(('EASY', 'NORMAL', 'HARD', 'EXPERT', 'MASTER'), 1):
            self.assertEqual(preset_settings(name)['condition'], i * 10)
        self.assertEqual(preset_settings('MASTER')['snap_division'], 32)
        self.assertEqual(preset_settings('EXPERT')['snap_division'], 16)
    def test_failed_actual_export_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / 'chart.json'
            data = {'two_finger': {'exported_verification': {'passed': True, 'visual_overlap_checked': True, 'visual_overlap_count': 1}}}
            atomic_json(file, data)
            with self.assertRaises(ValueError): checked_chart(file)
    def test_grid_failure_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / 'chart.json'
            data = {'two_finger': {'exported_verification': {'passed': True, 'visual_overlap_checked': True, 'visual_overlap_count': 0}},
                    'genelive_postprocessing': {'snap_division': 16},
                    'sus_preview': {'grid_verification': {'passed': False, 'off_grid_count': 1}}}
            atomic_json(file, data)
            with self.assertRaises(ValueError): checked_chart(file)

if __name__ == '__main__': unittest.main()
