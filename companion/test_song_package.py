import tempfile
import unittest
from pathlib import Path
import zipfile
from companion.song_package import score_from_sus,extract_existing,music_id

HEADER='#00002: 4\n#BPM01: 120\n#00008: 01\n'


class PackageContract(unittest.TestCase):
    def test_connected_flick_end_and_hidden_control(self):
        text=HEADER+'#00012: 23000000\n#00013: 00003300\n#00014: 00000013\n#000320: 13000000\n#000330: 00005300\n#000340: 00000023\n#00054: 00000033\n'
        score=score_from_sus(text)
        self.assertEqual([note['ticks'] for note in score['NoteList']],[0,960,1440])
        first,middle,last=score['NoteList']
        self.assertEqual((first['category'],first['noteBaseType'],first['type']),(1,2,1))
        self.assertEqual((middle['category'],middle['noteBaseType'],middle['isSkip']),(13,6,True))
        self.assertEqual((last['category'],last['noteBaseType'],last['direction']),(3,3,1))
        self.assertEqual(first['nextConnectionId'],middle['id'])
        self.assertEqual(middle['nextConnectionId'],last['id'])
        self.assertEqual(last['previousConnectionId'],middle['id'])

    def test_friction_long_and_critical_single(self):
        score=score_from_sus(HEADER+'#00012: 63\n#000320: 13000023\n')
        self.assertEqual(score['NoteList'][0]['category'],6)
        self.assertEqual(score['NoteList'][0]['noteBaseType'],8)
        self.assertEqual(score['NoteList'][0]['type'],1)

    def test_refuse_unclosed_chain(self):
        with self.assertRaises(ValueError):score_from_sus(HEADER+'#000320: 13\n')

    def test_refuse_zip_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            with zipfile.ZipFile(root/'bad.zip','w') as archive:archive.writestr('../outside.txt','bad')
            with self.assertRaises(ValueError):extract_existing(root/'bad.zip',root/'extracted')
            self.assertFalse((root/'outside.txt').exists())

    def test_game_identifier_is_negative_and_case_independent(self):
        self.assertLess(music_id('abc123'),0)
        self.assertEqual(music_id('ABC123'),music_id('abc123'))


if __name__=='__main__':unittest.main()
