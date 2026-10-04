"""Verify actual exported judgment positions, not the pre-export audio grid."""
from mapper_pjsk.schema import parse_position_free


def verify_sus_grid(text,bpm,division):
    if division not in (8,16,32):raise ValueError('Invalid judgment grid division')
    step_ticks=1920//division
    judgments=[e for e in parse_position_free(text)['events']
               if e['kind'] not in ('slide_hidden','slide_start_hidden','slide_end_hidden')]
    errors=[]
    for e in judgments:
        tick=round(e['time']*bpm/60*480)
        if tick%step_ticks:errors.append(dict(time=e['time'],kind=e['kind'],tick=tick))
    return dict(passed=not errors,division=division,step_ticks=step_ticks,
                checked_judgments=len(judgments),off_grid_count=len(errors),off_grid=errors,
                reference='SUS measure zero; hidden nonjudgment shape controls excluded')
