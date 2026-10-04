"""Layout input features shared by Torch and the portable ONNX runtime."""
import numpy as np
from mapper_pjsk.schema import KINDS,DIRECTIONS

INTERVALS = [(lane,width) for lane in range(12) for width in range(1,13-lane)]


def features(events,bpm=143.,offset=0.,tempos=None):
    times=np.array([e['time'] for e in events]); n=len(times)
    if tempos:
        origin=np.array([t['time'] for t in tempos]); idx=np.maximum(np.searchsorted(origin,times,side='right')-1,0)
        beats=np.array([tempos[i]['beat'] for i in idx])+(times-origin[idx])*np.array([tempos[i]['bpm'] for i in idx])/60
        speed=np.array([tempos[i]['bpm'] for i in idx])
    else:
        beats=(times-offset)*bpm/60; speed=np.full(n,bpm)
    before=np.r_[0,np.diff(times)]; after=np.r_[np.diff(times),0]
    x=np.zeros((n,len(KINDS)+len(DIRECTIONS)+8),np.float32)
    for i,e in enumerate(events):
        x[i,KINDS.index(e['kind'])]=1; x[i,len(KINDS)+DIRECTIONS.index(e['direction'])]=1
        x[i,-8:]=[e['critical'],e['trace'],min(before[i]*speed[i]/60,8)/8,
                  min(after[i]*speed[i]/60,8)/8,np.sin(beats[i]*np.pi/2),np.cos(beats[i]*np.pi/2),
                  np.sin(beats[i]*np.pi*2),np.cos(beats[i]*np.pi*2)]
    return x
