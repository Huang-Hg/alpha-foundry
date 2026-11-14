"""Research clock and publication-time joins. All bounds are inclusive session dates.

Daily row t is observed at session close. Its target can first execute at the next
session open. Event rows carry both their effective date and publication date;
later publications never rewrite an earlier decision row.
"""
import numpy as np


def event_grid(dates, assets, records):
    """Publication-aware state for (symbol, effective, published, value) records.

    A newly published correction replaces the same effective observation only
    from publication onward. An older observation never displaces a newer one.
    Dates and events are session-close dates, not intraday timestamps.
    """
    out=np.full((len(dates),len(assets)),np.nan)
    columns={symbol:i for i,symbol in enumerate(assets)}
    events=sorted(records,key=lambda r:(max(r['effective'],r['published']),r['published'],r['effective']))
    state=np.full(len(assets),np.nan)
    latest=np.full(len(assets),np.datetime64('0001-01-01','D'))
    i=0
    for t,date in enumerate(dates):
        while i<len(events) and np.datetime64(max(events[i]['effective'],events[i]['published']))<=date:
            row=events[i];column=columns[row['symbol']]
            effective=np.datetime64(row['effective'],'D')
            if effective>=latest[column]:
                state[column]=row['value'];latest[column]=effective
            i+=1
        out[t]=state
    return out


def membership(dates,assets,intervals):
    """Effective interval [start,end); membership is observed at that day's close.

    A null end means there has been no exit event. An exit after the study cutoff
    cannot change earlier membership. Membership is distinct from quote presence.
    """
    mask=np.zeros((len(dates),len(assets)),dtype=bool)
    columns={symbol:i for i,symbol in enumerate(assets)}
    for row in intervals:
        end=np.datetime64('9999-12-31') if row['end'] is None else np.datetime64(row['end'],'D')
        if row['symbol'] in columns:
            mask[:,columns[row['symbol']]]|=(dates>=np.datetime64(row['start'],'D'))&(dates<end)
    return mask
