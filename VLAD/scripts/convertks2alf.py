""" 
Script to find all sessions that have not been converted from kilosort to alf and convert them
Be sure to update the ONE cache before running
"""
import sys
sys.path.append("../")
from one.api import One
from utils import sessions_to_include
import cibrrig.postprocess.convert_ks_to_alf as convert_ks2_to_alf
if sys.platform == 'linux':
    cache_dir = '/data/rss/helens/ramirez_j/ramirezlab/alf_data_repo'
else:
    cache_dir = r'X:\alf_data_repo'

# List all subjects
subjects = sessions_to_include['subject'].values

# Initialize ONE
one = One(cache_dir=cache_dir)

# get sessions with spikes
eids_spikes = one.search(subject=subjects,datasets='clusters.*')
print(len(eids_spikes))

# get all sessions
eids_all = one.search(subject=subjects)

# subtract sessions with spikes from all sessions
eids_convert = list(set(eids_all) - set(eids_spikes))

# Print the sessions for the user
for eid in eids_convert:
    print(one.eid2ref(eid))

# Convert the remaining sessions
for eid in eids_convert:
    session_path = one.eid2path(eid)
    try:
        convert_ks2_to_alf.run_session(session_path,'kilosort4')
    except:
        print('='*80)
        print(f'Failed to convert {session_path}')
        print('='*80)
        continue

