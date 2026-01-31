from pynwb import NWBHDF5IO

path = "/Users/saru/Local_Work/rnn_winter_project/000129/sub-Indy/sub-Indy_desc-train_behavior+ecephys.nwb"

io = NWBHDF5IO(path, "r")
nwb = io.read()

def describe_trials(nwb):
    if getattr(nwb, "trials", None) is None:
        print("No nwb.trials table found.")
        return
    tbl = nwb.trials
    print("Trials columns:")
    for c in tbl.colnames:
        print(" -", c)

def describe_processing(nwb):
    print("Processing modules:", list(nwb.processing.keys()))
    for k, mod in nwb.processing.items():
        print(f"\n== processing['{k}'] ==")
        try:
            print("data_interfaces:", list(mod.data_interfaces.keys()))
        except Exception as e:
            print("couldn't list data_interfaces:", e)

def describe_acquisition(nwb):
    print("Acquisition:", list(nwb.acquisition.keys()))
    for k in list(nwb.acquisition.keys())[:20]:
        obj = nwb.acquisition[k]
        print(" -", k, type(obj))

describe_trials(nwb)
describe_processing(nwb)
describe_acquisition(nwb)