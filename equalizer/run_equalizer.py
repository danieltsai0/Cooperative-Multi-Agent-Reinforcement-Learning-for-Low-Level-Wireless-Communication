from Data_generator import Data_generator
import itertools
import lstm_equalizer

def make_res_path(mod, dl):
    # compose path 'equalizer_results/mod_dl'
    return "equalizer_results/" + \
           str(mod) + "_" + str(dl) 
def make_model_path(mod, dl):
    return "equalizer_models/" + \
           str(mod) + "_" + str(dl) + ".ckpt"

# generate test parameters
constellations = Data_generator().constellations.keys()
delay_lengths = list(range(2, 40, 3))

for (constellation, delay) in itertools.product(constellations, delay_lengths):
    # write filenames for output data and results
    res_path = make_res_path(constellation, delay)
    model_path = make_model_path(constellation, delay)

    print("running " + str(constellation) + "_" + str(delay) + " model")
    # run equalizer over all tests
    lstm_equalizer.main(constellation, delay, res_path, model_path)

# read in results
# plot results
# make gifs

# animate training error
# overlay validation error
